use crate::error::{Result, WebshartError};
use crate::metadata::ShardMetadata;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[derive(Debug, Clone)]
pub struct MetadataResolver {
    /// Optional separate metadata location (local path or HF repo)
    metadata_source: Option<String>,
    /// HF token for accessing metadata
    hf_token: Option<String>,
    /// Client for HTTP requests
    client: reqwest::Client,
    /// Runtime for async operations
    runtime: Arc<Runtime>,
}

impl MetadataResolver {
    fn source_is_hub_repo(source: &str) -> bool {
        !source.starts_with("http")
            && !Path::new(source).exists()
            && !Path::new(source).is_absolute()
            && !source.starts_with('.')
            && source.split('/').filter(|part| !part.is_empty()).count() == 2
    }

    pub fn new(
        metadata_source: Option<String>,
        hf_token: Option<String>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            metadata_source,
            hf_token,
            client: reqwest::Client::new(),
            runtime,
        }
    }

    /// Resolve metadata location for a given tar file
    pub fn resolve_metadata_path(
        &self,
        tar_path: &str,
        base_name: &str,
        is_remote: bool,
    ) -> String {
        if let Some(metadata_source) = &self.metadata_source {
            // Extract subfolder path from tar_path
            let subfolder = self.extract_subfolder(tar_path, is_remote);

            if Self::source_is_hub_repo(metadata_source) {
                let base_url = format!(
                    "https://huggingface.co/datasets/{}/resolve/main",
                    metadata_source
                );
                if is_remote {
                    if let Some(sub) = subfolder {
                        format!("{}/{}/{}.json", base_url, sub, base_name)
                    } else {
                        format!("{}/{}.json", base_url, base_name)
                    }
                } else {
                    format!("{}/{}.json", base_url, base_name)
                }
            } else if metadata_source.starts_with("http") {
                let base_url = metadata_source.trim_end_matches('/');
                if is_remote {
                    if let Some(sub) = subfolder {
                        format!("{}/{}/{}.json", base_url, sub, base_name)
                    } else {
                        format!("{}/{}.json", base_url, base_name)
                    }
                } else {
                    format!("{}/{}.json", base_url, base_name)
                }
            } else {
                let mut path = Path::new(metadata_source).to_path_buf();
                if is_remote {
                    if let Some(sub) = subfolder {
                        path = path.join(sub);
                    }
                }
                path.join(format!("{}.json", base_name))
                    .to_string_lossy()
                    .to_string()
            }
        } else {
