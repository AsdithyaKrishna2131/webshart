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
            // Default: co-located with tar
            if is_remote {
                tar_path
                    .strip_suffix(".tar")
                    .map(|path| format!("{path}.json"))
                    .unwrap_or_else(|| format!("{tar_path}.json"))
            } else {
                Path::new(tar_path)
                    .with_extension("json")
                    .to_string_lossy()
                    .to_string()
            }
        }
    }

    pub fn get_source(&self) -> Option<String> {
        self.metadata_source.clone()
    }

    #[cfg(test)]
    pub(crate) fn get_hf_token(&self) -> Option<&str> {
        self.hf_token.as_deref()
    }

    /// Extract subfolder from tar path
    fn extract_subfolder(&self, tar_path: &str, is_remote: bool) -> Option<String> {
        if is_remote {
            // For URLs like: https://huggingface.co/datasets/repo/resolve/main/subfolder/file.tar
            // Extract "subfolder" part
            if let Some(pos) = tar_path.find("/resolve/main/") {
                let after_main = &tar_path[pos + 14..]; // Skip "/resolve/main/"
                if let Some(last_slash) = after_main.rfind('/') {
                    if last_slash > 0 {
                        return Some(after_main[..last_slash].to_string());
                    }
