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
