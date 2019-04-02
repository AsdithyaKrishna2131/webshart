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

