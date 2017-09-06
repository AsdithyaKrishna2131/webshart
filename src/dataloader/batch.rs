use crate::dataloader::file_loading::file_http_client;
use crate::discovery::{DatasetDiscovery, DiscoveredDataset};
use crate::error::{Result, WebshartError};
use futures::future::join_all;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Batch read request for a single file
#[derive(Debug, Clone)]
pub struct FileReadRequest {
    /// Dataset to read from
    pub dataset_idx: usize,
    /// Shard index within the dataset
    pub shard_idx: usize,
    /// File index within the shard
    pub file_idx: usize,
}

/// Result of a batch operation
#[derive(Debug)]
pub enum BatchResult<T> {
    Ok(T),
    Err(String),
}

/// Batch operations handler
pub struct BatchOperations {
    runtime: Arc<Runtime>,
}

impl BatchOperations {
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(Runtime::new().expect("Failed to create Tokio runtime")),
        }
    }

    pub fn with_runtime(runtime: Arc<Runtime>) -> Self {
        Self { runtime }
    }

    /// Discover multiple datasets in parallel
    pub fn discover_datasets_batch(
        &self,
        sources: Vec<String>,
        hf_token: Option<String>,
        subfolders: Option<Vec<Option<String>>>,
    ) -> Vec<BatchResult<DiscoveredDataset>> {
        let runtime = self.runtime.clone();

