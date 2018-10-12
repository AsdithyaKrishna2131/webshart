use crate::digest_to_hex;
use crate::error::{Result, WebshartError};
use crate::metadata::{FileInfo, ShardMetadata, ShardMetadataFormat};
use futures::future::join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn is_image_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();
    path_lower.ends_with(".png")
        || path_lower.ends_with(".jpg")
        || path_lower.ends_with(".jpeg")
        || path_lower.ends_with(".webp")
        || path_lower.ends_with(".tiff")
        || path_lower.ends_with(".tif")
        || path_lower.ends_with(".bmp")
        || path_lower.ends_with(".gif")
        || path_lower.ends_with(".ico")
        || path_lower.ends_with(".jxl")
        || path_lower.ends_with(".avif")
}

fn is_json_file(path: &str) -> bool {
    path.rsplit_once('.')
        .map(|(_, ext)| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

fn extract_image_dimensions(data: &[u8]) -> Option<(u32, u32, f32)> {
    match imagesize::blob_size(data) {
        Ok(size) => {
            let width = size.width as u32;
            let height = size.height as u32;
            let aspect = width as f32 / height as f32;
            Some((width, height, aspect))
        }
        Err(_) => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCheckpoint {
    pub shard_name: String,
    pub status: CheckpointStatus,
    pub offset: u64,
    pub files_processed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointStatus {
    Pending,
    InProgress,
    Complete,
    Failed(String),
}

#[derive(Debug, Clone)]
struct UnindexedShard {
    name: String,
    path: String,
    size: u64,
    is_remote: bool,
}

pub struct MetadataExtractor {
    runtime: Arc<Runtime>,
    hf_token: Option<String>,
    client: reqwest::Client,
    shard_pattern: Regex,
    compute_sha256: bool,
    include_image_geometry: bool,
}

impl MetadataExtractor {
    pub fn new(hf_token: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // Increase timeout for large files
            .connect_timeout(std::time::Duration::from_secs(30))
            .pool_idle_timeout(std::time::Duration::from_secs(30))
            .pool_max_idle_per_host(1) // Reduce connection pool to save memory
            .build()
            .expect("Failed to build HTTP client");

        // Create runtime without complex signal handling in thread start
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("webshart-worker")
            .build()
            .expect("Failed to create runtime");

        Self {
            runtime: Arc::new(runtime),
            hf_token,
            client,
            shard_pattern: Regex::new(r"^(.+?)\.tar$").unwrap(),
            compute_sha256: false,
            include_image_geometry: false,
        }
    }

    pub fn with_sha256(mut self, compute: bool) -> Self {
        self.compute_sha256 = compute;
        self
    }

    pub fn with_image_geometry(mut self, include: bool) -> Self {
        self.include_image_geometry = include;
