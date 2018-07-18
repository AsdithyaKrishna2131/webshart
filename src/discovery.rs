use crate::dataloader::shard_cache::ShardCache;
use crate::error::{Result, WebshartError};
use crate::metadata::ShardMetadata;
use crate::metadata_resolver::MetadataResolver;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Represents a discovered shard pair (tar + json)
#[derive(Debug, Clone)]
pub struct ShardPair {
    /// Base name without extension (e.g., "data-0000")
    pub name: String,
    /// Path/URL to the tar file
    pub tar_path: String,
    /// Path/URL to the json file
    pub json_path: String,
    /// Loaded metadata (lazy loaded)
    pub metadata: Option<ShardMetadata>,
}

/// Represents a discovered dataset with all its shards
#[derive(Debug, Clone)]
pub struct DiscoveredDataset {
    pub name: String,
    pub subfolder: Option<String>,
    pub is_remote: bool,
    pub shards: Vec<ShardPair>,
    discovery_token: Option<String>,
    cached_total_size: Option<u64>,
    cached_total_files: Option<usize>,
    pub metadata_source: Option<String>,
    runtime: Arc<Runtime>,
    /// Optional cache directory for metadata
    cache_dir: Option<PathBuf>,
    /// Track if we've hit rate limits
    rate_limit_delay: Option<Duration>,
    pub shard_cache: Option<Arc<ShardCache>>,
}

impl DiscoveredDataset {
    /// Get total number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Get the HuggingFace token if available
    pub fn get_hf_token(&self) -> Option<String> {
        self.discovery_token.clone()
    }

    pub(crate) fn metadata_cache_dir(&self) -> Option<PathBuf> {
        self.cache_dir.clone()
    }

    pub(crate) fn replace_shard_metadata(
        &mut self,
        shard_index: usize,
        metadata: ShardMetadata,
        persist_to_cache: bool,
    ) -> Result<()> {
        let shard_name = self
            .shards
            .get(shard_index)
            .ok_or_else(|| {
                WebshartError::InvalidShardFormat(format!(
                    "Shard index {} out of range",
                    shard_index
                ))
            })?
            .name
            .clone();
        if persist_to_cache {
            self.save_metadata_to_cache(&shard_name, &metadata)?;
        }
        self.shards[shard_index].metadata = Some(metadata);
        Ok(())
    }

    pub async fn enable_shard_cache(
        &mut self,
        location: PathBuf,
        cache_limit_gb: f64,
        parallel_downloads: usize,
    ) -> Result<()> {
        let mut cache = ShardCache::new(location, cache_limit_gb, parallel_downloads);
        cache.ensure_cache_dir().await?;
        cache.initialize_from_disk().await?;
        self.shard_cache = Some(Arc::new(cache));
        Ok(())
    }

    /// Enable metadata caching and optionally pre-load some shards
    pub fn enable_metadata_cache(
        &mut self,
        cache_location: &str,
        init_shard_count: usize,
    ) -> Result<()> {
        let cache_path = PathBuf::from(cache_location);

        // Create cache directory if it doesn't exist
        if !cache_path.exists() {
            fs::create_dir_all(&cache_path)?;
        }

        // Create a subdirectory for this dataset
        let dataset_cache = cache_path.join(self.get_cache_key());
        if !dataset_cache.exists() {
            fs::create_dir_all(&dataset_cache)?;
        }

        self.cache_dir = Some(dataset_cache);

        // Pre-load initial shards
        if init_shard_count > 0 {
            println!(
                "[webshart] Pre-loading {} shard metadata...",
                init_shard_count.min(self.shards.len())
            );
            let count = init_shard_count.min(self.shards.len());

            for i in 0..count {
                match self.ensure_shard_metadata(i) {
                    Ok(_) => {
                        println!("[webshart] Loaded metadata for shard {}/{}", i + 1, count);
                    }
                    Err(e) => {
                        eprintln!(
                            "[webshart] Warning: Failed to load metadata for shard {}: {}",
                            i, e
                        );
                        // Check if it's a rate limit error
                        if let WebshartError::RateLimited = e {
                            // Stop pre-loading if we hit rate limit
                            println!("[webshart] Rate limited, stopping pre-load");
                            break;
                        }
                    }
                }

                // Add small delay between requests to be nice to the server
                if i < count - 1 && self.is_remote {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        Ok(())
    }

    /// Generate a cache key for this dataset
    fn get_cache_key(&self) -> String {
        let mut key = self.name.replace('/', "_");
        if let Some(subfolder) = &self.subfolder {
            key.push_str("__");
            key.push_str(&subfolder.replace('/', "_"));
        }
        key
    }

    /// Get cached metadata path for a shard
    fn get_cached_metadata_path(&self, shard_name: &str) -> Option<PathBuf> {
        self.cache_dir
            .as_ref()
            .map(|dir| dir.join(format!("{}.json", shard_name)))
    }

    /// Load metadata from cache if available
    fn load_cached_metadata(&self, shard_name: &str) -> Option<ShardMetadata> {
        let cache_path = self.get_cached_metadata_path(shard_name)?;

        if cache_path.exists() {
            match fs::read_to_string(&cache_path) {
                Ok(content) => {
                    match serde_json::from_str(&content) {
                        Ok(metadata) => {
                            println!("[webshart] Loaded metadata for {} from cache", shard_name);
                            Some(metadata)
                        }
                        Err(e) => {
                            eprintln!(
                                "[webshart] Failed to parse cached metadata for {}: {}",
                                shard_name, e
                            );
                            // Remove corrupted cache file
                            let _ = fs::remove_file(&cache_path);
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[webshart] Failed to read cached metadata for {}: {}",
                        shard_name, e
                    );
                    None
                }
            }
        } else {
            None
        }
    }

    /// Save metadata to cache
    fn save_metadata_to_cache(&self, shard_name: &str, metadata: &ShardMetadata) -> Result<()> {
        if let Some(cache_path) = self.get_cached_metadata_path(shard_name) {
            if let Some(parent) = cache_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let json = serde_json::to_string_pretty(metadata)?;
            fs::write(&cache_path, json)?;
            println!("[webshart] Cached metadata for {}", shard_name);
        }
        Ok(())
    }

    /// Get total number of files (requires loading all metadata)
    pub fn total_files(&mut self) -> Result<usize> {
        self.ensure_all_metadata_loaded()?;
        Ok(self
            .shards
            .iter()
            .filter_map(|s| s.metadata.as_ref())
            .map(|m| m.num_files())
            .sum())
    }

    /// Get total size (requires loading all metadata)
    pub fn total_size(&mut self) -> Result<u64> {
        self.ensure_all_metadata_loaded()?;
        Ok(self
            .shards
            .iter()
            .filter_map(|s| s.metadata.as_ref())
            .map(|m| m.filesize)
            .sum())
    }

    /// Get quick stats from cached values (instant, no metadata loading)
    pub fn quick_stats(&self) -> (Option<u64>, Option<usize>) {
        (self.cached_total_size, self.cached_total_files)
    }

    /// Ensure metadata is loaded for a specific shard
    pub fn ensure_shard_metadata(&mut self, shard_index: usize) -> Result<()> {
        // Scope the first borrow to check if metadata is already loaded
        if let Some(shard) = self.shards.get(shard_index) {
            if shard.metadata.is_some() {
                return Ok(());
            }
        } else {
            return Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range for shard {}",
                shard_index, self.name
            )));
        }

        // Clone shard info to release the borrow on `self`
        let (shard_name, json_path) = {
            let shard = &self.shards[shard_index];
            (shard.name.clone(), shard.json_path.clone())
        };

        // First, check cache
        if let Some(cached) = self.load_cached_metadata(&shard_name) {
            if let Some(shard) = self.shards.get_mut(shard_index) {
                shard.metadata = Some(cached);
            }
            return Ok(());
        }

        // Apply rate limit delay if needed
        if let Some(delay) = self.rate_limit_delay {
            println!("[webshart] Rate limit delay: {:?}", delay);
            std::thread::sleep(delay);
            // Reset delay after using it
            self.rate_limit_delay = None;
        }

        let discovery = DatasetDiscovery::with_runtime(self.runtime.clone())
            .with_optional_token(self.discovery_token.clone());

        // Use block_in_place to avoid blocking the async runtime
        let result = tokio::task::block_in_place(|| {
            if json_path.starts_with("http") {
                self.runtime
                    .block_on(discovery.load_remote_metadata(&json_path))
            } else {
                self.runtime
                    .block_on(discovery.load_local_metadata(&json_path))
            }
        });

        match result {
            Ok(metadata) => {
                // Save to cache if caching is enabled
                let _ = self.save_metadata_to_cache(&shard_name, &metadata);
                // Re-borrow to update the shard
                if let Some(shard) = self.shards.get_mut(shard_index) {
                    shard.metadata = Some(metadata);
                }
                Ok(())
            }
            Err(e) => {
                // Check if it's a rate limit error
                if let WebshartError::RateLimited = e {
                    // Set exponential backoff delay
                    let current_delay = self.rate_limit_delay.unwrap_or(Duration::from_secs(1));
                    self.rate_limit_delay = Some(current_delay * 2);
                    println!(
                        "[webshart] Rate limited, next delay will be {:?}",
                        self.rate_limit_delay
                    );
                }
                Err(e)
            }
        }
    }

    /// Clear the metadata cache for this dataset
    pub fn clear_cache(&self) -> Result<()> {
        if let Some(cache_dir) = &self.cache_dir {
            if cache_dir.exists() {
                fs::remove_dir_all(cache_dir)?;
                println!("[webshart] Cleared metadata cache for {}", self.name);
            }
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<(usize, u64)> {
        if let Some(cache_dir) = &self.cache_dir {
            let mut count = 0;
            let mut total_size = 0u64;

            if cache_dir.exists() {
                for entry in fs::read_dir(cache_dir)? {
                    let entry = entry?;
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                        count += 1;
                        total_size += entry.metadata()?.len();
                    }
                }
            }

            Ok((count, total_size))
        } else {
            Ok((0, 0))
        }
    }

    /// Ensure all metadata is loaded
    fn ensure_all_metadata_loaded(&mut self) -> Result<()> {
        // Use index-based loop to avoid borrow checker issues
        let num_shards = self.shards.len();
        for i in 0..num_shards {
            self.ensure_shard_metadata(i)?;
        }
        Ok(())
    }

    /// Find which shard contains a file by global index (loads metadata as needed)
    pub fn find_shard_for_file(&mut self, file_index: usize) -> Result<Option<(usize, usize)>> {
        let mut current_offset = 0;

        // Use index-based loop to avoid borrow checker issues
        for shard_idx in 0..self.shards.len() {
            self.ensure_shard_metadata(shard_idx)?;

            if let Some(metadata) = &self.shards[shard_idx].metadata {
                let num_files = metadata.num_files();
                if file_index < current_offset + num_files {
                    return Ok(Some((shard_idx, file_index - current_offset)));
                }
                current_offset += num_files;
            }
        }

        Ok(None)
    }

    /// Open a shard for reading
    pub fn open_shard(&mut self, shard_index: usize) -> Result<ShardReader> {
        // Ensure metadata is loaded
        self.ensure_shard_metadata(shard_index)?;

        if let Some(shard) = self.shards.get(shard_index) {
            if let Some(metadata) = &shard.metadata {
                ShardReader::new(
                    &shard.tar_path,
                    self.is_remote,
                    metadata.clone(),
                    self.discovery_token.clone(),
                    self.runtime.clone(),
                )
            } else {
                Err(WebshartError::InvalidShardFormat(
                    "Metadata not loaded".to_string(),
                ))
            }
        } else {
            Err(WebshartError::InvalidShardFormat(format!(
                "Shard index {} out of range",
                shard_index
            )))
        }
    }
}

/// Reader for accessing files within a shard
pub struct ShardReader {
    /// Path or URL to the tar file
    tar_location: String,

    /// Whether this is a remote shard
    is_remote: bool,

    /// Shard metadata
    metadata: ShardMetadata,

    /// Optional HuggingFace token for remote access
    hf_token: Option<String>,

    /// Runtime for async operations
    runtime: Arc<Runtime>,
}

impl ShardReader {
    /// Create a new shard reader
    pub fn new(
        tar_location: &str,
        is_remote: bool,
        metadata: ShardMetadata,
        hf_token: Option<String>,
        runtime: Arc<Runtime>,
    ) -> Result<Self> {
        Ok(Self {
            tar_location: tar_location.to_string(),
            is_remote,
            metadata,
            hf_token,
            runtime,
        })
    }

    /// Read a file by index within this shard
    pub fn read_file(&self, file_index: usize) -> Result<Vec<u8>> {
        if file_index >= self.metadata.num_files() {
            return Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range for shard",
                file_index
            )));
        }

