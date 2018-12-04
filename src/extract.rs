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
        self
    }

    pub fn extract_metadata(
        &self,
        source: &str,
        destination: &str,
        checkpoint_dir: Option<&str>,
        max_workers: usize,
        shard_range: Option<(usize, usize)>, // NEW: Add range parameter
    ) -> Result<()> {
        self.runtime.block_on(async {
            // Set up Ctrl+C handler
            let ctrl_c = tokio::signal::ctrl_c();

            // Create the main extraction future
            let extraction = async {
                // Discover unindexed shards
                let mut shards = self
                    .discover_unindexed_shards(source, checkpoint_dir)
                    .await?;

                // Apply range filter if provided
                if let Some((start, end)) = shard_range {
                    println!("[webshart] Filtering shards to range [{}, {})", start, end);

                    // Sort shards by name first to ensure consistent ordering
                    shards.sort_by(|a, b| a.name.cmp(&b.name));

                    // Filter to the specified range
                    let total_shards = shards.len();
                    shards = shards
                        .into_iter()
                        .enumerate()
                        .filter_map(|(idx, shard)| {
                            if idx >= start && idx < end {
                                Some(shard)
                            } else {
                                None
                            }
                        })
                        .collect();

                    println!(
                        "[webshart] Processing {} shards out of {} total (indices {}-{})",
                        shards.len(),
                        total_shards,
                        start,
                        std::cmp::min(end, total_shards) - 1
                    );
                }

                if shards.is_empty() {
                    println!("[webshart] No unindexed shards found in specified range");
                    return Ok(());
                }

                println!(
                    "[webshart] Found {} unindexed shards to process",
                    shards.len()
                );

                // Load checkpoints
                let checkpoints = if let Some(dir) = checkpoint_dir {
                    self.load_checkpoints(dir)?
                } else {
                    HashMap::new()
                };

                // Create multi-progress for managing multiple progress bars
                let multi_progress = Arc::new(MultiProgress::new());

                // Process shards in parallel
                let semaphore = Arc::new(tokio::sync::Semaphore::new(max_workers));
                let futures = shards.into_iter().map(|shard| {
                    let sem = semaphore.clone();
                    let checkpoint = checkpoints.get(&shard.name).cloned();
                    let token = self.hf_token.clone();
                    let dest = destination.to_string();
                    let checkpoint_dir = checkpoint_dir.map(|s| s.to_string());
                    let extractor = self.clone();
                    let mp = multi_progress.clone();

                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = extractor
                            .process_shard(
                                shard.clone(),
                                checkpoint,
                                &dest,
                                checkpoint_dir.as_deref(),
                                token,
                                mp,
                            )
                            .await;
                        result
                    }
                });

                let results = join_all(futures).await;

                // Check for failures
                let mut failed = 0;
                let mut errors = Vec::new();
                for (i, result) in results.iter().enumerate() {
                    if let Err(e) = result {
                        eprintln!("[webshart] Failed to process shard {}: {}", i, e);
                        errors.push(e.to_string());
                        failed += 1;
                    }
                }

                if failed > 0 {
                    return Err(WebshartError::DiscoveryFailed(format!(
                        "Failed to process {} shards. First error: {}",
                        failed,
                        errors.first().unwrap_or(&"Unknown error".to_string())
                    )));
                }

                println!("[webshart] Successfully extracted metadata for all shards");
                Ok(())
            };

            // Run with interrupt handling
            tokio::select! {
                result = extraction => result,
                _ = ctrl_c => {
                    println!("\n[webshart] Received interrupt signal, stopping...");
                    Err(WebshartError::DiscoveryFailed("Cancelled by user".to_string()))
                }
            }
        })
    }

    pub fn extract_metadata_internal(
        &self,
        source: &str,
        destination: &str,
        checkpoint_dir: Option<&str>,
        max_workers: usize,
    ) -> Result<()> {
        self.runtime.block_on(async {
            // Discover unindexed shards
            let shards = self
                .discover_unindexed_shards(source, checkpoint_dir)
                .await?;

            if shards.is_empty() {
                println!("[webshart] No unindexed shards found");
                return Ok(());
            }

            println!(
                "[webshart] Found {} unindexed shards to process",
                shards.len()
            );

            // Load checkpoints
            let checkpoints = if let Some(dir) = checkpoint_dir {
                self.load_checkpoints(dir)?
            } else {
                HashMap::new()
            };

            // Create multi-progress for managing multiple progress bars
            let multi_progress = Arc::new(MultiProgress::new());

            // Process shards in parallel
            let semaphore = Arc::new(tokio::sync::Semaphore::new(max_workers));
            let futures = shards.into_iter().map(|shard| {
                let sem = semaphore.clone();
                let checkpoint = checkpoints.get(&shard.name).cloned();
                let token = self.hf_token.clone();
                let dest = destination.to_string();
                let checkpoint_dir = checkpoint_dir.map(|s| s.to_string());
                let extractor = self.clone();
                let mp = multi_progress.clone();

                async move {
                    let _permit = sem.acquire().await.unwrap();
                    let result = extractor
                        .process_shard(
                            shard.clone(),
                            checkpoint,
                            &dest,
                            checkpoint_dir.as_deref(),
                            token,
                            mp,
                        )
                        .await;
                    result
                }
            });

            let results = join_all(futures).await;

            // Check for failures and stop immediately
            let mut failed = 0;
            let mut errors = Vec::new();
            for (i, result) in results.iter().enumerate() {
                if let Err(e) = result {
                    eprintln!("[webshart] Failed to process shard {}: {}", i, e);
                    errors.push(e.to_string());
                    failed += 1;
                }
            }

            if failed > 0 {
                return Err(WebshartError::DiscoveryFailed(format!(
                    "Failed to process {} shards. First error: {}",
                    failed,
                    errors.first().unwrap_or(&"Unknown error".to_string())
                )));
            }

            println!("[webshart] Successfully extracted metadata for all shards");
            Ok(())
        })
    }

    async fn discover_unindexed_shards(
        &self,
        source: &str,
        checkpoint_dir: Option<&str>,
    ) -> Result<Vec<UnindexedShard>> {
        let mut shards = Vec::new();

        // Load existing checkpoints to filter out completed shards
        let checkpoints = if let Some(dir) = checkpoint_dir {
            self.load_checkpoints(dir)?
        } else {
            HashMap::new()
        };

        if Path::new(source).exists() {
            // Local discovery
            self.discover_local_unindexed(Path::new(source), &mut shards)?;
        } else {
            // HuggingFace discovery
            shards = self.discover_hf_unindexed(source).await?;
        }

        // Filter based on checkpoints and existing JSON files
        let mut filtered_shards = Vec::new();
        for shard in shards {
            // let base_name = shard.name.trim_end_matches(".tar");
            let json_exists = if shard.is_remote {
                false // Can't easily check remote JSON existence
            } else {
                Path::new(&shard.path).with_extension("json").exists()
            };

            if let Some(checkpoint) = checkpoints.get(&shard.name) {
                match &checkpoint.status {
                    CheckpointStatus::Complete => {
                        println!(
                            "[webshart] Skipping {} (marked complete in checkpoint)",
                            shard.name
                        );
                        continue;
                    }
                    CheckpointStatus::Failed(err) => {
                        println!(
                            "[webshart] Retrying {} (previously failed: {})",
                            shard.name, err
                        );
                    }
                    CheckpointStatus::InProgress => {
                        println!(
                            "[webshart] Resuming {} from offset {}",
                            shard.name, checkpoint.offset
                        );
                    }
                    CheckpointStatus::Pending => {
                        println!("[webshart] Processing {} (pending)", shard.name);
                    }
                }
            } else if json_exists {
                // Has JSON but no checkpoint - might be from a previous incomplete run
                println!(
                    "[webshart] Found {} with existing JSON but no checkpoint, will verify",
                    shard.name
                );
            } else {
                println!(
                    "[webshart] Processing {} (no JSON, no checkpoint)",
                    shard.name
                );
            }

            filtered_shards.push(shard);
        }

        // Sort by name for consistent ordering
        filtered_shards.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(filtered_shards)
    }

    fn discover_local_unindexed(
        &self,
        path: &Path,
        shards: &mut Vec<UnindexedShard>,
    ) -> Result<()> {
        // Find ALL tar files, not just ones without JSON
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();

                if entry_path.is_dir() {
                    // Recurse into subdirectories
                    self.discover_local_unindexed(&entry_path, shards)?;
                } else if let Some(file_name) = entry_path.file_name() {
                    let file_name_str = file_name.to_string_lossy();

                    if let Some(_captures) = self.shard_pattern.captures(&file_name_str) {
                        // Found a tar file - add it regardless of JSON existence
                        shards.push(UnindexedShard {
                            name: file_name_str.to_string(),
                            path: entry_path.to_string_lossy().to_string(),
                            size: entry.metadata()?.len(),
                            is_remote: false,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    async fn discover_hf_unindexed(&self, repo_id: &str) -> Result<Vec<UnindexedShard>> {
        // Use HF dataset info API
        let api_url = format!("https://huggingface.co/api/datasets/{}", repo_id);
        let mut request = self.client.get(&api_url);

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(WebshartError::DiscoveryFailed(format!(
                "Failed to get dataset info: {}",
                response.status()
            )));
        }

        #[derive(Deserialize)]
        struct Sibling {
            rfilename: String,
            size: Option<u64>,
        }

        #[derive(Deserialize)]
        struct DatasetInfo {
            siblings: Vec<Sibling>,
        }

        let info: DatasetInfo = response.json().await?;

        // Find ALL tar files
        let mut shards = Vec::new();

        for sibling in info.siblings {
            let path = Path::new(&sibling.rfilename);
            if let Some(file_name) = path.file_name() {
                let file_name_str = file_name.to_string_lossy();

                if let Some(_captures) = self.shard_pattern.captures(&file_name_str) {
                    let size = sibling.size.unwrap_or(0);
                    let size_str = if size == 0 {
                        "unknown".to_string()
                    } else {
                        format!("{} bytes", size)
                    };

                    println!(
                        "[webshart] Found tar file: {} (size: {})",
                        file_name_str, size_str
                    );

                    shards.push(UnindexedShard {
                        name: file_name_str.to_string(),
                        path: format!(
                            "https://huggingface.co/datasets/{}/resolve/main/{}",
                            repo_id, sibling.rfilename
                        ),
                        size,
                        is_remote: true,
                    });
                }
            }
        }

        Ok(shards)
    }

    async fn process_shard(
        &self,
        shard: UnindexedShard,
        checkpoint: Option<ShardCheckpoint>,
        destination: &str,
        checkpoint_dir: Option<&str>,
        hf_token: Option<String>,
        multi_progress: Arc<MultiProgress>,
    ) -> Result<()> {
        let start_offset = checkpoint.as_ref().map(|c| c.offset).unwrap_or(0);

        // Update checkpoint to in-progress
        if let Some(dir) = checkpoint_dir {
            let checkpoint = ShardCheckpoint {
                shard_name: shard.name.clone(),
                status: CheckpointStatus::InProgress,
                offset: start_offset,
                files_processed: 0,
            };
            self.save_checkpoint(dir, &checkpoint)?;
        }

        let metadata = match if shard.is_remote {
            self.extract_remote_metadata(&shard, start_offset, hf_token.clone(), multi_progress)
                .await
        } else {
            self.extract_local_metadata(&shard, start_offset, multi_progress)
        } {
            Ok(m) => m,
            Err(e) => {
                // Save failed checkpoint
                if let Some(dir) = checkpoint_dir {
                    let checkpoint = ShardCheckpoint {
                        shard_name: shard.name.clone(),
                        status: CheckpointStatus::Failed(e.to_string()),
                        offset: start_offset,
                        files_processed: 0,
                    };
                    self.save_checkpoint(dir, &checkpoint)?;
                }
                return Err(e);
            }
        };

        // Save metadata
        self.save_metadata(&shard, metadata, destination).await?;

        // Update checkpoint to complete
        if let Some(dir) = checkpoint_dir {
            let checkpoint = ShardCheckpoint {
                shard_name: shard.name.clone(),
                status: CheckpointStatus::Complete,
                offset: shard.size,
                files_processed: 0,
            };
            self.save_checkpoint(dir, &checkpoint)?;
        }

        Ok(())
    }

    async fn extract_remote_metadata(
        &self,
        shard: &UnindexedShard,
        start_offset: u64,
        hf_token: Option<String>,
        multi_progress: Arc<MultiProgress>,
    ) -> Result<ShardMetadata> {
        // Create progress bar for download
        let download_pb = multi_progress.add(ProgressBar::new(shard.size));
        download_pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        download_pb.set_message(format!("↓ {}", shard.name));

        // Create request
        let mut request = self
            .client
            .get(&shard.path)
            .header("Accept-Encoding", "identity")
            .timeout(std::time::Duration::from_secs(600));

        if let Some(token) = &hf_token {
            request = request.bearer_auth(token);
        }

        if start_offset > 0 {
            request = request.header("Range", format!("bytes={}-", start_offset));
