// dataloader/shard_cache.rs
use crate::digest_to_hex;
use crate::error::{Result, WebshartError};
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

const LOCK_DIR_NAME: &str = ".webshart-locks";
const DOWNLOAD_DIR_NAME: &str = ".webshart-downloads";
const ACCESS_DIR_NAME: &str = ".webshart-access";
const CACHE_LOCK_NAME: &str = "cache.lock";

static DOWNLOAD_ID: AtomicU64 = AtomicU64::new(0);

/// An RAII guard that holds a shared lock on a shard.
/// The lock is released when this struct is dropped.
#[derive(Debug)]
pub struct ShardLockGuard {
    #[allow(dead_code)]
    file: std::fs::File,
}

#[derive(Debug)]
struct ExclusiveLockGuard {
    #[allow(dead_code)]
    file: std::fs::File,
}

#[derive(Debug)]
struct CachedShard {
    name: String,
    size: u64,
    last_used: SystemTime,
}

#[derive(Debug, Clone)]
pub struct ShardCache {
    cache_dir: PathBuf,
    cache_limit_bytes: u64,
    current_size_bytes: Arc<Mutex<u64>>,
    lru_queue: Arc<Mutex<VecDeque<String>>>,
    shard_sizes: Arc<Mutex<HashMap<String, u64>>>,
    download_semaphore: Arc<Semaphore>,
    active_downloads: Arc<Mutex<HashMap<String, PathBuf>>>,
}

impl ShardCache {
    pub fn new(cache_dir: PathBuf, cache_limit_gb: f64, parallel_downloads: usize) -> Self {
        Self {
            cache_dir,
            cache_limit_bytes: (cache_limit_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            current_size_bytes: Arc::new(Mutex::new(0)),
            lru_queue: Arc::new(Mutex::new(VecDeque::new())),
            shard_sizes: Arc::new(Mutex::new(HashMap::new())),
            download_semaphore: Arc::new(Semaphore::new(parallel_downloads)),
            active_downloads: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn ensure_cache_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(WebshartError::Io)?;
        fs::create_dir_all(self.lock_dir())
            .await
            .map_err(WebshartError::Io)?;
        fs::create_dir_all(self.download_dir())
            .await
            .map_err(WebshartError::Io)?;
        fs::create_dir_all(self.access_dir())
            .await
            .map_err(WebshartError::Io)
    }

    pub fn get_cached_shard_path(&self, shard_name: &str) -> PathBuf {
        self.cache_dir.join(shard_name)
    }

    pub async fn is_cached(&self, shard_name: &str) -> bool {
        self.get_cached_shard_path(shard_name).is_file()
    }

    pub async fn lock_shard_for_reading(&self, shard_name: &str) -> Result<ShardLockGuard> {
        let lock_path = self.shard_lock_path(shard_name);
        let cached_path = self.get_cached_shard_path(shard_name);
        let file = tokio::task::spawn_blocking(move || -> std::io::Result<std::fs::File> {
            let file = Self::open_lock_file(&lock_path)?;
            file.lock_shared()?;
            Ok(file)
        })
        .await
        .map_err(Self::join_error)??;

        // Check after taking the lock. An evictor cannot remove the shard between
        // this check and the caller finishing its read.
        if !cached_path.is_file() {
            return Err(WebshartError::CacheMiss(shard_name.to_string()));
        }

        Ok(ShardLockGuard { file })
    }

    pub fn is_shard_locked(&self, shard_name: &str) -> bool {
        let Ok(file) = Self::open_lock_file(&self.shard_lock_path(shard_name)) else {
