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
            return false;
        };
        file.try_lock_exclusive().is_err()
    }

    pub async fn cache_shard(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
    ) -> Result<PathBuf> {
        let (path, _) = self
            .ensure_cached(shard_name, remote_url, token, false)
            .await?;
        Ok(path)
    }

    pub async fn cache_shard_for_reading(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
    ) -> Result<(PathBuf, ShardLockGuard)> {
        let (path, read_lock) = self
            .ensure_cached(shard_name, remote_url, token, true)
            .await?;
        Ok((path, read_lock.expect("read lock requested")))
    }

    async fn ensure_cached(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
        lock_for_reading: bool,
    ) -> Result<(PathBuf, Option<ShardLockGuard>)> {
        let cached_path = self.get_cached_shard_path(shard_name);

        // The semaphore limits this process. The file lock prevents a second
        // process using the same cache directory from downloading the same shard.
        let _permit = self.download_semaphore.acquire().await.map_err(|e| {
            WebshartError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to acquire download permit: {}", e),
            ))
        })?;
        let _download_lock = self
            .lock_exclusive(self.download_lock_path(shard_name))
            .await?;

        // Recheck after taking the cross-process download lock.
        if let Ok(read_lock) = self.lock_shard_for_reading(shard_name).await {
            self.touch_shard(shard_name).await;
            return Ok((
                cached_path,
                if lock_for_reading {
                    Some(read_lock)
                } else {
                    None
                },
            ));
        }

        let temp_path = self.temp_download_path(shard_name);
        self.active_downloads
            .lock()
            .unwrap()
            .insert(shard_name.to_string(), temp_path.clone());

        let result = self
            .download_shard_to_disk(remote_url, token, shard_name, &temp_path)
            .await;

        self.active_downloads.lock().unwrap().remove(shard_name);

        match result {
            Ok(shard_size) => {
                self.record_cached_shard(shard_name, shard_size);
                // The download lock is still held, and evictors take that lock
                // before the shard lock. This closes the commit-to-read gap.
                let read_lock = if lock_for_reading {
                    Some(self.lock_shard_for_reading(shard_name).await?)
                } else {
                    None
                };
                Ok((cached_path, read_lock))
            }
            Err(error) => {
                let _ = fs::remove_file(&temp_path).await;
                Err(error)
            }
        }
    }

    async fn download_shard_to_disk(
        &self,
        url: &str,
        token: Option<String>,
        shard_name: &str,
        temp_path: &Path,
    ) -> Result<u64> {
        use futures::StreamExt;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(WebshartError::from)?;

        let mut request = client.get(url);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .map_err(WebshartError::from)?
            .error_for_status()
            .map_err(WebshartError::from)?;

        let mut file = File::create(temp_path).await.map_err(WebshartError::Io)?;
        let mut stream = response.bytes_stream();
        let mut bytes_written = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(WebshartError::from)?;
            bytes_written += chunk.len() as u64;
            file.write_all(&chunk).await.map_err(WebshartError::Io)?;

            if bytes_written % (1024 * 1024) == 0 {
                file.sync_data().await.map_err(WebshartError::Io)?;
            }
        }

        file.flush().await.map_err(WebshartError::Io)?;
        file.sync_all().await.map_err(WebshartError::Io)?;
        drop(file);

        self.commit_download(temp_path, shard_name, bytes_written)
            .await?;

        Ok(bytes_written)
    }

    /// Serializes the disk-space decision and final rename across processes.
    async fn commit_download(
        &self,
        temp_path: &Path,
        shard_name: &str,
        shard_size: u64,
    ) -> Result<()> {
        let _cache_lock = self.lock_exclusive(self.cache_lock_path()).await?;
        let _shard_lock = self
            .lock_exclusive(self.shard_lock_path(shard_name))
            .await?;

        self.evict_if_needed_locked(shard_size, Some(shard_name))
            .await?;

        fs::rename(temp_path, self.get_cached_shard_path(shard_name))
            .await
            .map_err(WebshartError::Io)?;
        self.touch_shard(shard_name).await;
        Ok(())
    }

    /// The caller must hold the cache-wide exclusive lock.
    async fn evict_if_needed_locked(
        &self,
        needed_bytes: u64,
        shard_to_keep: Option<&str>,
    ) -> Result<()> {
        let mut cached_shards = self.scan_cached_shards().await?;
        let mut current_size = cached_shards.iter().map(|shard| shard.size).sum::<u64>();
        cached_shards.sort_by_key(|shard| shard.last_used);

        for shard in &cached_shards {
            if current_size.saturating_add(needed_bytes) <= self.cache_limit_bytes {
                break;
            }
            if shard_to_keep == Some(shard.name.as_str()) {
                continue;
            }

            let download_lock = match Self::open_lock_file(&self.download_lock_path(&shard.name)) {
                Ok(file) => file,
                Err(_) => continue,
            };
            if download_lock.try_lock_exclusive().is_err() {
                continue;
            }

            let lock_path = self.shard_lock_path(&shard.name);
            let lock_file = match Self::open_lock_file(&lock_path) {
                Ok(file) => file,
