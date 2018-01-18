use crate::{
    error::{Result, WebshartError},
    FileInfo,
};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::runtime::Runtime;

pub trait FileLoader: Send + Sync {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>>;
}

pub(crate) fn file_http_client() -> Result<reqwest::Client> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }

    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(8)
        .build()
        .map_err(WebshartError::from)?;

    let _ = CLIENT.set(client);
    Ok(CLIENT.get().expect("HTTP client initialized").clone())
}

pub struct LocalFileLoader {
    tar_path: String,
}

impl LocalFileLoader {
    pub fn new(tar_path: String) -> Self {
        Self { tar_path }
    }
