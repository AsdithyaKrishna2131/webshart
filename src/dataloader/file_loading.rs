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
