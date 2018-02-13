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
}

impl FileLoader for LocalFileLoader {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(&self.tar_path)?;
        file.seek(SeekFrom::Start(file_info.offset))?;

        let mut buffer = vec![0u8; file_info.length as usize];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }
}

pub struct RemoteFileLoader {
    url: String,
    token: Option<String>,
    runtime: Arc<Runtime>,
}

impl RemoteFileLoader {
    pub fn new(url: String, token: Option<String>, runtime: Arc<Runtime>) -> Self {
        Self {
            url,
            token,
            runtime,
        }
    }
}

impl FileLoader for RemoteFileLoader {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>> {
        self.runtime.block_on(async {
            let client = file_http_client()?;

            let mut request = client
                .get(&self.url)
