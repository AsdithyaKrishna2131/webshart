use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub load_file_data: bool,
    pub max_file_size: u64,
    pub buffer_size: usize,
    pub chunk_size_mb: usize,
