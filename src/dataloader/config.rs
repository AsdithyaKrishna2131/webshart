use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub load_file_data: bool,
    pub max_file_size: u64,
    pub buffer_size: usize,
    pub chunk_size_mb: usize,
    pub hf_token: Option<String>,
    pub batch_size: Option<usize>,
}

impl DataLoaderConfig {
    pub fn from_state_dict(state_dict: &Bound<'_, PyDict>) -> Self {
        Self {
            load_file_data: state_dict
                .get_item("load_file_data")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true),
            max_file_size: state_dict
                .get_item("max_file_size")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(50_000_000),
