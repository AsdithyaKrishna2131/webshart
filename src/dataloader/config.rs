use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub load_file_data: bool,
