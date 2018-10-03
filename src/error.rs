use pyo3::exceptions::PyException;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WebshartError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

