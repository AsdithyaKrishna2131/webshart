use pyo3::exceptions::PyException;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WebshartError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Metadata not found: {0}")]
    MetadataNotFound(String),
