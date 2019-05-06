use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod dataloader;
mod discovery;
mod error;
mod extract;
mod metadata;
mod metadata_resolver;
// Re-export main types
use dataloader::{scale_dimensions, PyBucketDataLoader, PyTarDataLoader, PyTarFileEntry};
pub use dataloader::{AspectBucketIterator, BatchOperations, BatchResult, FileReadRequest};
pub use discovery::{DatasetDiscovery, DiscoveredDataset};
pub use error::{Result, WebshartError};
