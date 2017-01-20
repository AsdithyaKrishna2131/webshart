use crate::discovery::{DatasetDiscovery, DiscoveredDataset, PyDiscoveredDataset};
use crate::error::{Result, WebshartError};
use crate::metadata::{CaptionValue, ShardMetadata};
use crate::FileInfo;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use rand::{rng, seq::SliceRandom};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

type IndexedFileEntry = (usize, String, FileInfo);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RemoteByteRange {
    start: u64,
    end: u64,
}

fn plan_remote_byte_ranges(
    file_entries: &[IndexedFileEntry],
    max_gap: u64,
    max_span: u64,
) -> Vec<RemoteByteRange> {
    let mut entries: Vec<&IndexedFileEntry> = file_entries
        .iter()
        .filter(|(_, _, file_info)| file_info.length > 0)
        .collect();
    entries.sort_by_key(|(file_idx, _, file_info)| (file_info.offset, *file_idx));

    let mut ranges: Vec<RemoteByteRange> = Vec::new();
    for (_, _, file_info) in entries {
        let entry_end = file_info.offset.saturating_add(file_info.length);
        let Some(current) = ranges.last_mut() else {
            ranges.push(RemoteByteRange {
                start: file_info.offset,
                end: entry_end,
            });
            continue;
        };

        let gap = file_info.offset.saturating_sub(current.end);
        let combined_end = current.end.max(entry_end);
        let combined_span = combined_end.saturating_sub(current.start);
        if gap > max_gap || combined_span > max_span {
            ranges.push(RemoteByteRange {
                start: file_info.offset,
                end: entry_end,
            });
        } else {
            current.end = combined_end;
        }
    }
    ranges
}

#[cfg(test)]
mod remote_range_tests {
    use super::{plan_remote_byte_ranges, FileInfo, IndexedFileEntry, RemoteByteRange};

    fn entry(file_idx: usize, offset: u64, length: u64) -> IndexedFileEntry {
        (
            file_idx,
            format!("{file_idx}.bin"),
            FileInfo {
                path: None,
                offset,
                length,
                sha256: None,
                width: None,
                height: None,
                aspect: None,
                json_path: None,
                json_offset: None,
                json_length: None,
                captions: None,
                json_metadata: None,
            },
        )
    }

    #[test]
    fn remote_ranges_split_large_physical_gaps_and_sort_by_offset() {
        let entries = vec![entry(0, 1_000, 100), entry(1, 0, 100), entry(2, 150, 100)];

        assert_eq!(
            plan_remote_byte_ranges(&entries, 100, 10_000),
            vec![
                RemoteByteRange { start: 0, end: 250 },
                RemoteByteRange {
                    start: 1_000,
                    end: 1_100,
                },
            ]
        );
    }

    #[test]
    fn remote_ranges_respect_maximum_combined_span() {
        let entries = vec![entry(0, 0, 100), entry(1, 150, 100)];

        assert_eq!(
            plan_remote_byte_ranges(&entries, 100, 200),
            vec![
                RemoteByteRange { start: 0, end: 100 },
                RemoteByteRange {
                    start: 150,
                    end: 250,
                },
            ]
        );
    }
}

// Import from modules
mod aspect_buckets;
mod batch;
mod config;
mod entry_types;
mod file_loading;
pub mod shard_cache;
use crate::impl_batch_iterator;
use crate::metadata::ensure_shard_metadata_with_retry;
use aspect_buckets::{calculate_bucket_key, BucketKeyType, BucketSamplingStrategy};
pub use aspect_buckets::{
    scale_dimensions_with_multiple, AspectBucketEntry, AspectBucketIterator, AspectBuckets,
};
pub use batch::{BatchIterable, BatchOperations, BatchResult, FileReadRequest, PyBatchOperations};
use config::DataLoaderConfig;
pub use entry_types::{create_tar_entry, BucketEntry, PyTarFileEntry};
use file_loading::{create_file_loader, file_http_client};

// Re-export the entry type as it's part of the public API
pub use entry_types::PyTarFileEntry as TarFileEntry;

// Generate batch iterators using the macro
impl_batch_iterator!(
    PyBatchIterator,
    "PyBatchIterator",
    PyTarDataLoader,
    PyTarFileEntry
);

impl_batch_iterator!(
    PyBucketBatchIterator,
    "PyBucketBatchIterator",
    PyBucketDataLoader,
    PyTarFileEntry
);

// ===== TAR DATA LOADER =====
#[pyclass]
pub struct PyRangeIterator {
    loader: Py<PyTarDataLoader>,
    start: usize,
    end: usize,
    current: usize,
}

#[pymethods]
impl PyRangeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        if slf.current >= slf.end {
            return Ok(None);
        }

        slf.current += 1;

        Python::attach(|py| {
            let mut loader = slf.loader.borrow_mut(py);
            loader.next_entry()
        })
    }
}

#[pyclass(name = "TarDataLoader")]
pub struct PyTarDataLoader {
    dataset: Arc<Mutex<DiscoveredDataset>>,
    runtime: Arc<Runtime>,
    config: DataLoaderConfig,
    current_shard: usize,
    entry_buffer: Vec<PyTarFileEntry>,
    buffer_position: usize,
    next_file_to_load: usize,
    source: String,
    metadata_source: Option<String>,
    ranges: Option<Vec<(usize, usize)>>,
    current_range_idx: usize,
}

impl BatchIterable<PyTarFileEntry> for PyTarDataLoader {
    fn next_item(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        self.next_entry()
    }

    fn get_batch_size(&self) -> Option<usize> {
        self.config.batch_size
    }
}

#[pymethods]
impl PyTarDataLoader {
    #[new]
    #[pyo3(signature = (dataset_or_path, load_file_data=true, max_file_size=50_000_000, buffer_size=100, hf_token=None, chunk_size_mb=10, batch_size=None))]
    fn new(
        dataset_or_path: &Bound<'_, PyAny>,
        load_file_data: bool,
        max_file_size: u64,
        buffer_size: usize,
        hf_token: Option<String>,
        chunk_size_mb: usize,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        let runtime =
            Arc::new(Runtime::new().map_err(|e| {
                WebshartError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?);

        let (dataset, source) =
            if let Ok(py_dataset) = dataset_or_path.extract::<PyRef<PyDiscoveredDataset>>() {
                let source = py_dataset.inner.name.clone();
                (py_dataset.inner.clone(), source)
            } else if let Ok(path) = dataset_or_path.extract::<String>() {
                let discovery = DatasetDiscovery::new().with_optional_token(hf_token.clone());

                let dataset = if Path::new(&path).exists() {
                    discovery.discover_local(Path::new(&path))?
                } else {
                    runtime.block_on(discovery.discover_huggingface(&path, None))?
                };
                (dataset, path)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected either a DiscoveredDataset or a path string",
                ));
            };

        let metadata_source = dataset.metadata_source.clone();

        Ok(Self {
            dataset: Arc::new(Mutex::new(dataset)),
            runtime,
            config: DataLoaderConfig {
                load_file_data,
                max_file_size,
                buffer_size: buffer_size.max(1),
                chunk_size_mb,
                hf_token,
                batch_size,
            },
            current_shard: 0,
            entry_buffer: Vec::with_capacity(buffer_size),
            buffer_position: 0,
            metadata_source,
            next_file_to_load: 0,
            current_range_idx: 0,
            ranges: None,
            source,
        })
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let current_file_index = self.calculate_current_file_index();

        dict.set_item("current_shard", self.current_shard)?;
        dict.set_item("current_file_index", current_file_index)?;
        dict.set_item("buffer_position", self.buffer_position)?;

        self.config.to_state_dict(&dict)?;

        dict.set_item("source", &self.source)?;
        dict.set_item("metadata_source", &self.metadata_source)?;

        let dataset = self.dataset.lock().unwrap();
        dict.set_item("num_shards", dataset.num_shards())?;
        dict.set_item("is_remote", dataset.is_remote)?;
        dict.set_item("version", 4)?;

        Ok(dict.into_any().unbind())
    }

    fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Ok(Some(version_item)) = state_dict.get_item("version") {
            if let Ok(version) = version_item.extract::<i32>() {
                if version > 4 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported state dict version: {}",
                        version
                    )));
                }
            }
        }

        let mut new_shard = self.current_shard;
        let mut new_file_index = 0;

        if let Ok(Some(item)) = state_dict.get_item("current_shard") {
            if let Ok(v) = item.extract::<usize>() {
                new_shard = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("current_file_index") {
            if let Ok(v) = item.extract::<usize>() {
                new_file_index = v;
            }
        }

        // Load configuration fields individually to preserve existing values
        if let Ok(Some(item)) = state_dict.get_item("load_file_data") {
            if let Ok(v) = item.extract::<bool>() {
                self.config.load_file_data = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("max_file_size") {
            if let Ok(v) = item.extract::<u64>() {
                self.config.max_file_size = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("buffer_size") {
            if let Ok(v) = item.extract::<usize>() {
                self.config.buffer_size = v.max(1);
                if self.entry_buffer.capacity() < self.config.buffer_size {
                    self.entry_buffer
                        .reserve(self.config.buffer_size - self.entry_buffer.capacity());
                }
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("chunk_size_mb") {
            if let Ok(v) = item.extract::<usize>() {
                self.config.chunk_size_mb = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("hf_token") {
            if let Ok(v) = item.extract::<Option<String>>() {
                self.config.hf_token = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("batch_size") {
            if let Ok(v) = item.extract::<Option<usize>>() {
                self.config.batch_size = v;
            }
        }

        if let Ok(Some(item)) = state_dict.get_item("metadata_source") {
            if let Ok(v) = item.extract::<Option<String>>() {
                self.metadata_source = v;
            }
        }

        if new_shard < self.dataset.lock().unwrap().num_shards() {
            ensure_shard_metadata_with_retry(&mut self.dataset.lock().unwrap(), new_shard)?;
        }

        self.entry_buffer.clear();
        self.buffer_position = 0;
        self.current_shard = new_shard;
        self.next_file_to_load = new_file_index;

        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (state_dict, dataset_or_path=None))]
    fn from_state_dict(
        py: Python,
        state_dict: &Bound<'_, PyDict>,
        dataset_or_path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let config = DataLoaderConfig::from_state_dict(state_dict);

        let source_obj = Self::determine_source_from_state_dict(py, state_dict, dataset_or_path)?;
        let source = source_obj.bind(py);

        let mut loader = Self::new(
            source,
            config.load_file_data,
            config.max_file_size,
            config.buffer_size,
            config.hf_token,
            config.chunk_size_mb,
            config.batch_size,
        )?;

        loader.load_state_dict(state_dict)?;
        Ok(loader)
    }

    fn get_state_summary(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let mut dataset = self.dataset.lock().unwrap();
        let current_file_index_in_shard = self.calculate_current_file_index();

        let files_processed =
            self.calculate_files_processed(&mut dataset, current_file_index_in_shard)?;
        let total_files = dataset.total_files().unwrap_or(0);

        dict.set_item("current_shard", self.current_shard)?;
        dict.set_item("total_shards", dataset.num_shards())?;
        dict.set_item("current_file_index", current_file_index_in_shard)?;
        dict.set_item("files_processed", files_processed)?;
        dict.set_item("total_files", total_files)?;
        dict.set_item(
            "progress_percent",
            if total_files > 0 {
                files_processed as f64 / total_files as f64 * 100.0
            } else {
                0.0
            },
        )?;
        dict.set_item("batch_size", self.config.batch_size)?;

        Ok(dict.into_any().unbind())
    }

    /// Create an iterator for a specific range
    pub fn iter_range(
        mut slf: PyRefMut<'_, Self>,
        start: usize,
        end: usize,
    ) -> PyResult<PyRangeIterator> {
