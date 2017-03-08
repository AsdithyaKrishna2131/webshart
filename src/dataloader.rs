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
        if start >= end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "start must be less than end",
            ));
        }

        // Skip to start position
        slf.skip(start)?;

        Ok(PyRangeIterator {
            loader: slf.into(),
            start,
            end,
            current: start,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        slf.next_entry()
    }

    fn next_batch(&mut self) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        <Self as BatchIterable<PyTarFileEntry>>::next_batch(self)
    }

    fn iter_batches(slf: PyRef<'_, Self>) -> PyResult<PyBatchIterator> {
        if slf.config.batch_size.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be set to use iter_batches()",
            ));
        }
        Ok(PyBatchIterator { loader: slf.into() })
    }

    fn will_block(&self) -> PyResult<bool> {
        // If we have entries in buffer, won't block
        if self.buffer_position < self.entry_buffer.len() {
            return Ok(false);
        }
        let dataset = self.dataset.lock().unwrap();
        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(false);
        }
        if self.current_shard >= dataset.num_shards() {
            return Ok(false);
        }
        let shard_name = dataset.shards[self.current_shard]
            .tar_path
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string();
        let cache = dataset.shard_cache.as_ref().unwrap().clone();

        drop(dataset);
        let is_cached = self.runtime.block_on(cache.is_cached(&shard_name));
        Ok(!is_cached)
    }

    fn is_shard_locked(&self, shard_name: &str) -> bool {
        let dataset = self.dataset.lock().unwrap();

        if let Some(cache) = &dataset.shard_cache {
            cache.is_shard_locked(shard_name)
        } else {
            false
        }
    }

    fn prepare_shard_by_name(&self, filename: &str) -> PyResult<bool> {
        let dataset = self.dataset.lock().unwrap();

        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(false);
        }

        let shard_idx = self.find_shard_by_filename(&dataset, filename)?;
        let shard = &dataset.shards[shard_idx];
        let tar_path = shard.tar_path.clone();
        let shard_name = tar_path.rsplit('/').next().unwrap_or(&tar_path).to_string();
        let token = dataset.get_hf_token();
        let cache = dataset.shard_cache.as_ref().unwrap().clone();

        // Drop the dataset lock before any async/await or spawn
        drop(dataset);

        // Check cache status before spawning
        let is_cached = self.runtime.block_on({
            let cache = cache.clone();
            let shard_name = shard_name.clone();
            async move { cache.is_cached(&shard_name).await }
        });
        if is_cached {
            return Ok(false);
        }

        // Clone all data needed for the async block before spawning
        let runtime = self.runtime.clone();
        let shard_name_cloned = shard_name.clone();
        let tar_path_cloned = tar_path.clone();
        let token_cloned = token.clone();
        let cache_cloned = cache.clone();

        // All clones above are Send, so nothing non-Send is captured
        runtime.spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                match cache_cloned
                    .cache_shard(&shard_name_cloned, &tar_path_cloned, token_cloned)
                    .await
                {
                    Ok(_) => {
                        println!("[webshart] Pre-cached shard: {}", shard_name_cloned);
                    }
                    Err(e) => {
                        eprintln!(
                            "[webshart] Failed to pre-cache shard {}: {}",
                            shard_name_cloned, e
                        );
                    }
                }
            })
        });

        Ok(true)
    }

    fn prepare_next_shard(&self) -> PyResult<bool> {
        let dataset = self.dataset.lock().unwrap();

        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(false);
        }

        if self.current_shard >= dataset.num_shards() {
            return Ok(false);
        }

        let shard = &dataset.shards[self.current_shard];
        let tar_path = shard.tar_path.clone();
        let shard_name = tar_path.rsplit('/').next().unwrap_or(&tar_path).to_string();
        let token = dataset.get_hf_token();
        let cache = dataset.shard_cache.as_ref().unwrap().clone();

        // Drop the dataset lock before any async/await or spawn
        drop(dataset);

        // Check cache status before spawning
        let is_cached = self.runtime.block_on({
            let cache = cache.clone();
            let shard_name = shard_name.clone();
            async move { cache.is_cached(&shard_name).await }
        });
        if is_cached {
            return Ok(false);
        }

        // Clone all data needed for the async block before spawning
        let runtime = self.runtime.clone();
        let shard_name_cloned = shard_name.clone();
        let tar_path_cloned = tar_path.clone();
        let token_cloned = token.clone();
        let cache_cloned = cache.clone();

        // All clones above are Send, so nothing non-Send is captured
        runtime.spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                match cache_cloned
                    .cache_shard(&shard_name_cloned, &tar_path_cloned, token_cloned)
                    .await
                {
                    Ok(_) => {
                        println!("[webshart] Pre-cached shard: {}", shard_name_cloned);
                    }
                    Err(e) => {
                        eprintln!(
                            "[webshart] Failed to pre-cache shard {}: {}",
                            shard_name_cloned, e
                        );
                    }
                }
            })
        });

        Ok(true)
    }

    /// Get information about which shard will be loaded next
    fn get_next_shard_info(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.buffer_position < self.entry_buffer.len() {
            return Ok(None); // Still have buffered entries
        }

        let dataset = self.dataset.lock().unwrap();

        if self.current_shard >= dataset.num_shards() {
            return Ok(None); // No more shards
        }

        let shard = &dataset.shards[self.current_shard];
        let dict = PyDict::new(py);

        dict.set_item("index", self.current_shard)?;
        dict.set_item("name", &shard.name)?;
        dict.set_item("tar_path", &shard.tar_path)?;

        if dataset.is_remote && dataset.shard_cache.is_some() {
            let shard_name = shard
                .tar_path
                .rsplit('/')
                .next()
                .unwrap_or(&shard.tar_path)
                .to_string();
            let cache = dataset.shard_cache.as_ref().unwrap().clone();
            drop(dataset);

            let is_cached = self.runtime.block_on(cache.is_cached(&shard_name));
            dict.set_item("is_cached", is_cached)?;
        } else {
            dict.set_item("is_cached", true)?; // Local files are always "cached"
        }

        Ok(Some(dict.into_any().unbind()))
    }

    /// Get information about a specific shard's cache status
    fn get_shard_cache_status(&self, py: Python, filename: &str) -> PyResult<Py<PyAny>> {
        let dataset = self.dataset.lock().unwrap();

        let shard_idx = self.find_shard_by_filename(&dataset, filename)?;
        let shard = &dataset.shards[shard_idx];

        let dict = PyDict::new(py);
        dict.set_item("index", shard_idx)?;
        dict.set_item("name", &shard.name)?;
        dict.set_item("tar_path", &shard.tar_path)?;

        if dataset.is_remote && dataset.shard_cache.is_some() {
            let shard_filename = shard
                .tar_path
                .rsplit('/')
                .next()
                .unwrap_or(&shard.tar_path)
                .to_string();
            let cache = dataset.shard_cache.as_ref().unwrap().clone();
            drop(dataset);

            let (is_cached, size) = self.runtime.block_on(async {
                let is_cached = cache.is_cached(&shard_filename).await;
                let size = cache
                    .get_cached_file_size(&shard_filename)
                    .await
                    .unwrap_or(0);
                (is_cached, size)
            });
            dict.set_item("is_cached", is_cached)?;
            dict.set_item("cur_filesize", size)?;
        } else {
            drop(dataset);
            dict.set_item("is_cached", true)?; // Local files are always "cached"
        }

        Ok(dict.into_any().unbind())
    }

    #[pyo3(signature = (lookahead=5))]
    fn get_lookahead_cache_status(&self, py: Python, lookahead: usize) -> PyResult<Py<PyAny>> {
        let dataset = self.dataset.lock().unwrap();
        let list = PyList::empty(py);

        let start_shard = self.current_shard;
        let end_shard = (start_shard + lookahead).min(dataset.num_shards());

        for shard_idx in start_shard..end_shard {
            let shard = &dataset.shards[shard_idx];
            let dict = PyDict::new(py);

            dict.set_item("index", shard_idx)?;
            dict.set_item("name", &shard.name)?;

            if dataset.is_remote && dataset.shard_cache.is_some() {
                let shard_name = shard.tar_path.rsplit('/').next().unwrap_or(&shard.tar_path);
                let cache = dataset.shard_cache.as_ref().unwrap().clone();
                let is_cached = self.runtime.block_on(cache.is_cached(shard_name));
                dict.set_item("is_cached", is_cached)?;
            } else {
                dict.set_item("is_cached", true)?;
            }

            list.append(dict)?;
        }

        Ok(list.into_any().unbind())
    }

    #[pyo3(signature = (num_shards=1))]
    fn prepare_shards_ahead(&self, num_shards: usize) -> PyResult<Vec<String>> {
        let dataset = self.dataset.lock().unwrap();
        let mut started_caching = Vec::new();

        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(started_caching);
        }

        let start_shard = self.current_shard;
        let end_shard = (start_shard + num_shards).min(dataset.num_shards());

        for shard_idx in start_shard..end_shard {
            let shard = &dataset.shards[shard_idx];
            let tar_path = shard.tar_path.clone();
            let shard_name = tar_path.rsplit('/').next().unwrap_or(&tar_path).to_string();
            let token = dataset.get_hf_token();
            let cache = dataset.shard_cache.as_ref().unwrap().clone();

            // Check if already cached
            let is_cached = self.runtime.block_on(cache.is_cached(&shard_name));
            if !is_cached {
                // Spawn async task to cache the shard
                let runtime = self.runtime.clone();
                let shard_name_clone = shard_name.clone();
                let tar_path_clone = tar_path.clone();
                let token_clone = token.clone();
                let cache_clone = cache.clone();
                runtime.spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    rt.block_on(async {
                        match cache_clone
                            .cache_shard(&shard_name_clone, &tar_path_clone, token_clone)
                            .await
                        {
                            Ok(_) => {
                                println!("[webshart] Pre-cached shard: {}", shard_name_clone);
                            }
                            Err(e) => {
                                eprintln!(
                                    "[webshart] Failed to pre-cache shard {}: {}",
                                    shard_name_clone, e
                                );
                            }
                        }
                    })
                });

                started_caching.push(shard_name);
            }
        }

        drop(dataset);
        Ok(started_caching)
    }

    // Getters
    #[getter]
    fn num_shards(&self) -> usize {
        self.dataset.lock().unwrap().num_shards()
    }

    #[getter]
    fn current_shard_index(&self) -> usize {
        self.current_shard
    }

    #[getter]
    fn current_shard_filename(&self) -> String {
        let dataset = self.dataset.lock().unwrap();
        if self.current_shard < dataset.shards.len() {
            let shard = &dataset.shards[self.current_shard];
            shard
                .tar_path
                .rsplit('/')
                .next()
                .unwrap_or(&shard.tar_path)
                .to_string()
        } else {
            String::new()
        }
    }

    #[getter]
    fn current_file_index(&self) -> usize {
        self.calculate_current_file_index()
    }

    #[getter]
    fn buffer_size(&self) -> usize {
        self.config.buffer_size
    }

    #[getter]
    fn chunk_size_mb(&self) -> usize {
        self.config.chunk_size_mb
    }

    #[getter]
    fn load_file_data(&self) -> bool {
        self.config.load_file_data
    }

    #[getter]
    fn max_file_size(&self) -> u64 {
        self.config.max_file_size
    }

    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.config.batch_size
    }

    // Setters
    #[setter]
    fn set_buffer_size(&mut self, size: usize) {
        self.config.buffer_size = size.max(1);
        if self.entry_buffer.capacity() < self.config.buffer_size {
            self.entry_buffer
                .reserve(self.config.buffer_size - self.entry_buffer.capacity());
        }
    }

    #[setter]
    fn set_chunk_size_mb(&mut self, size_mb: usize) {
        self.config.chunk_size_mb = size_mb.max(1);
    }

    #[setter]
    fn set_batch_size(&mut self, batch_size: Option<usize>) {
        self.config.batch_size = batch_size;
    }

    fn set_ranges(&mut self, ranges: Vec<(usize, usize)>) -> PyResult<()> {
        // Validate ranges
        for (start, end) in &ranges {
            if start >= end {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid range: start {} must be less than end {}",
                    start, end
                )));
            }
        }

        self.ranges = Some(ranges);
        self.current_range_idx = 0;

        // Jump to first range start
        if let Some(ranges) = &self.ranges {
            if !ranges.is_empty() {
                self.skip(ranges[0].0)?;
            }
        }

        Ok(())
    }

    fn get_metadata(&self, shard_idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let dict = PyDict::new(py);

        if let Some(metadata) = &shard.metadata {
            for (filename, file_info) in metadata
                .iter_files()
                .filter(|(_, file_info)| file_info.length <= self.config.max_file_size)
            {
                let file_dict = pythonize::pythonize(py, &file_info)?;
                dict.set_item(filename, file_dict)?;
            }
        }

        Ok(dict.into_any().unbind())
    }

    /// List visible samples with their stable, unfiltered sample indices.
    fn list_samples_in_shard(&self, shard_idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let list = PyList::empty(py);
        if let Some(metadata) = &shard.metadata {
            for (sample_idx, (filename, _)) in metadata
                .sample_range(0, metadata.num_samples())
                .into_iter()
                .enumerate()
                .filter(|(_, (_, file_info))| file_info.length <= self.config.max_file_size)
            {
                let sample = PyDict::new(py);
                sample.set_item("sample_idx", sample_idx)?;
                sample.set_item("filename", filename)?;
                list.append(sample)?;
            }
        }

        Ok(list.into_any().unbind())
    }

    /// Load a logical sample, or return None when it exceeds max_file_size.
    fn load_sample(&self, shard_idx: usize, sample_idx: usize) -> PyResult<Option<PyTarFileEntry>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let metadata = shard
            .metadata
            .as_ref()
            .ok_or_else(|| WebshartError::MetadataNotFound("Metadata not loaded".to_string()))?;

        let (filename, file_info) = metadata.get_sample_by_index(sample_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!(
                "Sample index {} out of range for shard {}",
                sample_idx, shard_idx
            ))
        })?;
        let txt_sidecar = metadata
            .get_txt_sidecar_by_sample_index(sample_idx)
            .map(|(_, file_info)| file_info);

        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };
        drop(dataset);

        if file_info.length > self.config.max_file_size {
            return Ok(None);
        }

        let data = if self.config.load_file_data {
            self.load_single_file_data(&tar_path, &file_info, is_remote, token.clone())?
        } else {
            Vec::new()
        };

        let mut entry = create_tar_entry(
            filename,
            &file_info,
            data,
            Some(shard_idx),
            Some(sample_idx),
        );
        entry.json_data =
            self.load_json_sidecar_bytes(&tar_path, &file_info, is_remote, token.clone())?;
        entry.captions = self.load_caption_value(
            &tar_path,
            &file_info,
            txt_sidecar.as_ref(),
            entry.json_data.as_deref(),
            is_remote,
            token,
        )?;
        Ok(Some(entry))
    }

    /// Load the first caption for a logical sample from metadata or a paired sidecar.
    fn load_caption(&self, shard_idx: usize, sample_idx: usize) -> PyResult<Option<String>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;
        let metadata = shard
            .metadata
            .as_ref()
            .ok_or_else(|| WebshartError::MetadataNotFound("Metadata not loaded".to_string()))?;
        let (_filename, file_info) = metadata.get_sample_by_index(sample_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!(
                "Sample index {} out of range for shard {}",
                sample_idx, shard_idx
            ))
        })?;
        let txt_sidecar = metadata
            .get_txt_sidecar_by_sample_index(sample_idx)
            .map(|(_, file_info)| file_info);
        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };
        drop(dataset);

        if file_info.length > self.config.max_file_size {
            return Ok(None);
        }

        let captions = self.load_caption_value(
            &tar_path,
            &file_info,
            txt_sidecar.as_ref(),
            None,
            is_remote,
            token,
        )?;
        Ok(captions.and_then(|value| value.first().map(str::to_owned)))
    }

    /// Fold sidecar captions into normal webshart metadata JSON files.
    #[pyo3(signature = (destination=None, shard_indices=None))]
    fn coalesce_caption_metadata(
        &self,
        py: Python,
        destination: Option<String>,
        shard_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let (num_shards, cache_dir) = {
            let dataset = self.dataset.lock().unwrap();
            (dataset.num_shards(), dataset.metadata_cache_dir())
        };
        let persist_to_cache = destination.is_none();
        let destination = match destination {
            Some(path) => Path::new(&path).to_path_buf(),
            None => cache_dir.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "enable_metadata_cache() first or provide destination",
                )
            })?,
        };
        let shard_indices = shard_indices.unwrap_or_else(|| (0..num_shards).collect());
        let mut output_paths = Vec::with_capacity(shard_indices.len());
        let mut captioned_samples = 0usize;
        let mut coalesced_samples = 0usize;

        for shard_idx in shard_indices {
            let (shard_name, tar_path, is_remote, token, mut metadata) = {
                let mut dataset = self.dataset.lock().unwrap();
                ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;
                let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
                    WebshartError::InvalidShardFormat(format!(
                        "Shard index {} out of range",
                        shard_idx
                    ))
                })?;
                let metadata = shard.metadata.clone().ok_or_else(|| {
                    WebshartError::MetadataNotFound("Metadata not loaded".to_string())
                })?;
                (
                    shard.name.clone(),
                    shard.tar_path.clone(),
                    dataset.is_remote,
                    dataset.get_hf_token(),
                    metadata,
                )
            };

            for sample_idx in 0..metadata.num_samples() {
                let Some((_filename, file_info)) = metadata.get_sample_by_index(sample_idx) else {
                    continue;
                };
                let had_caption = file_info.captions.is_some();
                let txt_sidecar = metadata
                    .get_txt_sidecar_by_sample_index(sample_idx)
                    .map(|(_, file_info)| file_info);
                if let Some(captions) = self.load_caption_value(
                    &tar_path,
                    &file_info,
                    txt_sidecar.as_ref(),
                    None,
                    is_remote,
                    token.clone(),
                )? {
                    captioned_samples += 1;
                    coalesced_samples += usize::from(!had_caption);
                    metadata.set_sample_captions(sample_idx, captions);
                }
            }

            let output_path = Self::metadata_output_path(&destination, &shard_name)?;
            if persist_to_cache {
                self.dataset
                    .lock()
                    .unwrap()
                    .replace_shard_metadata(shard_idx, metadata, true)?;
            } else {
                Self::write_metadata_file(&output_path, &metadata)?;
                self.dataset
                    .lock()
                    .unwrap()
                    .replace_shard_metadata(shard_idx, metadata, false)?;
            }
            output_paths.push(output_path.to_string_lossy().to_string());
        }

        let result = PyDict::new(py);
        result.set_item("shards", output_paths.len())?;
        result.set_item("captioned_samples", captioned_samples)?;
        result.set_item("coalesced_samples", coalesced_samples)?;
        result.set_item("files", output_paths)?;
        Ok(result.into_any().unbind())
    }

    fn load_sample_json(
        &self,
        py: Python,
        shard_idx: usize,
        sample_idx: usize,
    ) -> PyResult<Option<Py<PyBytes>>> {
        let Some(entry) = self.load_sample(shard_idx, sample_idx)? else {
            return Ok(None);
        };
        Ok(entry
            .json_data
            .as_ref()
            .map(|data| PyBytes::new(py, data).unbind()))
    }

    fn reset(&mut self) -> PyResult<()> {
        self.current_shard = 0;
        self.entry_buffer.clear();
        self.buffer_position = 0;
        self.next_file_to_load = 0;
        Ok(())
    }

    fn skip(&mut self, idx: usize) -> PyResult<()> {
        let total_files = self.dataset.lock().unwrap().total_files().map_err(|e| {
            WebshartError::DiscoveryFailed(format!("Failed to get total files: {}", e))
        })?;

        if idx > total_files {
            return Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range",
                idx
            ))
            .into());
        }

        self.entry_buffer.clear();
        self.buffer_position = 0;

        let (target_shard, remaining) = self.find_shard_for_index(idx)?;

        self.current_shard = target_shard;
        self.next_file_to_load = remaining;

        Ok(())
    }

    #[pyo3(signature = (shard_idx=None, filename=None, cursor_idx=None))]
    fn shard(
        &mut self,
        shard_idx: Option<usize>,
        filename: Option<String>,
        cursor_idx: Option<usize>,
    ) -> PyResult<()> {
        self.entry_buffer.clear();
        self.buffer_position = 0;

        let dataset = self.dataset.lock().unwrap();

        let target_shard = if let Some(idx) = shard_idx {
            idx
        } else if let Some(fname) = filename {
            self.find_shard_by_filename(&dataset, &fname)?
        } else {
            return Err(WebshartError::InvalidShardFormat(
                "Either shard_idx or filename must be provided".to_string(),
            )
            .into());
        };

        if target_shard >= dataset.num_shards() {
            return Err(WebshartError::InvalidShardFormat(format!(
                "Shard index {} out of range (0-{})",
                target_shard,
                dataset.num_shards() - 1
            ))
            .into());
        }

        drop(dataset);

        self.current_shard = target_shard;
        self.next_file_to_load = cursor_idx.unwrap_or(0);

        Ok(())
    }

    #[pyo3(signature = (shard_indices, key="aspect", target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2)))]
    pub fn list_shard_aspect_buckets(
        &self,
        py: Python,
        shard_indices: Vec<usize>,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: u32,
        round_to: Option<usize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let results = self.get_aspect_buckets_for_shards(
            shard_indices,
            key,
            target_pixel_area,
            Some(target_resolution_multiple),
            round_to,
        )?;

        let py_results: Vec<Py<PyAny>> = results
            .into_iter()
            .map(|bucket| self.aspect_buckets_to_py_dict(py, bucket))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py_results)
    }

    #[pyo3(signature = (shard_indices, key="aspect", target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2)))]
    pub fn list_shard_sample_aspect_buckets(
        &self,
        py: Python,
