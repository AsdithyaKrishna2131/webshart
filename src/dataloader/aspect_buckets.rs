use crate::dataloader::PyTarDataLoader;
use crate::error::{Result, WebshartError};
use crate::FileInfo;
use pyo3::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct AspectBuckets {
    pub buckets: BTreeMap<String, Vec<AspectBucketEntry>>,
    pub shard_idx: usize,
    pub shard_name: String,
}

#[derive(Debug, Clone)]
pub struct AspectBucketEntry {
    pub filename: String,
    pub file_info: FileInfo,
    pub original_size: Option<(u32, u32)>,
    pub sample_idx: Option<usize>,
}

#[pyclass]
pub struct AspectBucketIterator {
    pub loader: Py<PyTarDataLoader>,
    pub key_type: String,
    pub target_pixel_area: Option<u32>,
    pub target_resolution_multiple: u32,
    pub round_to: Option<usize>,
    pub current_shard: usize,
    pub num_shards: usize,
}

#[pymethods]
impl AspectBucketIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        if slf.current_shard >= slf.num_shards {
            return Ok(None);
        }

        Python::attach(|py| {
            let shard_idx = slf.current_shard;
            slf.current_shard += 1;
            let loader = slf.loader.borrow(py);
            let result = loader.list_shard_aspect_buckets(
                py,
                vec![shard_idx],
                &slf.key_type,
                slf.target_pixel_area,
                slf.target_resolution_multiple,
                slf.round_to,
            )?;

            Ok(result.into_iter().next())
        })
    }
}

pub fn scale_dimensions_with_multiple(
    width: u32,
    height: u32,
    target_pixel_area: u32,
    multiple: u32,
) -> (u32, u32) {
    // Target resolution is the desired total area (width * height)
    // let aspect_ratio = width as f64 / height as f64;
    let current_area = (width as f64) * (height as f64);
