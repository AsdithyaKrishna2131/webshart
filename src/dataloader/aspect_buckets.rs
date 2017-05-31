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
