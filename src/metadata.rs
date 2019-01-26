use crate::discovery::DiscoveredDataset;
use crate::error::{Result, WebshartError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Caption metadata extracted from paired JSON sidecars.
///
/// This serializes as either a single string or a list of strings so Python
/// callers see `captions: str | list[str] | None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaptionValue {
    Single(String),
    Multiple(Vec<String>),
}

impl CaptionValue {
    pub fn first(&self) -> Option<&str> {
        match self {
            Self::Single(text) => Some(text.as_str()),
            Self::Multiple(captions) => captions.first().map(String::as_str),
        }
    }
}

/// Information about a single file within a tar shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// Name/path of the file (optional in JSON, as it may be the HashMap key)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    #[serde(alias = "fname", alias = "filename")]
    pub path: Option<String>,

    /// Offset within the tar file
    pub offset: u64,

    /// Length of the file in bytes
    #[serde(alias = "size")]
    pub length: u64,

    /// SHA256 hash of the file (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// Image width in pixels (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,

    /// Image height in pixels (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,

    /// Image aspect ratio (width / height) (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect: Option<f32>,

    /// Path of a paired JSON metadata file inside the same tar shard.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_path: Option<String>,

    /// Offset of the paired JSON metadata file inside the tar shard.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_offset: Option<u64>,

    /// Length of the paired JSON metadata file in bytes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_length: Option<u64>,

    /// Caption field extracted from paired JSON metadata.
    #[serde(skip_serializing_if = "Option::is_none", default, alias = "caption")]
    pub captions: Option<CaptionValue>,

    /// Parsed paired JSON metadata when it is stored directly in the index.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_metadata: Option<Value>,
}

/// Metadata for a single shard - supports both HashMap and Vec formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShardMetadataFormat {
    /// Standard format with HashMap
