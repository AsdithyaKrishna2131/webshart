use crate::{metadata::CaptionValue, FileInfo};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyclass(name = "TarFileEntry", from_py_object)]
#[derive(Clone)]
pub struct PyTarFileEntry {
    pub path: String,
    pub offset: u64,
    pub size: u64,
    pub data: Vec<u8>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub aspect: Option<f32>,
    pub json_path: Option<String>,
    pub json_data: Option<Vec<u8>>,
    pub captions: Option<CaptionValue>,
    pub json_metadata: Option<serde_json::Value>,
    pub shard_idx: Option<usize>,
    pub file_idx: Option<usize>,
}

#[pymethods]
impl PyTarFileEntry {
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }

    #[getter]
    fn size(&self) -> u64 {
        self.size
    }

    #[getter]
    fn data(&self) -> PyResult<Py<PyBytes>> {
        Python::attach(|py| Ok(PyBytes::new(py, &self.data).unbind()))
    }

    #[getter]
    fn width(&self) -> Option<u32> {
        self.width
    }
