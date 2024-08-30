use crate::python;
use pyo3::{
    types::{IntoPyDict, PyAny, PyAnyMethods, PyDict},
    FromPyObject, IntoPy, Py, PyResult,
};

#[cfg(unix)]
fn healpix_index<T: IntoPy<Py<PyAny>> + Copy>(nside: usize, ra: T, dec: T) -> usize {
    python::just_panic(|py| -> PyResult<usize> {
        let hp = py.import_bound("healpy")?;
        hp.call_method(
            "ang2pix",
            (nside, ra, dec),
            Some(&[("lonlat", true)].into_py_dict_bound(py)),
        )?
        .extract::<usize>()
    })
}
