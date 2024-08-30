use std::ops::Deref;

use pyo3::{
    prelude::PyAnyMethods, pyclass, pyfunction, pymethods, types::PyFunction, Bound, PyAny,
    PyObject, PyResult, Python,
};

enum Function<'py> {
    Native(fn(f64, f64, f64) -> (Vec<f64>, Vec<f64>)),
    Python(Bound<'py, PyFunction>),
}

impl Function<'_> {
    fn call(self, a: f64, b: f64, c: f64) -> (Vec<f64>, Vec<f64>) {
        match self {
            Function::Native(function) => function(a, b, c),
            Function::Python(function) => function
                .call((a, b, c), None)
                .unwrap()
                .extract::<(Vec<f64>, Vec<f64>)>()
                .unwrap(),
        }
    }
}

#[pyfunction]
unsafe fn new<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
    Bound::from_borrowed_ptr(py, Periodogram::new(|_, _, _| (vec![], vec![])))
}

struct Periodogram {
    function: Function<'static>,
}

impl Deref for Periodogram {
    type Target = PyObject;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

impl Into<PyAny> for Periodogram {
    fn into(self) -> PyAny {
        todo!()
    }
}

impl Periodogram {
    fn new(function: fn(f64, f64, f64) -> (Vec<f64>, Vec<f64>)) -> Self {
        Self {
            function: function.into(),
        }
    }
}
impl From<fn(f64, f64, f64) -> (Vec<f64>, Vec<f64>)> for Function<'_> {
    fn from(value: fn(f64, f64, f64) -> (Vec<f64>, Vec<f64>)) -> Self {
        Self::Native(value)
    }
}
impl<'py> From<Bound<'py, PyFunction>> for Function<'py> {
    fn from(value: Bound<'py, PyFunction>) -> Self {
        Self::Python(value)
    }
}
