use pyo3::{
    prepare_freethreaded_python, types::PyAnyMethods, IntoPy, Py, PyAny, PyErr, PyResult, Python,
};

pub trait DunderMethods: Sized {
    fn add<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self>;
    fn sub<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self>;
    fn mul<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self>;
    fn matmul<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self>;
    fn div<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self>;
}

impl DunderMethods for PyAny {
    fn add<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self> {
        self.call_method1("__add__", (args,))
    }
    fn sub<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self> {
        self.call_method1("__sub__", (args,))
    }
    fn mul<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self> {
        self.call_method1("__mul__", (args,))
    }
    fn div<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self> {
        self.call_method1("__truediv__", (args,))
    }
    fn matmul<'py, T: IntoPy<Py<PyAny>>>(&'py self, args: T) -> PyResult<&'py Self> {
        self.call_method1("__matmul__", (args,))
    }
}

pub fn with_virtualenv(env_dir: &str) {
    prepare_freethreaded_python();
    just_panic(|py| {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;

        path.call_method1("append", (format!("{}\\Lib\\site-packages", env_dir),))?;
        Ok::<(), PyErr>(())
    });
}

pub fn just_panic<'py, T: 'py, F: Fn(Python) -> PyResult<T>>(func: F) -> T {
    match Python::<'py>::with_gil(func) {
        Ok(t) => t,
        Err(e) => panic!("{:?}", e),
    }
}
