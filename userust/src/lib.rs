mod args;
mod diagnostic;
mod ensemble;
mod experiment;
mod transient;
// mod exposure;
pub mod python;
// mod periodogram;

use kdam::{par_tqdm, tqdm};
use pyo3::{prelude::*, types::PyList};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

mod timeseries {
    use std::borrow::Cow;

    use pyo3::{pyclass, pymethods};
    use rand::prelude::SliceRandom;

    use crate::ensemble::Generator;

    #[pyclass]
    #[derive(Clone)]
    pub struct TimeSeries {
        pub time: Cow<'static, [f64]>,
        pub timeunit: Option<String>,
        pub magnitude: Cow<'static, [f64]>,
        pub magunit: Option<String>,
    }

    #[pymethods]
    impl TimeSeries {
        #[getter]
        pub fn time(&self) -> Vec<f64> {
            self.time.to_vec().clone()
        }
        #[getter]
        pub fn magnitude(&self) -> Vec<f64> {
            self.magnitude.to_vec().clone()
        }
        #[new]
        pub fn new(time: Vec<f64>, unit: Option<String>) -> Self {
            Self {
                time: time.into(),
                timeunit: unit,
                magnitude: Cow::Owned(Vec::new()),
                magunit: None,
            }
        }

        pub fn with_magnitudes(&self, magnitude: Vec<f64>, unit: Option<String>) -> Self {
            Self {
                magnitude: magnitude
                    .iter()
                    .take(self.time.len())
                    .map(|a| *a)
                    .collect::<Vec<f64>>()
                    .into(),
                magunit: unit,
                ..self.clone()
            }
        }

        pub fn sample_from_values(&self, values: Vec<f64>, rng: &mut Generator) -> Self {
            Self {
                magnitude: values
                    .choose_multiple(&mut rng.rng, self.time.len())
                    .map(|a| *a)
                    .collect::<Vec<f64>>()
                    .into(),
                magunit: None,
                ..self.clone()
            }
        }
    }
}

#[pyfunction]
fn load_many_numpyz(files: Vec<String>, key: String, index: usize) -> PyResult<Vec<f64>> {
    let res = par_tqdm!(files.into_par_iter(), leave = false)
        .map(|file| match npyz::npz::NpzArchive::open(file) {
            Ok(mut f) => match f.by_name(&key) {
                Ok(Some(npv)) => match npv.data::<f64>() {
                    Ok(v) => {
                        return v
                            .into_iter()
                            // .unwrap()
                            .skip(index)
                            .take(1)
                            .collect::<Result<Vec<f64>, std::io::Error>>()
                            .unwrap()[0];
                    }
                    Err(_) => f64::NAN,
                },
                Ok(None) | Err(_) => f64::NAN,
            },
            Err(_) => f64::NAN,
        })
        .collect::<Vec<f64>>();
    Ok(res)
}

#[pymodule]
#[pyo3(name = "main")]
fn _main(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<args::Arguments>()?;
    m.add_function(wrap_pyfunction!(args::parse_arguments, m)?)?;
    m.add_class::<ensemble::Ensemble>()?;
    m.add_function(wrap_pyfunction!(
        ensemble::generate_periodogram_ensembles,
        m
    )?)?;
    m.add_class::<ensemble::Generator>()?;
    m.add_function(wrap_pyfunction!(
        ensemble::generate_timeseries_subsample,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ensemble::generate_signal_filter, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble::snr::greedy_harmonic_sum, m)?)?;
    m.add_function(wrap_pyfunction!(load_many_numpyz, m)?)?;
    Ok(())
}
