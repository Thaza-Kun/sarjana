use std::sync::Arc;

use pyo3::{prelude::*, types::PyFunction};

mod args;
mod ensemble;

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
    m.add_function(wrap_pyfunction!(ensemble::generate_signal_filter, m)?)?;
    Ok(())
}
