use std::{cmp, fmt::Display, path};

use itertools::Itertools;
use pyo3::pyclass;
use serde;

mod chime;

#[pyclass]
struct Transient {
    magnitude: Vec<f64>,
    time: Vec<f64>,
    noise: f64,
}

impl Display for Transient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Transient]")?;
        write!(
            f,
            "magnitude: {} [{},{},{}]",
            self.magnitude.len(),
            self.magnitude.iter().fold(f64::INFINITY, |i, &n| i.min(n)),
            self.magnitude.iter().sum::<f64>() / self.magnitude.len() as f64,
            self.magnitude.iter().fold(f64::MIN, |i, &n| i.max(n))
        )?;
        write!(
            f,
            "time: {} [{},{}]",
            self.time.len(),
            self.time.iter().fold(f64::INFINITY, |i, &n| i.min(n)),
            self.time.iter().fold(f64::MIN, |i, &n| i.max(n))
        )?;
        write!(f, "noise: {}", self.noise)
    }
}

/// Metadata for transient pointing to relevant files
///
/// The data is expected to be stored in the following strcuture:
/// ```sh
///  DIR/
///  └── NAME/
///      ├── magnitude.file
///      └── time.file
/// ```
#[derive(serde::Serialize)]
pub(crate) struct TransientToml {
    name: String,
    /// Directory containing data
    dir: path::PathBuf,
    /// Burst rate (per hour)
    rate: f64,
    /// Filename containing magnitude data
    magnitude: String,
    /// Filename containing time data
    time: String,
}
