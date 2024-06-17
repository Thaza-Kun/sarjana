use clap::Parser;
use pyo3::prelude::*;

#[derive(Parser, Clone)]
#[allow(non_snake_case)]
#[pyclass]
pub struct Arguments {
    // Required because clap interprets `file.py` from `python file.py` as the first argument
    #[cfg(feature = "pyargs")]
    pub(crate) _pyfile: String,
    #[arg(short, long)]
    #[pyo3(get)]
    pub parent: std::path::PathBuf,
    #[arg(short, long)]
    #[pyo3(get)]
    pub name: String,
    #[arg(short, long, default_value_t = 42)]
    #[pyo3(get)]
    pub seed: u8,
    #[arg(short, long, default_value_t = 1_000)]
    #[pyo3(get)]
    pub grid: u128,
    #[arg(short, long)]
    #[pyo3(get)]
    pub outdir: std::path::PathBuf,
    #[arg(long, default_value_t = 16)]
    #[pyo3(get)]
    pub harmonics: u8,
    #[arg(short, long, default_value = None)]
    #[pyo3(get)]
    pub freq_grid: Option<std::path::PathBuf>,
    #[arg(long, default_value = None)]
    #[pyo3(get)]
    pub rate: Option<f64>,
    #[arg(long, default_value_t = 0., allow_hyphen_values = true)]
    #[pyo3(get)]
    pub min_SNR: f32,
    #[arg(long, default_value_t = 0., allow_hyphen_values = true)]
    #[pyo3(get)]
    pub min_power: f32,
    #[arg(short, long, default_value_t = 100)]
    #[pyo3(get)]
    pub runs: u128,
    #[arg(long, default_value = None)]
    #[pyo3(get)]
    pub period: Option<f64>,
    #[arg(long, default_value_t = 1.)]
    #[pyo3(get)]
    pub snr_scale: f64,
    #[arg(long)]
    #[pyo3(get)]
    pub periodogram: Option<String>,
}

#[pyfunction]
pub fn parse_arguments() -> PyResult<Arguments> {
    Ok(Arguments::parse())
}
