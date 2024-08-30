use crate::{ensemble::snr, transient};

pub(crate) struct PdgramDiagnostic<'a> {
    transient: &'a transient::TransientToml,
    periodogram: &'a PdgramConfig,
    snr: &'a SignalNoiseRatio,
    min_snr: f64,
    min_power: f64,
}

struct PdgramConfig {
    name: String,
    freq_grid: Vec<f64>,
}

struct SignalNoiseRatio {
    pub name: String,
    pub harmonics: u8,
    pub scale: f64,
    pub function: fn(Vec<f64>, Vec<f64>, u8) -> Vec<f64>,
}
