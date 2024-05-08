use std::collections::HashMap;

use clap::Parser;
use pyo3::{prelude::*, types::PyFunction};

use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Parser, Clone)]
#[allow(non_snake_case)]
#[pyclass]
struct Arguments {
    // Required because clap interprets `file.py` from `python file.py` as the first argument
    _pyfile: String,
    #[arg(short, long)]
    #[pyo3(get)]
    parent: std::path::PathBuf,
    #[arg(short, long)]
    #[pyo3(get)]
    name: String,
    #[arg(short, long, default_value_t = 42)]
    #[pyo3(get)]
    seed: u8,
    #[arg(short, long, default_value_t = 1_000)]
    #[pyo3(get)]
    grid: u128,
    #[arg(short, long)]
    #[pyo3(get)]
    outdir: std::path::PathBuf,
    #[arg(long, default_value_t = 16)]
    #[pyo3(get)]
    harmonics: u8,
    #[arg(short, long)]
    #[pyo3(get)]
    freq_grid: std::path::PathBuf,
    #[arg(long, default_value_t = 25.)]
    #[pyo3(get)]
    min_SNR: f32,
    #[arg(long, default_value_t = 0.)]
    #[pyo3(get)]
    min_power: f32,
    #[arg(short, long, default_value_t = 100)]
    #[pyo3(get)]
    runs: u128,
    #[arg(long, default_value = None)]
    #[pyo3(get)]
    period: Option<f64>,
}

#[pyfunction]
fn parse_arguments() -> PyResult<Arguments> {
    Ok(Arguments::parse())
}

#[derive(Clone, serde::Deserialize, serde::Serialize, Debug)]
#[pyclass]
struct Ensemble {
    runs: u128,
    #[pyo3(get)]
    power: Vec<f64>,
    #[pyo3(get)]
    snr: Vec<f64>,
    #[pyo3(get)]
    freq: Vec<f64>,
}

#[pymethods]
impl Ensemble {
    #[new]
    fn new() -> Self {
        Ensemble {
            runs: 0,
            power: vec![],
            snr: vec![],
            freq: vec![],
        }
    }
    fn filter(&mut self, power: Option<f64>, snr: Option<f64>) {
        if let Some(snr_) = snr {
            let filter = Vec::from_iter(self.snr.iter())
                .iter()
                .map(|&&i| i >= snr_)
                .collect::<Vec<bool>>();
            self.snr = self
                .snr
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<f64>>();
            self.power = self
                .power
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<f64>>();
            self.freq = self
                .freq
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<f64>>();
        }
        if let Some(power_) = power {
            let filter = Vec::from_iter(self.snr.iter())
                .iter()
                .map(|&&i| i >= power_)
                .collect::<Vec<bool>>();
            self.snr = self
                .snr
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<f64>>();
            self.power = self
                .power
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<f64>>();
            self.freq = self
                .freq
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<f64>>();
        }
    }
}

impl Ensemble {
    fn append(&mut self, power: &[f64], snr: &[f64], freq: &[f64]) {
        self.power.extend_from_slice(power);
        self.snr.extend_from_slice(snr);
        self.freq.extend_from_slice(freq);
        self.runs += 1;
    }
}

#[pyclass]
struct NdArray(Vec<f64>);

impl NdArray {
    fn select(&mut self, indices: &[u128]) -> NdArray {
        NdArray(
            self.0
                .clone()
                .iter()
                .enumerate()
                .filter(|(i, _f)| indices.contains(&(*i as u128)))
                .map(|(_i, f)| *f)
                .collect::<Vec<f64>>(),
        )
    }
}

// struct RandNumGen(StdRng);

// impl RngCore for RandNumGen {
//     fn next_u32(&mut self) -> u32 {
//         self.0.next_u32()
//     }

//     fn next_u64(&mut self) -> u64 {
//         self.0.next_u64()
//     }

//     fn fill_bytes(&mut self, dest: &mut [u8]) {
//         self.0.fill_bytes(dest)
//     }

//     fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
//         self.0.try_fill_bytes(dest)
//     }
// }

// impl Rng for RandNumGen {}

// impl RandNumGen {
//     fn with_seed(seed: u64) -> RandNumGen {
//         RandNumGen(StdRng::seed_from_u64(seed))
//     }
// }

#[pyfunction]
fn iterate_periodogram(
    py: Python<'_>,
    sim_signal: PyObject,
    sim_time: PyObject,
    frb_signal: PyObject,
    frb_time: PyObject,
    view_index: u16,
    view_length: u16,
    detection_rate: f64,
    arguments: Arguments,
    seed: usize,
    periodogram: Bound<'_, PyFunction>,
    snrfunc: Bound<'_, PyFunction>,
    find_peak: Bound<'_, PyFunction>,
    mut sim_ensemble: Ensemble,
    mut frb_ensemble: Ensemble,
    freq_grid: PyObject,
) -> PyResult<(Ensemble, Ensemble)> {
    // for _ in tqdm(range(runs)):
    let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
    let sim_signal = sim_signal.extract::<Vec<f64>>(py)?;
    let prepend = std::iter::repeat(false)
        .take(view_index.into())
        // .into_iter()
        .collect::<Vec<bool>>();
    let append = std::iter::repeat(false)
        .take(sim_signal.len() - (view_index + view_length) as usize)
        .collect::<Vec<bool>>();
    for _ in tqdm::tqdm(0..arguments.runs) {
        let filter: Vec<bool> = (0..view_length)
            .map(|_| {
                [
                    (false, 1. - detection_rate as f32),
                    (true, detection_rate as f32),
                ]
                .choose_weighted(&mut rng, |item| item.1)
                .expect("Rand error")
                .0
            })
            .into_iter()
            .collect::<Vec<bool>>();
        let mut signal = Vec::clone(&prepend);
        signal.extend_from_slice(&filter);
        signal.extend(&append);
        assert_eq!(sim_signal.len(), signal.len());
        let sim_signal = sim_signal
            .iter()
            .zip(signal)
            .map(|(sim, filter)| if filter { 0. } else { *sim })
            .collect::<Vec<f64>>();

        {
            let power = periodogram
                .call(
                    (sim_time.clone(), sim_signal.clone(), freq_grid.clone()),
                    None,
                )?
                .extract::<Vec<f64>>()?;
            // TODO: Greedy Harmonic Sum in Rust
            let snr = snrfunc
                .call((power.clone(), freq_grid.clone()), None)?
                .extract::<Vec<f64>>()?;
            let (peaks, _props) = find_peak
                .call((power.clone(),), None)?
                .extract::<(Vec<u128>, HashMap<String, Vec<f64>>)>()?;

            sim_ensemble.append(
                &NdArray(power).select(peaks.as_slice()).0,
                &NdArray(snr).select(peaks.as_slice()).0,
                &NdArray(freq_grid.getattr(py, "value")?.extract::<Vec<f64>>(py)?)
                    .select(peaks.as_slice())
                    .0,
            );
        }

        {
            let power = periodogram
                .call(
                    (frb_time.clone(), frb_signal.clone(), freq_grid.clone()),
                    None,
                )?
                .extract::<Vec<f64>>()?;
            let snr = snrfunc
                .call((power.clone(), freq_grid.clone()), None)?
                .extract::<Vec<f64>>()?;
            let (peaks, _props) = find_peak
                .call((power.clone(),), None)?
                .extract::<(Vec<u128>, HashMap<String, Vec<f64>>)>()?;

            frb_ensemble.append(
                &NdArray(power).select(peaks.as_slice()).0,
                &NdArray(snr).select(peaks.as_slice()).0,
                &NdArray(freq_grid.getattr(py, "value")?.extract::<Vec<f64>>(py)?)
                    .select(peaks.as_slice())
                    .0,
            );
        }
    }
    Ok((sim_ensemble, frb_ensemble))
    // todo!()
}
//     sim_frb.signal = np.array(
//         list(
//             map(
//                 lambda x: x[0] if x[1] else 0,
//                 zip(noised, sim_frb.signal.astype(bool)),
//             )
//         )
//     )

//     # create freq_grid if not provided
//     detect_on = sim_frb.telescope_observations[sim_frb.signal.astype(bool)]
//     if len(detect_on) <= 2:
//         continue
//     if arguments.freq_grid is None:
//         burst_rate = len(detect_on) / ((detect_on.max() - detect_on.min()) * 24)
//         freq_min = 0.5 * burst_rate * (1 / u.hour)
//         freq_max = 0.0001 * (1 / u.hour)
//         freq_grid = np.linspace(freq_max, freq_min, arguments.grid)

//     # Simulated
//     power = LombScargle_periodogram(sim_frb.time, sim_frb.signal, freq_grid)
//     SNR = snr_greedy_harmonic_sum(power, freq_grid)
//     peaks, _ = find_peaks(power)

//     sim_ensemble.append(power[peaks], SNR[peaks], freq_grid.value[peaks])

//     # FRB Data
//     power = LombScargle_periodogram(
//         thisfrb.time, thisfrb.telescope_signal, freq_grid
//     )
//     SNR = snr_greedy_harmonic_sum(power, freq_grid)
//     peaks, _ = find_peaks(power)

//     frb_ensemble.append(power[peaks], SNR[peaks], freq_grid.value[peaks])

#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_arguments, m)?)?;
    m.add_function(wrap_pyfunction!(iterate_periodogram, m)?)?;
    m.add_class::<Arguments>()?;
    m.add_class::<Ensemble>()?;
    Ok(())
}
