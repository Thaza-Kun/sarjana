use crate::timeseries::TimeSeries;
use kdam::tqdm;
use pyo3::{prelude::*, types::PyFunction};
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use ndarray::prelude::*;
use numpy::{PyArray1, PyArrayMethods};

#[derive(Clone, serde::Deserialize, serde::Serialize, Debug)]
#[pyclass]
pub struct Ensemble {
    pub(crate) runs: u128,
    #[pyo3(get)]
    pub power: Vec<f64>,
    #[pyo3(get)]
    pub snr: Vec<f64>,
    #[pyo3(get)]
    pub freq: Vec<f64>,
    #[pyo3(get)]
    pub group: Vec<u128>,
}

#[pymethods]
impl Ensemble {
    #[staticmethod]
    fn empty() -> Self {
        Ensemble {
            runs: 0,
            power: vec![],
            snr: vec![],
            freq: vec![],
            group: vec![],
        }
    }

    #[new]
    fn new(power: Vec<f64>, snr: Vec<f64>, freq: Vec<f64>, group: Vec<u128>) -> Self {
        Ensemble {
            runs: *group
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap(),
            power,
            snr,
            freq,
            group,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
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
            self.group = self
                .group
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<u128>>();
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
            self.group = self
                .group
                .iter()
                .zip(&filter)
                .filter(|(_s, f)| f == &&true)
                .map(|(&s, _f)| s)
                .collect::<Vec<u128>>();
        }
    }
}

impl Ensemble {
    pub fn append(&mut self, power: &[f64], snr: &[f64], freq: &[f64]) {
        self.power.extend_from_slice(power);
        self.snr.extend_from_slice(snr);
        self.freq.extend_from_slice(freq);
        self.runs += 1;
        self.group.extend_from_slice(
            std::iter::repeat(self.runs)
                .take(power.len())
                .collect::<Vec<u128>>()
                .as_slice(),
        )
    }
}

pub mod snr {
    use kdam::par_tqdm;
    use ndarray::Array1;
    use pyo3::pyfunction;
    use rayon::{iter::ParallelIterator, prelude::*};
    use std::{
        cmp,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn get_snr(sums: f64, mean: f64, std: f64) -> f64 {
        (sums - mean) / std
    }

    fn compare_snr(snr: f64, harmonics: usize, snr_max: f64, harmonics_m: f64) -> (f64, f64) {
        (
            cmp::max_by(snr, snr_max, |x, y| {
                x.partial_cmp(y).unwrap_or(cmp::Ordering::Greater)
            }),
            cmp::max_by(harmonics as f64, harmonics_m, |x, y| {
                x.partial_cmp(y).unwrap_or(cmp::Ordering::Greater)
            }),
        )
    }

    #[pyfunction]
    pub fn greedy_harmonic_sum(power: Vec<f64>, grid: Vec<f64>, harmonics: u8) -> Vec<f64> {
        greedy_harmonic_sum_(
            Array1::from_vec(power),
            Array1::from_vec(grid),
            harmonics as usize,
            1.,
        )
        .to_vec()
    }

    pub fn greedy_harmonic_sum_(
        power: Array1<f64>,
        grid: Array1<f64>,
        harmonics: usize,
        scale: f64,
    ) -> Array1<f64> {
        let vecpower = power.to_vec();
        let arrpower = power.clone();
        scale
            * Array1::from_vec(
                par_tqdm!((0..grid.len()).into_par_iter(), position = 1)
                    .map(|i| {
                        let mut let_sum = 0.;
                        let d = AtomicUsize::new(0);
                        let snr_max;
                        let harmonics_m = 0.;
                        // fundamendal bin
                        let x_d = vecpower[i];
                        let x_d_plus_1 = vecpower[cmp::min(i + 1, grid.len() - 1)];
                        if x_d_plus_1 > x_d {
                            d.fetch_add(1, Ordering::SeqCst);
                            let_sum += x_d_plus_1;
                        } else {
                            let_sum += x_d;
                        }
                        snr_max = get_snr(let_sum, arrpower.mean().unwrap_or(0.), arrpower.std(1.));
                        // Higher harmonics
                        (0..harmonics)
                            .into_par_iter()
                            .map(|_| {
                                let mut inner_sum = let_sum.clone();
                                let x_d = vecpower[cmp::min(
                                    harmonics * i + d.load(Ordering::SeqCst),
                                    grid.len() - 1,
                                )];
                                let x_d_plus_1 = vecpower[cmp::min(
                                    harmonics * i + d.load(Ordering::SeqCst) + 1,
                                    grid.len() - 1,
                                )];
                                if x_d_plus_1 > x_d {
                                    d.fetch_add(1, Ordering::SeqCst);
                                    inner_sum += x_d_plus_1
                                } else {
                                    inner_sum += x_d
                                }
                                let snr = get_snr(
                                    inner_sum,
                                    arrpower.mean().unwrap_or(0.),
                                    arrpower.std(1.),
                                );
                                let (snr_max, _harmonics_m) =
                                    compare_snr(snr, harmonics, snr_max, harmonics_m);
                                snr_max
                            })
                            .max_by(|x, y| x.partial_cmp(y).unwrap_or(cmp::Ordering::Equal))
                            .expect("Error while getting max")
                    })
                    .collect::<Vec<f64>>(),
            )
    }
}

#[macro_export]
macro_rules! timeit {
    ($name:literal <- $evaluatee:expr) => {{
        // let now = std::time::Instant::now();
        let res = $evaluatee;
        // println!("{}\t: {:.5} s", $name, now.elapsed().as_secs());
        res
        // }
    }};
}

fn evaluate_periodogram(
    py: Python<'_>,
    periodogram: Bound<'_, PyFunction>,
    find_peak: Bound<'_, PyFunction>,
    ensemble: &mut Ensemble,
    time: Vec<f64>,
    signal: Vec<f64>,
    freq_grid: PyObject,
    // arguments: Arguments,
    harmonics: usize,
    snr_scale: f64,
    inspections: Bound<'_, PyFunction>,
) -> PyResult<()> {
    let res = periodogram
        .call((time.clone(), signal.clone(), freq_grid.clone()), None)?
        .extract::<(Vec<f64>, Vec<f64>)>()?;
    let power = timeit! {"Sim-power" <- Array1::from_vec(res.0)};
    let freq_grid = timeit! {"Sim-freq_grid" <- Array1::from_vec(res.1)};
    let snr = timeit! {
        "Sim-SignalNoise" <- snr::greedy_harmonic_sum_(power.clone(), freq_grid.clone(), harmonics, snr_scale)
    };
    inspections.call1((
        signal.to_vec(),
        time.to_vec(),
        power.to_vec(),
        snr.to_vec(),
        freq_grid.clone().to_vec(),
        ensemble.runs.clone(),
    ))?;
    let (peaks, _props) = timeit! {"Sim-Peak" <- find_peak
    .call((power.to_vec().clone(),), None)?
        .extract::<(Vec<usize>, HashMap<String, Vec<f64>>)>()?};

    timeit! {"Sim-Ensemble" <- ensemble.append(
        &power.select(Axis(0), peaks.as_slice()).as_slice().ok_or(PyErr::fetch(py))?,
        &snr.select(Axis(0), peaks.as_slice()).as_slice().ok_or(PyErr::fetch(py))?,
        &freq_grid.select(Axis(0), peaks.as_slice()).as_slice().ok_or(PyErr::fetch(py))?)
    }
    Ok(())
}

#[pyclass]
pub struct Generator {
    pub rng: ChaCha8Rng,
}

#[pymethods]
impl Generator {
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

#[pyfunction]
pub fn generate_timeseries_subsample(
    timeseries: Vec<f64>,
    sample: Vec<f64>,
    n: usize,
    rng: &mut Generator,
) -> TimeSeries {
    let subseries = if n <= timeseries.len() {
        timeseries
            .choose_multiple(&mut rng.rng, n)
            .map(|a| a.to_owned())
            .collect::<Vec<f64>>()
    } else {
        timeseries
    };
    let tseries = TimeSeries::new(subseries, None);
    tseries.sample_from_values(sample, rng)
}

#[pyfunction]
pub fn generate_signal_filter(
    length: usize,
    probability: f32,
    prepend: Vec<bool>,
    append: Vec<bool>,
    rng: &mut Generator,
) -> Vec<bool> {
    let mut filter = Vec::clone(&prepend);
    filter.extend_from_slice(
        &(0..length)
            .map(|_| {
                [(true, probability), (false, 1. - probability)]
                    .choose_weighted(&mut rng.rng, |i| i.1)
                    .unwrap_or(&(false, 0.))
                    .0
            })
            .into_iter()
            .collect::<Vec<bool>>(),
    );
    filter.extend(&append);
    filter
}

#[pyfunction]
// #[cfg(feature = "pyargs")]
pub fn generate_periodogram_ensembles(
    py: Python<'_>,
    sim_signal: Bound<'_, PyArray1<f64>>,
    // astropy.Time
    sim_time: Bound<'_, PyArray1<f64>>,
    time_length: usize,
    // frb_signal: Bound<'_, PyArray1<f64>>,
    // astropy.Time
    // frb_time: Bound<'_, PyArray1<PyAny>>,
    // view_index: u16,
    // view_length: u16,
    runs: usize,
    seed: usize,
    periodogram: Bound<'_, PyFunction>,
    find_peak: Bound<'_, PyFunction>,
    py_freq_grid: PyObject,
    harmonics: usize,
    snr_scale: f64,
    inspections: Bound<'_, PyFunction>,
) -> PyResult<Ensemble> {
    let sim_signal = sim_signal.to_vec()?;
    let sim_time = sim_time.to_vec()?;
    let mut rng = Generator::new(seed as u64);
    let mut sim_ensemble = Ensemble::empty();

    for iternum in tqdm!(0..runs, position = 0) {
        // let filter: Vec<bool> = generate_signal_filter(
        //     view_length as usize,
        //     detection_rate as f32,
        //     prepend.clone(),
        //     append.clone(),
        //     &mut rng,
        // );
        let noise = generate_timeseries_subsample(
            sim_time.clone(),
            sim_signal.clone(),
            time_length,
            &mut rng,
        );
        // sim_signal
        //     .iter()
        //     .zip(filter)
        //     .map(|(sim, filter)| if filter { 0. } else { *sim })
        //     .collect::<Vec<f64>>();

        evaluate_periodogram(
            py,
            periodogram.clone(),
            find_peak.clone(),
            &mut sim_ensemble,
            noise.time.to_vec(),
            noise.magnitude.to_vec(),
            py_freq_grid.clone(),
            harmonics.clone(),
            snr_scale.clone(),
            inspections.clone(),
        )?;
    }
    Ok(sim_ensemble)
}
