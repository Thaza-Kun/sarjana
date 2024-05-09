use pyo3::prelude::*;

mod args {
    use clap::Parser;
    use pyo3::prelude::*;

    #[derive(Parser, Clone)]
    #[allow(non_snake_case)]
    #[pyclass]
    pub struct Arguments {
        // Required because clap interprets `file.py` from `python file.py` as the first argument
        #[cfg(feature = "pyargs")]
        _pyfile: String,
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
        #[arg(long, default_value_t = 25.)]
        #[pyo3(get)]
        pub min_SNR: f32,
        #[arg(long, default_value_t = 0.)]
        #[pyo3(get)]
        pub min_power: f32,
        #[arg(short, long, default_value_t = 100)]
        #[pyo3(get)]
        pub runs: u128,
        #[arg(long, default_value = None)]
        #[pyo3(get)]
        pub period: Option<f64>,
        #[arg(long, default_value_t = 10.)]
        #[pyo3(get)]
        pub snr_scale: f64,
    }

    #[pyfunction]
    pub fn parse_arguments() -> PyResult<Arguments> {
        Ok(Arguments::parse())
    }
}

mod ensemble {
    use crate::args::Arguments;
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
        runs: u128,
        #[pyo3(get)]
        pub power: Vec<f64>,
        #[pyo3(get)]
        pub snr: Vec<f64>,
        #[pyo3(get)]
        pub freq: Vec<f64>,
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
        pub fn append(&mut self, power: &[f64], snr: &[f64], freq: &[f64]) {
            self.power.extend_from_slice(power);
            self.snr.extend_from_slice(snr);
            self.freq.extend_from_slice(freq);
            self.runs += 1;
        }
    }

    mod snr {
        use kdam::par_tqdm;
        use ndarray::Array1;
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
                cmp::max_by(snr, snr_max, |x, y| x.partial_cmp(y).unwrap()),
                cmp::max_by(harmonics as f64, harmonics_m, |x, y| {
                    x.partial_cmp(y).unwrap()
                }),
            )
        }

        pub fn greedy_harmonic_sum(
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
                            snr_max =
                                get_snr(let_sum, arrpower.mean().unwrap_or(0.), arrpower.std(1.));
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
                                .max_by(|x, y| x.partial_cmp(y).expect("Comparison error"))
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

    #[pyfunction]
    pub fn iterate_periodogram(
        py: Python<'_>,
        sim_signal: Bound<'_, PyArray1<f64>>,
        // astropy.Time
        sim_time: PyObject,
        frb_signal: Bound<'_, PyArray1<f64>>,
        // astropy.Time
        frb_time: PyObject,
        view_index: u16,
        view_length: u16,
        detection_rate: f64,
        arguments: Arguments,
        seed: usize,
        periodogram: Bound<'_, PyFunction>,
        find_peak: Bound<'_, PyFunction>,
        mut sim_ensemble: Ensemble,
        mut frb_ensemble: Ensemble,
        py_freq_grid: PyObject,
    ) -> PyResult<(Ensemble, Ensemble)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let sim_signal = sim_signal.to_vec().unwrap();
        let freq_grid =
            Array1::from_vec(py_freq_grid.getattr(py, "value")?.extract::<Vec<f64>>(py)?);
        let prepend = std::iter::repeat(false)
            .take(view_index.into())
            .collect::<Vec<bool>>();
        let append = std::iter::repeat(false)
            .take(sim_signal.len() - (view_index + view_length) as usize)
            .collect::<Vec<bool>>();
        for _ in tqdm!(0..arguments.runs, position = 0) {
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
                let power = timeit! {"Sim-power" <- Array1::from_vec(periodogram
                .call(
                    (sim_time.clone(), sim_signal.clone(), py_freq_grid.clone()),
                    None,
                )?
                .extract::<Vec<f64>>()?)};
                let snr = timeit! {
                    "Sim-SignalNoise" <- snr::greedy_harmonic_sum(power.clone(), freq_grid.clone(), arguments.harmonics as usize, arguments.snr_scale)
                };
                let (peaks, _props) = timeit! {"Sim-Peak" <- find_peak
                .call((power.to_vec().clone(),), None)?
                .extract::<(Vec<usize>, HashMap<String, Vec<f64>>)>()?};

                timeit! {"Sim-Ensemble" <- sim_ensemble.append(
                        &power.select(Axis(0), peaks.as_slice()).as_slice().unwrap(),
                        &snr.select(Axis(0), peaks.as_slice()).as_slice().unwrap(),
                        &freq_grid.select(Axis(0), peaks.as_slice()).as_slice().unwrap()
                    )
                }
            }

            {
                let power = timeit! {"FRB-Power" <- Array1::from_vec(periodogram
                    .call(
                    (frb_time.clone(), frb_signal.clone(), py_freq_grid.clone()),
                    None,
                )?
                .extract::<Vec<f64>>()?)};
                let snr = timeit! {
                    "FBR-SignalNoise" <- snr::greedy_harmonic_sum(power.clone(), freq_grid.clone(), arguments.harmonics as usize, arguments.snr_scale)
                };
                let (peaks, _props) = timeit! {"FRB-Peak"<-find_peak
                .call((power.to_vec(),), None)?
                .extract::<(Vec<usize>, HashMap<String, Vec<f64>>)>()?};

                timeit! {"FRB-Ensemble" <- frb_ensemble.append(
                        &power.select(Axis(0), peaks.as_slice()).as_slice().unwrap(),
                        &snr.select(Axis(0), peaks.as_slice()).as_slice().unwrap(),
                        &freq_grid.select(Axis(0), peaks.as_slice()).as_slice().unwrap()
                    )
                }
            }
        }
        println!("\n");
        Ok((sim_ensemble, frb_ensemble))
    }
}

#[pymodule]
#[pyo3(name = "main")]
fn _main(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<args::Arguments>()?;
    m.add_function(wrap_pyfunction!(args::parse_arguments, m)?)?;
    m.add_class::<ensemble::Ensemble>()?;
    m.add_function(wrap_pyfunction!(ensemble::iterate_periodogram, m)?)?;
    Ok(())
}
