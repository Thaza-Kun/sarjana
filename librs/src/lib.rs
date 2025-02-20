use kdam::par_tqdm;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
    m.add_function(wrap_pyfunction!(load_many_numpyz, m)?)?;
    Ok(())
}
