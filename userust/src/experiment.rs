use std::path;

use crate::diagnostic::PdgramDiagnostic;

struct ExperimentConfig {
    output: path::PathBuf,
    seed: usize,
    outdir: path::PathBuf,
    runs: usize,
    periodogram: PdgramDiagnostic<'static>,
}
