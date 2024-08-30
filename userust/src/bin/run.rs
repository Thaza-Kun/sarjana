use userust::python;

fn main() {
    python::with_virtualenv(".venv");
    python::just_panic(|py| {
        py.import_bound("numpy")?;
        Ok(())
    });
}
