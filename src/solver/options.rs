/// Determines the range of step sizes allowed: `min * beta < h < max * beta`.
pub(crate) struct StepPrecision {
    pub min: f64,
    pub max: f64,
}

impl Default for StepPrecision {
    fn default() -> Self {
        StepPrecision {
            min: 1e-10,
            max: 1e-0,
        }
    }
}
