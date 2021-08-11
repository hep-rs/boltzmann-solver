use crate::{
    model::{Empty, Model, Particle},
    prelude::ModelInteractions,
    solver::SolverBuilder,
};
use ndarray::Array1;
use std::{env::temp_dir, error, fs, path::Path, sync::RwLock};

mod decay;
mod scattering;
mod standard_de;
mod unit;

const BETA_START: f64 = 1e-10;
const BETA_END: f64 = 1e1;

///
fn solve<P, M>(
    name: P,
    builder: SolverBuilder<M>,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn error::Error>>
where
    P: AsRef<Path>,
    M: ModelInteractions,
{
    let n = builder
        .model
        .as_ref()
        .map(|m| m.len_particles())
        .unwrap_or_default();

    // Run the solver
    let dir = temp_dir().join("boltzmann-solver").join("empty-model");
    fs::create_dir_all(&dir)?;

    let mut csv = csv::Writer::from_path(dir.join(name))?;
    csv.write_field("step")?;
    csv.write_field("beta")?;
    for i in 1..n {
        csv.write_field(format!("n-{}", i))?;
        csv.write_field(format!("dn-{}", i))?;
        csv.write_field(format!("na-{}", i))?;
        csv.write_field(format!("dna-{}", i))?;
        csv.write_field(format!("eq-{}", i))?;
    }
    csv.write_record(None::<&[u8]>)?;
    let csv = RwLock::new(csv);

    Ok(builder
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };

            csv.write_field(format!("{:.e}", c.step)).unwrap();
            csv.write_field(format!("{:.e}", c.beta)).unwrap();
            for i in 1..n {
                csv.write_field(format!("{:.e}", c.n[i])).unwrap();
                csv.write_field(format!("{:.e}", dn[i])).unwrap();
                csv.write_field(format!("{:.e}", c.na[i])).unwrap();
                csv.write_field(format!("{:.e}", dna[i])).unwrap();
                csv.write_field(format!("{:.e}", c.eq[i])).unwrap();
            }
            csv.write_record(None::<&[u8]>).unwrap();
            csv.flush().unwrap();
        })
        .build()?
        .solve()?)
}

#[test]
fn no_interaction() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(2);

    let mut model = Empty::default();
    model.push_particle(Particle::new(0, 1e5, 1e1).name("1"));
    model.push_particle(Particle::new(0, 1e3, 1e1).name("2"));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, 1e1)
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check that number densities have not changed.
    assert_eq!(n0, n);
    assert_eq!(na0, na);

    Ok(())
}
