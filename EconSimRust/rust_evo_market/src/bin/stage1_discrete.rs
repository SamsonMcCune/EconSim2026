use rust_evo_market::{population_evolution_market, save_discrete_outputs, SimulationConfig};
use std::time::Instant;

fn main() {
    let config = SimulationConfig {
        num_agents: 1000,
        total_rounds: 20,
        grid_size: 50,
        gaussian_sigma: 5.0,
        delta: 0.1,
        seed: 67,
    };

    let start = Instant::now();
    let sim = population_evolution_market(&config);
    let elapsed = start.elapsed();
    println!("Time: {:.3} seconds", elapsed.as_secs_f64());

    let out_dir = "output/discrete";
    save_discrete_outputs(out_dir, &sim).expect("failed to write CSV outputs");

    println!("Wrote discrete outputs to {}", out_dir);
    println!("Files include history.csv, market.csv, orders.csv, snapshots.csv, resource_grid.csv, and empirical curve CSVs.");
    println!("Note: your original Python plotting calls use round index 20 after running 20 rounds, which is out of range in zero-based indexing; this Rust version writes the full data so you can inspect valid rounds directly.");
}
