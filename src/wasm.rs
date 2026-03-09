//! WASM bindings for u-routing.
//!
//! Exposes a VRP solver to JavaScript via `wasm-bindgen`. Only compiled when
//! the `wasm` feature is enabled.
//!
//! # Usage (JavaScript)
//! ```js
//! import init, { solve_vrp } from '@iyulab/u-routing';
//! await init();
//!
//! // Nearest neighbor (default)
//! const result = solve_vrp({
//!   customers: [
//!     { id: 1, x: 1.0, y: 2.0, demand: 10.0 },
//!     { id: 2, x: 3.0, y: 4.0, demand: 15.0 },
//!   ],
//!   vehicles: [{ capacity: 100.0 }],
//!   depot: { x: 0.0, y: 0.0 },
//!   method: "nn",   // "nn" | "savings" | "ga" | "alns"
//! });
//! console.log(result.routes, result.total_distance, result.num_vehicles);
//!
//! // GA with custom config
//! const gaResult = solve_vrp({
//!   customers: [...],
//!   vehicles: [{ capacity: 100.0 }],
//!   depot: { x: 0.0, y: 0.0 },
//!   method: "ga",
//!   config: { population_size: 100, max_generations: 500 },
//! });
//!
//! // ALNS with time windows
//! const alnsResult = solve_vrp({
//!   customers: [
//!     { id: 1, x: 1.0, y: 2.0, demand: 10.0, time_window: [8.0, 12.0] },
//!     { id: 2, x: 3.0, y: 4.0, demand: 15.0, time_window: [10.0, 16.0] },
//!   ],
//!   vehicles: [{ capacity: 100.0 }],
//!   depot: { x: 0.0, y: 0.0 },
//!   method: "alns",
//!   config: { max_iterations: 1000 },
//! });
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;


use crate::alns::destroy::RandomRemoval;
use crate::alns::repair::GreedyInsertion;
use crate::alns::RoutingAlnsProblem;
use crate::constructive::{clarke_wright_savings, nearest_neighbor};
use crate::distance::DistanceMatrix;
use crate::ga::split;
use crate::ga::RoutingGaProblem;
use crate::local_search::{or_opt_improve, two_opt_improve};
use crate::models::{Customer, TimeWindow, Vehicle};
use u_metaheur::alns::{AlnsConfig, AlnsRunner};
use u_metaheur::ga::{GaConfig, GaRunner};

// ============================================================================
// Error helper
// ============================================================================

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ============================================================================
// Input / output serde types
// ============================================================================

#[derive(Deserialize)]
struct InputCustomer {
    id: usize,
    x: f64,
    y: f64,
    #[serde(default)]
    demand: f64,
    #[serde(default)]
    service_time: f64,
    /// Optional time window as `[ready, due]`.
    #[serde(default)]
    time_window: Option<[f64; 2]>,
}

#[derive(Deserialize)]
struct InputVehicle {
    #[serde(default = "default_capacity")]
    capacity: f64,
}

fn default_capacity() -> f64 {
    1e9
}

#[derive(Deserialize)]
struct InputDepot {
    x: f64,
    y: f64,
}

/// Optional solver configuration.
///
/// Fields are shared across methods; each method uses only the relevant ones.
#[derive(Deserialize, Default)]
struct InputConfig {
    // --- GA parameters ---
    /// Population size for GA (default: 50).
    #[serde(default)]
    population_size: Option<usize>,
    /// Maximum generations for GA (default: 200).
    #[serde(default)]
    max_generations: Option<usize>,
    /// Mutation rate for GA in (0, 1] (default: 0.1).
    #[serde(default)]
    mutation_rate: Option<f64>,
    /// Elite ratio for GA in (0, 1] (default: 0.1).
    #[serde(default)]
    elite_ratio: Option<f64>,

    // --- ALNS parameters ---
    /// Maximum iterations for ALNS (default: 500).
    #[serde(default)]
    max_iterations: Option<usize>,

    // --- Shared ---
    /// Random seed for reproducibility.
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct VrpInput {
    customers: Vec<InputCustomer>,
    #[serde(default)]
    vehicles: Vec<InputVehicle>,
    depot: InputDepot,
    #[serde(default = "default_method")]
    method: String,
    #[serde(default)]
    config: Option<InputConfig>,
}

fn default_method() -> String {
    "nn".to_string()
}

#[derive(Debug, Serialize)]
struct VrpOutput {
    routes: Vec<Vec<usize>>,
    total_distance: f64,
    num_vehicles: usize,
    method_used: String,
    computation_time_ms: f64,
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Builds the internal customer list and ID mapping from input.
///
/// Returns `(customers, id_map)` where `customers[0]` is the depot and
/// `id_map[i]` is the original customer ID for internal index `i+1`.
fn build_customers(
    depot: &InputDepot,
    input_customers: &[InputCustomer],
) -> (Vec<Customer>, Vec<usize>) {
    let mut customers: Vec<Customer> = Vec::with_capacity(input_customers.len() + 1);
    customers.push(Customer::depot(depot.x, depot.y));

    let mut id_map: Vec<usize> = Vec::with_capacity(input_customers.len());

    for ic in input_customers {
        let demand = ic.demand.round() as i32;
        let idx = customers.len();
        id_map.push(ic.id);
        let mut c = Customer::new(idx, ic.x, ic.y, demand, ic.service_time);
        if let Some([ready, due]) = ic.time_window {
            if let Some(tw) = TimeWindow::new(ready, due) {
                c = c.with_time_window(tw);
            }
        }
        customers.push(c);
    }

    (customers, id_map)
}

/// Converts internal route indices back to original customer IDs.
fn map_routes(routes: &[Vec<usize>], id_map: &[usize]) -> Vec<Vec<usize>> {
    routes
        .iter()
        .map(|route| {
            route
                .iter()
                .map(|&internal_idx| id_map[internal_idx - 1])
                .collect()
        })
        .collect()
}

/// Applies intra-route 2-opt + or-opt local search to improve routes.
fn apply_local_search(routes: &[Vec<usize>], dm: &DistanceMatrix) -> (Vec<Vec<usize>>, f64) {
    let mut improved_routes = Vec::with_capacity(routes.len());
    let mut total = 0.0;
    for route in routes {
        let (r1, _) = two_opt_improve(route, 0, dm);
        let (r2, dist) = or_opt_improve(&r1, 0, dm);
        total += dist;
        improved_routes.push(r2);
    }
    (improved_routes, total)
}

/// Builds the vehicle list from input, falling back to a single unlimited vehicle.
fn build_vehicles(input_vehicles: &[InputVehicle]) -> Vec<Vehicle> {
    if input_vehicles.is_empty() {
        vec![Vehicle::new(0, i32::MAX)]
    } else {
        input_vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| Vehicle::new(i, v.capacity.round().min(i32::MAX as f64) as i32))
            .collect()
    }
}

/// Returns the vehicle capacity as i32, using the first vehicle or i32::MAX.
fn vehicle_capacity(input_vehicles: &[InputVehicle]) -> i32 {
    if input_vehicles.is_empty() {
        i32::MAX
    } else {
        input_vehicles[0].capacity.round().min(i32::MAX as f64) as i32
    }
}

// ============================================================================
// Solver methods
// ============================================================================

fn solve_nn(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicles: &[Vehicle],
    id_map: &[usize],
) -> VrpOutput {
    let solution = nearest_neighbor(customers, dm, vehicles);
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    let mapped = map_routes(&routes, id_map);
    VrpOutput {
        total_distance: solution.total_distance(),
        num_vehicles: mapped.len(),
        routes: mapped,
        method_used: "nn".to_string(),
        computation_time_ms: 0.0,
    }
}

fn solve_savings(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicles: &[Vehicle],
    id_map: &[usize],
) -> VrpOutput {
    let vehicle_template = &vehicles[0];
    let solution = clarke_wright_savings(customers, dm, vehicle_template);
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    let mapped = map_routes(&routes, id_map);
    VrpOutput {
        total_distance: solution.total_distance(),
        num_vehicles: mapped.len(),
        routes: mapped,
        method_used: "savings".to_string(),
        computation_time_ms: 0.0,
    }
}

fn solve_ga(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    id_map: &[usize],
    cfg: &InputConfig,
) -> Result<VrpOutput, String> {
    let problem = RoutingGaProblem::new(customers.to_vec(), dm.clone(), capacity);

    let mut ga_config = GaConfig::default()
        .with_population_size(cfg.population_size.unwrap_or(50))
        .with_max_generations(cfg.max_generations.unwrap_or(200))
        .with_parallel(false); // rayon unavailable in WASM

    if let Some(mr) = cfg.mutation_rate {
        ga_config = ga_config.with_mutation_rate(mr);
    }
    if let Some(er) = cfg.elite_ratio {
        ga_config = ga_config.with_elite_ratio(er);
    }
    if let Some(seed) = cfg.seed {
        ga_config = ga_config.with_seed(seed);
    }

    ga_config
        .validate()
        .map_err(|e| format!("GA config error: {}", e))?;

    let ga_result = GaRunner::run(&problem, &ga_config)
        .map_err(|e| format!("GA execution error: {}", e))?;

    // Split the best individual to get routes
    let split_result = split(ga_result.best.customers(), customers, dm, capacity);

    // Apply local search to improve routes
    let (improved_routes, total_distance) = apply_local_search(&split_result.routes, dm);
    let mapped = map_routes(&improved_routes, id_map);

    Ok(VrpOutput {
        num_vehicles: mapped.len(),
        total_distance,
        routes: mapped,
        method_used: "ga".to_string(),
        computation_time_ms: 0.0,
    })
}

fn solve_alns(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    id_map: &[usize],
    cfg: &InputConfig,
) -> Result<VrpOutput, String> {
    let problem = RoutingAlnsProblem::new(customers.to_vec(), dm.clone(), capacity);

    let destroy_ops = vec![RandomRemoval];
    let repair_ops = vec![GreedyInsertion::new(
        dm.clone(),
        customers.to_vec(),
        capacity,
    )];

    let mut alns_config =
        AlnsConfig::default().with_max_iterations(cfg.max_iterations.unwrap_or(500));

    if let Some(seed) = cfg.seed {
        alns_config = alns_config.with_seed(seed);
    }

    alns_config
        .validate()
        .map_err(|e| format!("ALNS config error: {}", e))?;

    let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &alns_config)
        .map_err(|e| format!("ALNS execution error: {}", e))?;

    // Apply local search to improve ALNS result
    let alns_routes: Vec<Vec<usize>> = result.best.routes().to_vec();
    let (improved_routes, total_distance) = apply_local_search(&alns_routes, dm);
    let mapped = map_routes(&improved_routes, id_map);

    Ok(VrpOutput {
        num_vehicles: mapped.len(),
        total_distance,
        routes: mapped,
        method_used: "alns".to_string(),
        computation_time_ms: 0.0,
    })
}

// ============================================================================
// Public WASM entry point
// ============================================================================

/// Solve a capacitated VRP problem given as JSON.
///
/// # Arguments
/// * `problem_json` — A JS object matching the VRP input schema.
///
/// # Supported methods
/// - `"nn"` — Nearest Neighbor (default, fast)
/// - `"savings"` — Clarke-Wright Savings
/// - `"ga"` — Genetic Algorithm with Prins split + local search
/// - `"alns"` — Adaptive Large Neighborhood Search + local search
///
/// # Returns
/// A JS object with `routes`, `total_distance`, `num_vehicles`,
/// `method_used`, and `computation_time_ms`.
///
/// # Errors
/// Returns a `JsValue` string describing the error if input is invalid.
#[wasm_bindgen]
pub fn solve_vrp(problem_json: JsValue) -> Result<JsValue, JsValue> {
    let input: VrpInput = serde_wasm_bindgen::from_value(problem_json).map_err(js_err)?;

    let (customers, id_map) = build_customers(&input.depot, &input.customers);

    if customers.len() <= 1 {
        let output = VrpOutput {
            routes: vec![],
            total_distance: 0.0,
            num_vehicles: 0,
            method_used: input.method.clone(),
            computation_time_ms: 0.0,
        };
        return serde_wasm_bindgen::to_value(&output).map_err(js_err);
    }

    let dm = DistanceMatrix::from_customers(&customers);
    let vehicles = build_vehicles(&input.vehicles);
    let capacity = vehicle_capacity(&input.vehicles);
    let config = input.config.unwrap_or_default();

    let start = web_time();

    let mut output = match input.method.as_str() {
        "nn" => solve_nn(&customers, &dm, &vehicles, &id_map),
        "savings" => solve_savings(&customers, &dm, &vehicles, &id_map),
        "ga" => solve_ga(&customers, &dm, capacity, &id_map, &config).map_err(js_err)?,
        "alns" => solve_alns(&customers, &dm, capacity, &id_map, &config).map_err(js_err)?,
        other => {
            return Err(js_err(format!(
                "unknown method '{}'. Supported: \"nn\", \"savings\", \"ga\", \"alns\"",
                other
            )))
        }
    };

    output.computation_time_ms = elapsed_ms(start);

    serde_wasm_bindgen::to_value(&output).map_err(js_err)
}

// ============================================================================
// Timing utilities (WASM-compatible)
// ============================================================================

/// Returns a timestamp in milliseconds (uses `performance.now()` in WASM,
/// falls back to `Instant` on native).
#[cfg(target_arch = "wasm32")]
fn web_time() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn web_time() -> f64 {
    // For native testing — not actually used in WASM builds
    0.0
}

#[cfg(target_arch = "wasm32")]
fn elapsed_ms(start: f64) -> f64 {
    js_sys::Date::now() - start
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_ms(_start: f64) -> f64 {
    0.0
}

// ============================================================================
// Tests (native — exercise internal solver functions without JsValue)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMatrix;
    use crate::models::Customer;

    /// Helper: build a small test problem with N customers around the origin.
    fn test_customers(n: usize) -> (Vec<Customer>, DistanceMatrix, Vec<usize>) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        let mut id_map = Vec::new();
        for i in 1..=n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            customers.push(Customer::new(
                i,
                angle.cos() * 10.0,
                angle.sin() * 10.0,
                5,
                0.0,
            ));
            id_map.push(i);
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm, id_map)
    }

    // ---- GA: valid minimal input ----

    #[test]
    fn ga_valid_minimal() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method_used, "ga");
        assert!(!output.routes.is_empty());
        assert!(output.total_distance > 0.0);
    }

    // ---- GA: population_size too small ----

    #[test]
    fn ga_population_size_too_small() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(1),
            max_generations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "population_size=1 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.contains("population_size"),
            "error should mention population_size: {}",
            err
        );
    }

    #[test]
    fn ga_population_size_zero() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(0),
            max_generations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "population_size=0 should fail validation");
    }

    // ---- GA: max_generations zero ----

    #[test]
    fn ga_zero_generations() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "max_generations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.contains("max_generations"),
            "error should mention max_generations: {}",
            err
        );
    }

    // ---- GA: elite_ratio too high ----

    #[test]
    fn ga_elite_ratio_fills_population() {
        let (customers, dm, id_map) = test_customers(3);
        // elite_ratio is clamped to 1.0, so with pop=2 all are elite → validation error
        let cfg = InputConfig {
            population_size: Some(2),
            max_generations: Some(5),
            elite_ratio: Some(1.5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(
            result.is_err(),
            "elite_ratio filling entire population should fail"
        );
    }

    // ---- GA: single customer ----

    #[test]
    fn ga_single_customer() {
        let (customers, dm, id_map) = test_customers(1);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with 1 customer should succeed");
        let output = result.unwrap();
        assert_eq!(output.routes.len(), 1);
    }

    // ---- GA: mutation_rate clamped (not an error, just verifies no panic) ----

    #[test]
    fn ga_extreme_mutation_rate() {
        let (customers, dm, id_map) = test_customers(3);
        // mutation_rate > 1.0 is clamped by GaConfig::with_mutation_rate
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            mutation_rate: Some(5.0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(
            result.is_ok(),
            "clamped mutation_rate should not cause error"
        );
    }

    // ---- ALNS: valid minimal input ----

    #[test]
    fn alns_valid_minimal() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            max_iterations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_alns(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "ALNS with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method_used, "alns");
        assert!(!output.routes.is_empty());
    }

    // ---- ALNS: zero iterations ----

    #[test]
    fn alns_zero_iterations() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            max_iterations: Some(0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_alns(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "max_iterations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.contains("max_iterations"),
            "error should mention max_iterations: {}",
            err
        );
    }

    // ---- Default config (no config provided) ----

    #[test]
    fn ga_default_config() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with default config should succeed");
    }

    #[test]
    fn alns_default_config() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_alns(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "ALNS with default config should succeed");
    }

    // ---- GA: larger problem to stress-test GA operators ----

    #[test]
    fn ga_larger_problem() {
        let (customers, dm, id_map) = test_customers(20);
        let cfg = InputConfig {
            population_size: Some(30),
            max_generations: Some(20),
            seed: Some(123),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 1000, &id_map, &cfg);
        assert!(
            result.is_ok(),
            "GA with 20 customers should succeed: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert!(!output.routes.is_empty());
        assert!(output.total_distance > 0.0);
    }

    // ---- GA: tight capacity forces many routes ----

    #[test]
    fn ga_tight_capacity() {
        let (customers, dm, id_map) = test_customers(10);
        // Each customer has demand=5, capacity=5 forces one customer per route
        let cfg = InputConfig {
            population_size: Some(20),
            max_generations: Some(10),
            seed: Some(99),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 5, &id_map, &cfg);
        assert!(
            result.is_ok(),
            "GA with tight capacity should succeed: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert_eq!(output.routes.len(), 10, "each customer needs its own route");
    }

    // ---- GA: two customers (minimal crossover) ----

    #[test]
    fn ga_two_customers() {
        let (customers, dm, id_map) = test_customers(2);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with 2 customers should succeed");
    }
}
