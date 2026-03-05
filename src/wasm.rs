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
//! const result = solve_vrp({
//!   customers: [
//!     { id: 1, x: 1.0, y: 2.0, demand: 10.0 },
//!     { id: 2, x: 3.0, y: 4.0, demand: 15.0 },
//!   ],
//!   vehicles: [{ capacity: 100.0 }],
//!   depot: { x: 0.0, y: 0.0 },
//!   method: "nn",   // "nn" | "savings"
//! });
//! console.log(result.routes, result.total_distance, result.num_vehicles);
//! ```

#![cfg(feature = "wasm")]

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::constructive::{clarke_wright_savings, nearest_neighbor};
use crate::distance::DistanceMatrix;
use crate::models::{Customer, Vehicle};

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

#[derive(Deserialize)]
struct VrpInput {
    customers: Vec<InputCustomer>,
    #[serde(default)]
    vehicles: Vec<InputVehicle>,
    depot: InputDepot,
    #[serde(default = "default_method")]
    method: String,
}

fn default_method() -> String {
    "nn".to_string()
}

#[derive(Serialize)]
struct VrpOutput {
    routes: Vec<Vec<usize>>,
    total_distance: f64,
    num_vehicles: usize,
}

// ============================================================================
// Public WASM entry point
// ============================================================================

/// Solve a capacitated VRP problem given as JSON.
///
/// # Arguments
/// * `problem_json` — A JS object matching the VRP input schema.
///
/// # Returns
/// A JS object with `routes`, `total_distance`, and `num_vehicles`.
///
/// # Errors
/// Returns a `JsValue` string describing the error if input is invalid.
#[wasm_bindgen]
pub fn solve_vrp(problem_json: JsValue) -> Result<JsValue, JsValue> {
    let input: VrpInput =
        serde_wasm_bindgen::from_value(problem_json).map_err(js_err)?;

    // Build customer list: index 0 = depot, then customers in order of input id
    // We preserve the customer IDs from input so the output routes use them.
    let mut customers: Vec<Customer> =
        Vec::with_capacity(input.customers.len() + 1);

    // Depot is always id=0
    customers.push(Customer::depot(input.depot.x, input.depot.y));

    // Map from internal index (1..n) to the original customer id
    let mut id_map: Vec<usize> = Vec::with_capacity(input.customers.len());

    for ic in &input.customers {
        let demand = ic.demand.round() as i32;
        // internal index = customers.len() before push
        id_map.push(ic.id);
        customers.push(Customer::new(customers.len(), ic.x, ic.y, demand, 0.0));
    }

    if customers.len() <= 1 {
        // No customers — return empty solution
        let output = VrpOutput {
            routes: vec![],
            total_distance: 0.0,
            num_vehicles: 0,
        };
        return serde_wasm_bindgen::to_value(&output).map_err(js_err);
    }

    let dm = DistanceMatrix::from_customers(&customers);

    // Build vehicle list; fall back to a single unlimited vehicle
    let vehicles: Vec<Vehicle> = if input.vehicles.is_empty() {
        vec![Vehicle::new(0, i32::MAX)]
    } else {
        input
            .vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| Vehicle::new(i, v.capacity.round().min(i32::MAX as f64) as i32))
            .collect()
    };

    // Solve
    let solution = match input.method.as_str() {
        "savings" => {
            // Clarke-Wright uses a single vehicle template for homogeneous fleet
            let vehicle_template = &vehicles[0];
            clarke_wright_savings(&customers, &dm, vehicle_template)
        }
        "nn" => nearest_neighbor(&customers, &dm, &vehicles),
        other => return Err(js_err(format!(
            "unknown method '{}'. Supported: \"nn\", \"savings\"", other
        ))),
    };

    // Convert internal indices back to original customer IDs
    let routes: Vec<Vec<usize>> = solution
        .routes()
        .iter()
        .map(|route| {
            route
                .customer_ids()
                .iter()
                .map(|&internal_idx| {
                    // internal_idx is 1-based in our customer list
                    id_map[internal_idx - 1]
                })
                .collect()
        })
        .collect();

    let total_distance = solution.total_distance();
    let num_vehicles = routes.len();

    let output = VrpOutput {
        routes,
        total_distance,
        num_vehicles,
    };

    serde_wasm_bindgen::to_value(&output).map_err(js_err)
}
