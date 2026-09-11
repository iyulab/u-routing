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
//! // Time windows ("nn" and "ga" keep them; "savings" and "alns" refuse them)
//! const twResult = solve_vrp({
//!   customers: [
//!     { id: 1, x: 1.0, y: 2.0, demand: 10.0, time_window: [8.0, 12.0] },
//!     { id: 2, x: 3.0, y: 4.0, demand: 15.0, time_window: [10.0, 16.0] },
//!   ],
//!   vehicles: [{ capacity: 100.0 }],
//!   depot: { x: 0.0, y: 0.0 },
//!   method: "ga",
//!   config: { population_size: 100, max_generations: 500 },
//! });
//! ```

use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::service::{self, InputConfig, InputCustomer, InputVehicle};

// ============================================================================
// Error helper
// ============================================================================

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    if value.as_string().is_some() {
        return Err(JsValue::from_str(&format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value = serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("{param}: {e}")))?;
    serde_json::from_value(json).map_err(|e| JsValue::from_str(&format!("{param}: {e}")))
}

// ============================================================================
// Input shape
// ============================================================================
//
// Customers, vehicles and solver settings are the shared wire types in
// `service`. Only the top level is this binding's own: the depot is an object.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputDepot {
    x: f64,
    y: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VrpInput {
    customers: Vec<InputCustomer>,
    #[serde(default)]
    vehicles: Vec<InputVehicle>,
    depot: InputDepot,
    #[serde(default = "service::default_method")]
    method: String,
    #[serde(default)]
    config: Option<InputConfig>,
}

// ============================================================================
// Public WASM entry point
// ============================================================================

/// Solve a capacitated VRP problem.
///
/// # Arguments
/// * `problem` — A native JS object matching the VRP input schema.
///
/// # Supported methods
/// - `"nn"` — Nearest Neighbor (default, fast)
/// - `"savings"` — Clarke-Wright Savings
/// - `"ga"` — Genetic Algorithm with Prins split + local search (the local
///   search is skipped when customers carry time windows)
/// - `"alns"` — Adaptive Large Neighborhood Search + local search
///
/// Only `"nn"` reads each vehicle; the others plan with one capacity. Only
/// `"nn"` and `"ga"` keep time windows.
///
/// # Returns
/// A JS object with `routes`, `total_distance`, `num_vehicles`,
/// `method_used`, and `computation_time_ms`.
///
/// # Errors
/// Returns a `JsValue` string describing the error if input is invalid: an
/// unknown method, a time window with `ready > due`, a demand or capacity that
/// is not a whole number of units, a mixed fleet or time windows the method
/// cannot model, or solver settings the method rejects.
#[wasm_bindgen]
pub fn solve_vrp(problem: JsValue) -> Result<JsValue, JsValue> {
    let input: VrpInput = from_js(problem, "problem")?;
    let config = input.config.unwrap_or_default();

    let start = web_time();
    let mut output = service::solve(
        (input.depot.x, input.depot.y),
        &input.customers,
        &input.vehicles,
        &input.method,
        &config,
    )
    .map_err(js_err)?;
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

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn vrp_input_rejects_unknown_keys() {
        // Real consumer defect class: GA options sent under `ga_config`
        // (schema key is `config`) were silently dropped before this guard.
        assert_rejects_unknown::<super::VrpInput>(json!({
            "customers": [
                { "id": 1, "x": 0.0, "y": 0.0 },
                { "id": 2, "x": 1.0, "y": 1.0 }
            ],
            "depot": { "x": 0.0, "y": 0.0 },
            "method": "ga",
            "ga_config": { "population_size": 60 }
        }));
    }

    #[test]
    fn depot_rejects_unknown_keys() {
        assert_rejects_unknown::<super::InputDepot>(json!({
            "x": 0.0, "y": 0.0, "id": 0
        }));
    }
}
