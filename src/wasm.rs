//! WASM bindings for u-routing.
//!
//! Exposes a VRP solver to JavaScript via `wasm-bindgen`. Only compiled when
//! the `wasm` feature is enabled.
//!
//! # Usage (JavaScript)
//! ```js
//! import { solve_vrp } from '@iyulab/u-routing';
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
//! // Time windows (every method keeps them)
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

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::service::{self, InputConfig, InputCustomer, InputVehicle, ServiceError};

// ============================================================================
// Error helper
// ============================================================================

/// Every refusal crosses into JavaScript as an `Error` whose `message` is the
/// readable text and which carries `code` -- a stable reason -- and the values
/// behind it as further properties (`id`, `index`, `parameter`, ...). A program
/// branches on `err.code` and reads the fields; `err.message` reads as it
/// always did.
fn js_err(error: ServiceError) -> JsValue {
    let js = js_sys::Error::new(&error.message);
    // `json_compatible` turns the map into a plain object; the default would
    // produce a JavaScript `Map`, which `Object.assign` does not read.
    if let Ok(fields) = error
        .fields
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    {
        js_sys::Object::assign(&js, &fields.into());
    }
    js.into()
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    let refuse = |message: String| js_err(ServiceError::malformed_input(param, message));
    if value.as_string().is_some() {
        return Err(refuse(format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    serde_json::from_value(json).map_err(|e| refuse(format!("{param}: {e}")))
}

// ============================================================================
// Input shape
// ============================================================================
//
// Customers, vehicles and solver settings are the shared wire types in
// `service`. Only the top level is this binding's own: the depot is an object.

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct InputDepot {
    x: f64,
    y: f64,
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct VrpInput {
    customers: Vec<InputCustomer>,
    #[serde(default)]
    #[tsify(optional)]
    vehicles: Vec<InputVehicle>,
    depot: InputDepot,
    #[serde(default = "service::default_method")]
    #[tsify(optional)]
    #[tsify(type = "\"nn\" | \"savings\" | \"ga\" | \"alns\"")]
    method: String,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "InputConfig | null")]
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
/// Only `"nn"` reads each vehicle; the others plan with one capacity. Every
/// method keeps time windows.
///
/// # Returns
/// A JS object with `routes`, `total_distance`, `num_vehicles`,
/// `method_used`, and `computation_time_ms`.
///
/// # Errors
/// Throws an `Error` carrying `code` and the values behind it: an unknown
/// method (`unknown_option`), a repeated customer id (`duplicate_id`), a time
/// window with `ready > due` (`invalid_time_window`), a demand or capacity
/// that is not a whole number of units (`not_whole_units`), a mixed fleet the
/// method cannot model (`mixed_fleet`), a plan over `max_vehicles`
/// (`routes_exceed_max_vehicles`), solver settings the method rejects
/// (`invalid_option`), or an input of the wrong shape (`malformed_input`).
#[wasm_bindgen(unchecked_return_type = "VrpOutput")]
pub fn solve_vrp(
    #[wasm_bindgen(unchecked_param_type = "VrpInput")] problem: JsValue,
) -> Result<JsValue, JsValue> {
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

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| js_err(ServiceError::malformed_input("result", e.to_string())))
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
