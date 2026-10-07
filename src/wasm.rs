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

/// A NaN or ±Infinity found in a JS argument, and where it sits.
///
/// JSON has no non-finite numbers, so on the way to the wire schema
/// `serde_json` turns one into `null` and the caller would be told a value has
/// the wrong type. [`find_non_finite`] looks before that happens, so the
/// refusal names the real reason and the place.
struct NonFinite {
    /// The argument's name, then `.key` and `[i]` steps down to the array or
    /// field that holds the number.
    parameter: String,
    /// The number's position, when it is an array element.
    index: Option<usize>,
    value: f64,
}

impl NonFinite {
    fn message(&self) -> String {
        let at = match self.index {
            Some(i) => format!("{}[{i}]", self.parameter),
            None => self.parameter.clone(),
        };
        let got = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        format!("{at}: expected a finite number, got {got}")
    }

    /// `parameter` and `index` (`null` when the number is not an array element).
    fn fields(&self) -> serde_json::Value {
        serde_json::json!({ "parameter": self.parameter, "index": self.index })
    }
}

/// The first NaN or ±Infinity in `value`, searching arrays, iterables and
/// plain objects. `allow_nan` lets NaN through for an input that reads it as a
/// missing value; it then arrives as `null`.
fn find_non_finite(value: &JsValue, parameter: &str, allow_nan: bool) -> Option<NonFinite> {
    let refused = |n: f64| !n.is_finite() && !(allow_nan && n.is_nan());
    let found = |index: Option<usize>, value: f64| NonFinite {
        parameter: parameter.to_string(),
        index,
        value,
    };
    if let Some(n) = value.as_f64() {
        return refused(n).then(|| found(None, n));
    }
    if !value.is_object() {
        return None;
    }
    if let Ok(Some(items)) = js_sys::try_iter(value) {
        for (i, item) in items.enumerate() {
            // An iterator that throws is left for serde to report.
            let item = item.ok()?;
            match item.as_f64() {
                Some(n) if refused(n) => return Some(found(Some(i), n)),
                Some(_) => {}
                None => {
                    let inner = find_non_finite(&item, &format!("{parameter}[{i}]"), allow_nan);
                    if inner.is_some() {
                        return inner;
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"), allow_nan);
        if inner.is_some() {
            return inner;
        }
    }
    None
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
    if let Some(found) = find_non_finite(&value, param, false) {
        return Err(js_err(ServiceError::value_not_finite(
            found.message(),
            found.fields(),
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    from_json(json, param).map_err(js_err)
}

/// The half of [`from_js`] that enforces the wire schema, apart from the
/// `JsValue` (which cannot be built off `wasm32`) so tests walk the same path a
/// JS caller does. A refusal names where it stopped: see [`failure_site`].
fn from_json<T: serde::de::DeserializeOwned>(
    json: serde_json::Value,
    param: &str,
) -> Result<T, ServiceError> {
    serde_path_to_error::deserialize(json).map_err(|e| {
        let (parameter, index) = failure_site(param, e.path());
        let mut err = ServiceError::malformed_input(
            &parameter,
            format!("{param}: {}: {}", e.path(), e.inner()),
        );
        if let Some(i) = index {
            err.fields["index"] = serde_json::json!(i);
        }
        err
    })
}

/// Where in the argument `param` a request stopped deserializing, as the
/// refusal reports it: the field (`design[1]`, `points[0].y`) and, when the
/// failure sits in an array, its position there -- the same `parameter` /
/// `index` a number array read element by element reports. Without it a
/// `null` three levels down was "invalid type: null, expected f64" with no
/// way to say which row (the gap `read_numbers` closed for bare arrays).
fn failure_site(param: &str, path: &serde_path_to_error::Path) -> (String, Option<usize>) {
    use serde_path_to_error::Segment;
    let segments: Vec<&Segment> = path.iter().collect();
    let index = segments.iter().rev().find_map(|s| match s {
        Segment::Seq { index } => Some(*index),
        _ => None,
    });
    // A trailing `[i]` is the index, not part of the name.
    let named = match segments.last() {
        Some(Segment::Seq { .. }) => &segments[..segments.len() - 1],
        _ => &segments[..],
    };
    let mut name = String::new();
    for segment in named {
        match segment {
            Segment::Seq { index } => name.push_str(&format!("[{index}]")),
            Segment::Map { key } | Segment::Enum { variant: key } => {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(key);
            }
            Segment::Unknown => name.push_str(".?"),
        }
    }
    // A top-level array argument (`data[1][2]`) or a failure at the root
    // (a missing field) is named by the argument itself.
    if name.is_empty() || name.starts_with('[') {
        name.insert_str(0, param);
    }
    (name, index)
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

#[cfg(test)]
mod path_tests {
    //! A value of the wrong type deep inside an argument is refused where it
    //! sits: `parameter` names the field, `index` its array position.

    #[test]
    fn a_bad_coordinate_is_refused_at_its_customer() {
        let err = super::from_json::<Vec<crate::service::InputCustomer>>(
            serde_json::json!([
                { "id": 1, "x": 1.0, "y": 2.0 },
                { "id": 2, "x": "far", "y": 0.0 }
            ]),
            "customers",
        )
        .err()
        .expect("a string is not a number");
        assert_eq!(err.fields["code"], "malformed_input");
        assert_eq!(err.fields["parameter"], "customers[1].x");
        assert_eq!(err.fields["index"], 1);
    }
}
