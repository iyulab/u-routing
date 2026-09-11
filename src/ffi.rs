//! FFI module for u-routing — JSON-in/JSON-out pattern
//!
//! Status codes:
//!   0 = OK
//!  -1 = null pointer input
//!  -2 = request is not valid JSON of the expected shape
//!  -3 = request rejected (unknown method, invalid time window, solver settings)
//!  -4 = internal panic
//!
//! Every non-zero status except `-1` comes with an `{"error": "..."}` body.
//! All entry points are wrapped in `catch_unwind` to prevent panic propagation.
//!
//! The solver itself is `crate::service`, shared with the WebAssembly binding;
//! this module owns only the transport and the request's top-level shape.

use std::ffi::{CStr, CString};
use std::panic;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::service::{self, InputConfig, InputCustomer, InputVehicle};

/// Status for a request whose JSON could not be read into the expected shape.
const ERR_PARSE: i32 = -2;

/// Status for a well-formed request the solver rejected.
const ERR_COMPUTE: i32 = -3;

// ── Request shape ───────────────────────────────────────────
//
// Customers, vehicles and solver settings are the shared wire types. Only the
// top level is this binding's own: the depot is two flat fields.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VrpInput {
    customers: Vec<InputCustomer>,
    #[serde(default)]
    vehicles: Vec<InputVehicle>,
    depot_x: f64,
    depot_y: f64,
    #[serde(default = "service::default_method")]
    method: String,
    #[serde(default)]
    config: Option<InputConfig>,
}

// ── Helpers ─────────────────────────────────────────────────

unsafe fn read_json(ptr: *const libc::c_char) -> Result<String, i32> {
    if ptr.is_null() {
        return Err(-1);
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().map(|s| s.to_string()).map_err(|_| ERR_PARSE)
}

fn write_json<T: Serialize>(result_ptr: *mut *mut libc::c_char, value: &T) -> i32 {
    if result_ptr.is_null() {
        return -1;
    }
    match serde_json::to_string(value) {
        Ok(json) => match CString::new(json) {
            Ok(cstr) => {
                unsafe { *result_ptr = cstr.into_raw() };
                0
            }
            Err(_) => ERR_COMPUTE,
        },
        Err(_) => ERR_COMPUTE,
    }
}

/// Writes `{"error": msg}` and returns `status`.
///
/// The status is the caller's to choose and is returned as given: the error
/// body is a diagnostic, not a result, so writing it successfully must not
/// turn the call into a success.
fn write_error(result_ptr: *mut *mut libc::c_char, status: i32, msg: &str) -> i32 {
    let err = serde_json::json!({ "error": msg });
    match write_json(result_ptr, &err) {
        0 => status,
        write_failure => write_failure,
    }
}

/// Wraps an FFI body in `catch_unwind`, initializing `result_ptr` to null.
fn ffi_catch(
    result_ptr: *mut *mut libc::c_char,
    f: impl FnOnce() -> i32 + panic::UnwindSafe,
) -> i32 {
    if !result_ptr.is_null() {
        unsafe { *result_ptr = std::ptr::null_mut() };
    }
    match panic::catch_unwind(f) {
        Ok(code) => code,
        Err(_) => write_error(result_ptr, -4, "internal panic"),
    }
}

// ── FFI exports ─────────────────────────────────────────────

/// Solve a VRP (TSP, CVRP or VRPTW, by what the request carries).
///
/// `method` is one of `"nn"` (default), `"savings"`, `"ga"` or `"alns"`, and
/// `config` carries the GA/ALNS settings -- the same methods and settings as
/// the WebAssembly binding, from the same code.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`urouting_free_string`].
#[no_mangle]
pub unsafe extern "C" fn urouting_solve_vrp(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let input: VrpInput = match serde_json::from_str(&json) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_PARSE, &format!("Invalid JSON: {e}")),
        };
        let config = input.config.unwrap_or_default();

        let start = Instant::now();
        match service::solve(
            (input.depot_x, input.depot_y),
            &input.customers,
            &input.vehicles,
            &input.method,
            &config,
        ) {
            Ok(mut output) => {
                output.computation_time_ms = start.elapsed().as_secs_f64() * 1e3;
                write_json(result_ptr, &output)
            }
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

/// Free a string allocated by u-routing FFI.
///
/// # Safety
///
/// `ptr` must be null or a string returned by this library that has not
/// already been freed.
#[no_mangle]
pub unsafe extern "C" fn urouting_free_string(ptr: *mut libc::c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}

/// Get u-routing version
#[no_mangle]
pub extern "C" fn urouting_version() -> *mut libc::c_char {
    let version = env!("CARGO_PKG_VERSION");
    CString::new(version)
        .expect("version string has no interior NUL")
        .into_raw()
}

// ── Tests ───────────────────────────────────────────────────
//
// These drive the exported symbol as a C caller does -- a C string in, a C
// string out, a status code -- so they pin the wire contract.

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(request: &serde_json::Value) -> (i32, serde_json::Value) {
        let request = CString::new(request.to_string()).expect("no interior NUL");
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        let code = unsafe { urouting_solve_vrp(request.as_ptr(), &mut out) };
        assert!(!out.is_null(), "every status must come with a JSON body");
        let body = unsafe { CStr::from_ptr(out) }
            .to_str()
            .expect("body is UTF-8")
            .to_owned();
        unsafe { urouting_free_string(out) };
        (code, serde_json::from_str(&body).expect("body is JSON"))
    }

    fn problem(method: &str) -> serde_json::Value {
        serde_json::json!({
            "customers": [
                { "id": 11, "x": 1.0, "y": 2.0, "demand": 10.0 },
                { "id": 12, "x": 3.0, "y": 1.0, "demand": 15.0 },
                { "id": 13, "x": -2.0, "y": 4.0, "demand": 5.0 },
                { "id": 14, "x": -1.0, "y": -3.0, "demand": 20.0 }
            ],
            "vehicles": [{ "capacity": 30.0 }],
            "depot_x": 0.0,
            "depot_y": 0.0,
            "method": method
        })
    }

    #[test]
    fn a_rejected_request_reports_a_failure_status() {
        // An error body under status 0 reaches the C# client as a successful
        // result: it only raises on a non-zero status.
        let (code, body) = solve(&problem("no-such-method"));
        assert_eq!(code, -3, "{body}");
        assert!(body["error"].is_string());
    }

    #[test]
    fn malformed_json_reports_the_parse_status() {
        let request = CString::new("{not json").expect("no interior NUL");
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        let code = unsafe { urouting_solve_vrp(request.as_ptr(), &mut out) };
        unsafe { urouting_free_string(out) };
        assert_eq!(code, -2);
    }

    #[test]
    fn every_documented_method_is_the_one_that_runs() {
        for method in ["nn", "savings", "ga", "alns"] {
            let mut request = problem(method);
            request["config"] =
                serde_json::json!({ "seed": 7, "max_generations": 20, "max_iterations": 50 });
            let (code, body) = solve(&request);
            assert_eq!(code, 0, "{method}: {body}");
            assert_eq!(body["method_used"], method);
            let served: usize = body["routes"]
                .as_array()
                .expect("routes")
                .iter()
                .map(|r| r.as_array().expect("route").len())
                .sum();
            let unassigned = body["unassigned"].as_array().expect("unassigned").len();
            assert_eq!(
                served + unassigned,
                4,
                "{method}: every customer is either routed or reported: {body}"
            );
        }
    }

    #[test]
    fn customers_the_fleet_cannot_carry_are_reported() {
        // One vehicle of 30 against 50 units of demand: nearest neighbour
        // cannot serve everyone with the fleet it was given. The customers it
        // leaves out used to be absent from the response with nothing saying
        // so -- a partial plan that read as a complete one.
        let (code, body) = solve(&problem("nn"));
        assert_eq!(code, 0, "{body}");
        let unassigned: Vec<u64> = body["unassigned"]
            .as_array()
            .expect("unassigned")
            .iter()
            .map(|v| v.as_u64().expect("id"))
            .collect();
        assert!(!unassigned.is_empty(), "{body}");
        let routed: Vec<u64> = body["routes"]
            .as_array()
            .expect("routes")
            .iter()
            .flat_map(|r| r.as_array().expect("route").iter())
            .map(|v| v.as_u64().expect("id"))
            .collect();
        for id in &unassigned {
            assert!(!routed.contains(id), "{id} is both routed and unassigned");
            assert!((11..=14).contains(id), "{id} is an input id");
        }
    }

    #[test]
    fn genetic_algorithm_settings_reach_the_solver() {
        // A population of one cannot breed. If the request reaches the GA the
        // settings are rejected; if it is quietly solved some other way, they
        // are not.
        let mut request = problem("ga");
        request["config"] = serde_json::json!({ "population_size": 1 });
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert!(
            body["error"]
                .as_str()
                .expect("error")
                .contains("population_size"),
            "{body}"
        );
    }

    #[test]
    fn an_inverted_time_window_is_rejected_not_dropped() {
        // Dropping it would solve the problem without the constraint the
        // caller asked for.
        let mut request = problem("nn");
        request["customers"][0]["time_window"] = serde_json::json!([5.0, 1.0]);
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert!(
            body["error"].as_str().expect("error").contains("11"),
            "{body}"
        );
    }

    #[test]
    fn total_distance_is_the_length_of_the_routes() {
        // Every customer sits away from the depot, so any tour has length.
        for method in ["nn", "savings"] {
            let (code, body) = solve(&problem(method));
            assert_eq!(code, 0, "{method}: {body}");
            let d = body["total_distance"].as_f64().expect("total_distance");
            assert!(d > 0.0, "{method}: total_distance {d}");
        }
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let mut request = problem("nn");
        request["customers"][0]["priority"] = serde_json::json!(2);
        let (code, _) = solve(&request);
        assert_eq!(code, -2);
    }
}
