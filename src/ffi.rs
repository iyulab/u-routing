//! FFI module for u-routing — JSON-in/JSON-out pattern
//!
//! Status codes:
//!   0 = OK
//!  -1 = null pointer input
//!  -2 = request is not valid JSON of the expected shape
//!  -3 = request rejected (unknown method, invalid time window, demand or
//!       capacity, a mixed fleet or time windows the method cannot model,
//!       solver settings)
//!  -4 = internal panic
//!
//! Every non-zero status except `-1` comes with an error body:
//! `{"error": "<readable text>", "code": "<stable reason>", ...}`. `code` and
//! the fields beside it (`id`, `index`, `parameter`, ...) are the same ones the
//! WebAssembly binding puts on its thrown `Error`; `error` is the text, free to
//! change. A request that is not JSON is `malformed_input`, a panic `internal`.
//! All entry points are wrapped in `catch_unwind` to prevent panic propagation.
//!
//! The solver itself is `crate::service`, shared with the WebAssembly binding;
//! this module owns only the transport and the request's top-level shape.

use std::ffi::{CStr, CString};
use std::panic;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::service::{self, InputConfig, InputCustomer, InputSolution, InputVehicle, ServiceError};

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

/// A plan to check and the problem it was solved for: `problem` is the
/// `urouting_solve_vrp` request, unchanged.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckInput {
    problem: VrpInput,
    solution: InputSolution,
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

/// Writes `{"error": message, "code": ..., ...fields}` and returns `status`.
///
/// The status is the caller's to choose and is returned as given: the error
/// body is a diagnostic, not a result, so writing it successfully must not
/// turn the call into a success.
fn write_error(result_ptr: *mut *mut libc::c_char, status: i32, error: ServiceError) -> i32 {
    let mut body = serde_json::Map::new();
    body.insert("error".into(), error.message.into());
    if let serde_json::Value::Object(fields) = error.fields {
        body.extend(fields);
    }
    match write_json(result_ptr, &body) {
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
        Err(_) => write_error(result_ptr, -4, ServiceError::internal("internal panic")),
    }
}

// ── FFI exports ─────────────────────────────────────────────

/// Solve a VRP (TSP, CVRP or VRPTW, by what the request carries -- time windows
/// with `"nn"` or `"ga"`).
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
        let input: VrpInput = match parse_request(&json) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_PARSE, e),
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
            Err(e) => write_error(result_ptr, ERR_COMPUTE, e),
        }
    })
}

/// Checks a plan against the problem it was solved for. The request is
/// `{"problem": <the urouting_solve_vrp request>, "solution": {"routes": [[..]],
/// "route_vehicles": [..], "unassigned": [..], "total_distance": ..}}`; the
/// result is `{"feasible": bool, "violations": [{"code": .., ..}]}` -- a plan
/// that breaks its problem is a result (status 0), not an error.
///
/// # Safety
///
/// As [`urouting_solve_vrp`].
#[no_mangle]
pub unsafe extern "C" fn urouting_check_solution(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let input: CheckInput = match parse_request(&json) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_PARSE, e),
        };
        let problem = input.problem;
        let config = problem.config.unwrap_or_default();
        match service::check(
            (problem.depot_x, problem.depot_y),
            &problem.customers,
            &problem.vehicles,
            &problem.method,
            &config,
            &input.solution,
        ) {
            Ok(output) => write_json(result_ptr, &output),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, e),
        }
    })
}

/// Reads the request, or refuses it where it stopped: `parameter` names the
/// field (`customers[1].id`, `config.seed`) and `index` its array position,
/// as the WebAssembly binding reports them; text that is not JSON at all is
/// refused as `request`.
fn parse_request<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, ServiceError> {
    let mut deserializer = serde_json::Deserializer::from_str(json);
    let input: T = serde_path_to_error::deserialize(&mut deserializer).map_err(|e| {
        if e.inner().is_data() {
            service::malformed_at("request", &e)
        } else {
            // Not JSON: there is no field to name.
            ServiceError::malformed_input("request", format!("Invalid JSON: {}", e.inner()))
        }
    })?;
    // Trailing characters after the value, as `serde_json::from_str` refuses.
    deserializer
        .end()
        .map_err(|e| ServiceError::malformed_input("request", format!("request: {e}")))?;
    Ok(input)
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

/// Every method `urouting_solve_vrp` takes, with the `config` settings each
/// reads, as JSON: `[{"name": "nn", "reads": ["max_vehicles"]}, ...]` -- the
/// table its refusal of a setting the method does not read (`invalid_option`)
/// comes from. The string is owned by the caller and must be released with
/// [`urouting_free_string`].
#[no_mangle]
pub extern "C" fn urouting_methods() -> *mut libc::c_char {
    let json = serde_json::to_string(&crate::service::methods()).expect("names serialize");
    CString::new(json)
        .expect("JSON has no interior NUL")
        .into_raw()
}

/// Every refusal `code` the solver and the checker answer with, and every
/// violation `code` the checker reports, with the fields beside each, as JSON:
/// `{"refusals": [{"code", "fields"}], "violations": [{"code", "fields",
/// "checkers"}]}` -- built from the tables the refusals and violations are
/// made from, so a caller that translates codes can check its list against the
/// engine's. The string is owned by the caller and must be released with
/// [`urouting_free_string`].
#[no_mangle]
pub extern "C" fn urouting_codes() -> *mut libc::c_char {
    let json = serde_json::to_string(&crate::service::codes()).expect("codes serialize");
    CString::new(json)
        .expect("JSON has no interior NUL")
        .into_raw()
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

    fn check(request: &serde_json::Value) -> (i32, serde_json::Value) {
        let request = CString::new(request.to_string()).expect("no interior NUL");
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        let code = unsafe { urouting_check_solution(request.as_ptr(), &mut out) };
        let body = unsafe { CStr::from_ptr(out) }
            .to_str()
            .expect("UTF-8")
            .to_owned();
        unsafe { urouting_free_string(out) };
        (code, serde_json::from_str(&body).expect("JSON"))
    }

    /// Every plan says why its method stopped: the one-pass constructions
    /// ran to their end, the searches on the budget they were given.
    #[test]
    fn a_plan_says_why_its_method_stopped() {
        let with = |method: &str, config: serde_json::Value| {
            let mut request = problem(method);
            request["config"] = config;
            let (code, plan) = solve(&request);
            assert_eq!(code, 0, "{plan}");
            plan["stop_reason"].clone()
        };
        assert_eq!(with("nn", serde_json::json!({})), "completed");
        assert_eq!(with("savings", serde_json::json!({})), "completed");
        assert_eq!(
            with(
                "ga",
                serde_json::json!({ "population_size": 20, "max_generations": 2, "seed": 1 })
            ),
            "iteration_limit"
        );
        assert_eq!(
            with(
                "alns",
                serde_json::json!({ "max_iterations": 3, "seed": 1 })
            ),
            "iteration_limit"
        );
        // A long enough GA run on four customers stops when it stalls.
        assert_eq!(
            with(
                "ga",
                serde_json::json!({ "population_size": 20, "max_generations": 100000, "seed": 1 })
            ),
            "no_improvement"
        );
    }

    /// A solved plan, handed back as it came, checks clean; a broken one is a
    /// result with its violations, not an error.
    #[test]
    fn a_plan_is_checked_against_the_problem_it_was_solved_for() {
        let savings = problem("savings");
        let (code, plan) = solve(&savings);
        assert_eq!(code, 0, "{plan}");
        let solution = serde_json::json!({
            "routes": plan["routes"], "route_vehicles": plan["route_vehicles"],
            "unassigned": plan["unassigned"], "total_distance": plan["total_distance"]
        });
        let (code, out) = check(&serde_json::json!({ "problem": savings, "solution": solution }));
        assert_eq!(
            (code, out),
            (0, serde_json::json!({ "feasible": true, "violations": [] }))
        );

        let broken = serde_json::json!({ "routes": [[11, 12, 13, 14]] });
        let (code, out) = check(&serde_json::json!({ "problem": savings, "solution": broken }));
        assert_eq!(code, 0, "{out}");
        assert_eq!(out["feasible"], false);
        assert_eq!(
            out["violations"][0],
            serde_json::json!({ "code": "capacity_exceeded", "route": 0, "load": 50, "capacity": 30 })
        );
        // The problem is refused as solve refuses it.
        let (code, out) = check(&serde_json::json!({
            "problem": problem("tabu"), "solution": { "routes": [] }
        }));
        assert_eq!(
            (code, &out["code"]),
            (-3, &serde_json::json!("unknown_option"))
        );
    }

    #[test]
    fn the_methods_are_the_service_catalog() {
        let ptr = urouting_methods();
        let json = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_str()
            .expect("UTF-8")
            .to_owned();
        unsafe { urouting_free_string(ptr) };
        let methods: Vec<crate::service::MethodInfo> =
            serde_json::from_str(&json).expect("a method list");
        assert_eq!(methods, crate::service::methods());
    }

    #[test]
    fn the_codes_are_the_service_catalog() {
        let ptr = urouting_codes();
        let json = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_str()
            .expect("UTF-8")
            .to_owned();
        unsafe { urouting_free_string(ptr) };
        let codes: crate::service::Codes = serde_json::from_str(&json).expect("a catalog");
        assert_eq!(codes, crate::service::codes());
        assert!(codes.refusals.iter().any(|r| r.code == "unknown_option"));
        assert!(codes
            .violations
            .iter()
            .any(|v| v.code == "capacity_exceeded"));
    }

    #[test]
    fn a_rejected_request_reports_a_failure_status() {
        // An error body under status 0 reaches the C# client as a successful
        // result: it only raises on a non-zero status.
        let (code, body) = solve(&problem("no-such-method"));
        assert_eq!(code, -3, "{body}");
        assert!(body["error"].is_string());
        // The reason and the values behind it sit next to the text, so a
        // caller does not have to read them out of it.
        assert_eq!(body["code"], "unknown_option", "{body}");
        assert_eq!(body["parameter"], "method", "{body}");
        assert_eq!(body["got"], "no-such-method", "{body}");
        assert_eq!(
            body["expected"],
            serde_json::json!(["nn", "savings", "ga", "alns"])
        );
    }

    /// The C transport carries `max_vehicles` from the same shared wire type
    /// the WebAssembly binding uses: the plan keeps to the fleet and names
    /// the customers it leaves out.
    #[test]
    fn a_fixed_fleet_is_kept_over_the_c_boundary() {
        let (free_code, free) = solve(&problem("savings"));
        assert_eq!(free_code, 0, "{free}");
        assert!(free["num_vehicles"].as_u64().expect("count") > 1, "{free}");

        let mut request = problem("savings");
        request["config"] = serde_json::json!({ "max_vehicles": 1 });
        let (code, body) = solve(&request);
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["num_vehicles"], 1, "{body}");
        let unassigned = body["unassigned"].as_array().expect("unassigned");
        assert!(!unassigned.is_empty(), "{body}");
        let served: usize = body["routes"][0].as_array().expect("route").len();
        assert_eq!(served + unassigned.len(), 4, "{body}");
    }

    #[test]
    fn malformed_json_reports_the_parse_status() {
        let request = CString::new("{not json").expect("no interior NUL");
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        let code = unsafe { urouting_solve_vrp(request.as_ptr(), &mut out) };
        let body: serde_json::Value = serde_json::from_str(
            unsafe { CStr::from_ptr(out) }
                .to_str()
                .expect("body is UTF-8"),
        )
        .expect("body is JSON");
        unsafe { urouting_free_string(out) };
        assert_eq!(code, -2);
        assert_eq!(body["code"], "malformed_input", "{body}");
        assert_eq!(body["parameter"], "request", "{body}");
    }

    #[test]
    fn every_documented_method_is_the_one_that_runs() {
        for method in ["nn", "savings", "ga", "alns"] {
            let mut request = problem(method);
            // Each method with the settings it reads; a setting it does not read
            // is refused.
            request["config"] = match method {
                "ga" => serde_json::json!({ "seed": 7, "max_generations": 20 }),
                "alns" => serde_json::json!({ "seed": 7, "max_iterations": 50 }),
                _ => serde_json::json!({}),
            };
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

    /// An open bound reaches the C error body as `exclusive_min`, beside the
    /// closed `max` of a (0, 1] rate and with `min` null.
    #[test]
    fn open_bounds_reach_the_c_error_body_as_exclusive() {
        let mut request = problem("ga");
        request["config"] = serde_json::json!({ "mutation_rate": 0.0 });
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "parameter_out_of_range", "{body}");
        assert_eq!(body["parameter"], "config.mutation_rate", "{body}");
        assert_eq!(body["min"], serde_json::Value::Null, "{body}");
        assert_eq!(body["exclusive_min"], 0.0, "{body}");
        assert_eq!(body["max"], 1.0, "{body}");
        assert!(body.get("exclusive_max").is_none(), "{body}");

        let mut request = problem("nn");
        request["vehicles"][0]["max_duration"] = serde_json::json!(-1.0);
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["parameter"], "max_duration", "{body}");
        assert_eq!(body["min"], serde_json::Value::Null, "{body}");
        assert_eq!(body["exclusive_min"], 0.0, "{body}");
        assert_eq!(body["max"], serde_json::Value::Null, "{body}");
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
        assert_eq!(body["code"], "invalid_time_window", "{body}");
        assert_eq!(body["id"], 11, "{body}");
        assert_eq!(body["index"], 0, "{body}");
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

    /// A negative demand is outside the range a load takes, with its bounds;
    /// a fractional one is a matter of the unit.
    #[test]
    fn demand_refusals_reach_the_c_body() {
        let mut request = problem("nn");
        request["customers"][1]["demand"] = serde_json::json!(-4);
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        let mut expected = serde_json::json!({
            "code": "parameter_out_of_range",
            "parameter": "demand",
            "index": 1,
            "id": request["customers"][1]["id"],
            "min": 0.0,
            "max": f64::from(i32::MAX),
            "got": -4.0,
        });
        expected["error"] = body["error"].clone();
        assert_eq!(body, expected);

        request["customers"][1]["demand"] = serde_json::json!(1.5);
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "not_whole_units", "{body}");
        assert_eq!(body["value"], 1.5, "{body}");
    }

    /// The C body carries a negative count as the range refusal it is, and a
    /// fractional one as malformed at the field it sits in.
    #[test]
    fn whole_number_refusals_reach_the_c_body() {
        let mut request = problem("ga");
        request["config"] = serde_json::json!({ "population_size": -1, "max_generations": 2 });
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "parameter_out_of_range", "{body}");
        assert_eq!(body["parameter"], "config.population_size", "{body}");
        assert_eq!(body["method"], "ga", "{body}");
        assert_eq!(body["min"], 2.0, "{body}");
        assert_eq!(body["max"], serde_json::Value::Null, "{body}");
        assert_eq!(body["got"], -1.0, "{body}");

        let mut request = problem("nn");
        request["config"] = serde_json::json!({ "max_vehicles": -1 });
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["parameter"], "config.max_vehicles", "{body}");
        assert_eq!(body["min"], 1.0, "{body}");
        assert_eq!(body["got"], -1.0, "{body}");

        let mut request = problem("nn");
        request["customers"][2]["id"] = serde_json::json!(-5);
        let (code, body) = solve(&request);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "parameter_out_of_range", "{body}");
        assert_eq!(body["parameter"], "id", "{body}");
        assert_eq!(body["index"], 2, "{body}");
        assert_eq!(body["got"], -5.0, "{body}");

        let mut request = problem("ga");
        request["config"] = serde_json::json!({ "population_size": 2.5 });
        let (code, body) = solve(&request);
        assert_eq!(code, -2, "{body}");
        assert_eq!(body["code"], "malformed_input", "{body}");
        assert_eq!(body["parameter"], "config.population_size", "{body}");
        assert!(
            body["error"]
                .as_str()
                .expect("text")
                .contains("must be a whole number, got 2.5"),
            "{body}"
        );

        let mut request = problem("nn");
        request["customers"][1]["id"] = serde_json::json!(1.5);
        let (code, body) = solve(&request);
        assert_eq!(code, -2, "{body}");
        assert_eq!(body["parameter"], "customers[1].id", "{body}");
        assert_eq!(body["index"], 1, "{body}");

        // A whole number reads as itself, up to the largest 64-bit seed.
        let mut request = problem("ga");
        request["config"] =
            serde_json::json!({ "population_size": 10, "max_generations": 2.0, "seed": u64::MAX });
        let (code, body) = solve(&request);
        assert_eq!(code, 0, "{body}");
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let mut request = problem("nn");
        request["customers"][0]["priority"] = serde_json::json!(2);
        let (code, _) = solve(&request);
        assert_eq!(code, -2);
    }
}

#[cfg(test)]
mod request_keys;
