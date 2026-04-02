//! FFI module for u-routing — JSON-in/JSON-out pattern

#[cfg(feature = "ffi")]
use std::ffi::{CStr, CString};

#[cfg(feature = "ffi")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "ffi")]
use crate::constructive::{clarke_wright_savings, nearest_neighbor};
#[cfg(feature = "ffi")]
use crate::distance::DistanceMatrix;
#[cfg(feature = "ffi")]
use crate::models::{Customer, TimeWindow, Vehicle};

// ── Types ───────────────────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct InputCustomer {
    id: usize,
    x: f64,
    y: f64,
    #[serde(default)]
    demand: f64,
    #[serde(default)]
    service_time: f64,
    #[serde(default)]
    time_window: Option<[f64; 2]>,
}

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct InputVehicle {
    #[serde(default = "default_capacity")]
    capacity: f64,
}

#[cfg(feature = "ffi")]
fn default_capacity() -> f64 { 1e9 }

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct VrpInput {
    customers: Vec<InputCustomer>,
    #[serde(default)]
    vehicles: Vec<InputVehicle>,
    depot_x: f64,
    depot_y: f64,
    #[serde(default = "default_method")]
    method: String,
}

#[cfg(feature = "ffi")]
fn default_method() -> String { "nn".to_string() }

#[cfg(feature = "ffi")]
#[derive(Serialize)]
struct VrpOutput {
    routes: Vec<Vec<usize>>,
    total_distance: f64,
    num_vehicles: usize,
    method_used: String,
}

// ── Helpers ─────────────────────────────────────────────────

#[cfg(feature = "ffi")]
unsafe fn read_json(ptr: *const libc::c_char) -> Result<String, i32> {
    if ptr.is_null() { return Err(-1); }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().map(|s| s.to_string()).map_err(|_| -2)
}

#[cfg(feature = "ffi")]
fn write_json<T: Serialize>(result_ptr: *mut *mut libc::c_char, value: &T) -> i32 {
    match serde_json::to_string(value) {
        Ok(json) => match CString::new(json) {
            Ok(cstr) => { unsafe { *result_ptr = cstr.into_raw() }; 0 }
            Err(_) => -3,
        },
        Err(_) => -3,
    }
}

#[cfg(feature = "ffi")]
fn write_error(result_ptr: *mut *mut libc::c_char, msg: &str) -> i32 {
    let err = serde_json::json!({ "error": msg });
    write_json(result_ptr, &err);
    -3
}

#[cfg(feature = "ffi")]
fn solve_internal(input: &VrpInput) -> Result<VrpOutput, String> {
    if input.customers.is_empty() {
        return Err("customers must not be empty".into());
    }

    // Build customer list (index 0 = depot)
    let mut customers = Vec::with_capacity(input.customers.len() + 1);
    customers.push(Customer::depot(input.depot_x, input.depot_y));

    let mut id_map = Vec::with_capacity(input.customers.len());
    for ic in &input.customers {
        let idx = customers.len();
        id_map.push(ic.id);
        let mut c = Customer::new(idx, ic.x, ic.y, ic.demand.round() as i32, ic.service_time);
        if let Some([ready, due]) = ic.time_window {
            if let Some(tw) = TimeWindow::new(ready, due) {
                c = c.with_time_window(tw);
            }
        }
        customers.push(c);
    }

    let dm = DistanceMatrix::from_customers(&customers);

    let cap = input.vehicles.first()
        .map(|v| v.capacity.round() as i32)
        .unwrap_or(i32::MAX);

    let vehicles: Vec<Vehicle> = if input.vehicles.is_empty() {
        vec![Vehicle::new(0, cap)]
    } else {
        input.vehicles.iter().enumerate()
            .map(|(i, v)| Vehicle::new(i, v.capacity.round() as i32))
            .collect()
    };

    let method = input.method.to_lowercase();

    let solution = match method.as_str() {
        "savings" => {
            clarke_wright_savings(&customers, &dm, &vehicles[0])
        }
        _ => {
            // Nearest neighbor (default)
            nearest_neighbor(&customers, &dm, &vehicles)
        }
    };

    // Extract routes with original customer IDs
    let routes: Vec<Vec<usize>> = solution.routes().iter()
        .map(|r| r.customer_ids().into_iter().map(|i| {
            if i > 0 && i <= id_map.len() { id_map[i - 1] } else { i }
        }).collect())
        .collect();

    Ok(VrpOutput {
        total_distance: solution.total_cost(),
        num_vehicles: routes.len(),
        routes,
        method_used: method,
    })
}

// ── FFI exports ─────────────────────────────────────────────

/// Solve VRP (unified TSP/CVRP/VRPTW interface)
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn urouting_solve_vrp(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };
    let input: VrpInput = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(e) => return write_error(result_ptr, &format!("JSON parse error: {e}")),
    };
    match solve_internal(&input) {
        Ok(output) => write_json(result_ptr, &output),
        Err(e) => write_error(result_ptr, &e),
    }
}

/// Free string allocated by u-routing FFI
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn urouting_free_string(ptr: *mut libc::c_char) {
    if !ptr.is_null() { unsafe { drop(CString::from_raw(ptr)) }; }
}

/// Get u-routing version
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn urouting_version() -> *mut libc::c_char {
    let version = env!("CARGO_PKG_VERSION");
    CString::new(version).unwrap().into_raw()
}
