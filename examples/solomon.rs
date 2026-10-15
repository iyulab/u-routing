//! Measures the solver on Solomon's VRPTW instances, through the C entry
//! points a consumer calls (`urouting_solve_vrp`, then `urouting_check_solution`
//! on every answer).
//!
//! The instances are not in this repository. Download the 100-customer set
//! (e.g. the zip on SINTEF's "Solomon benchmark" page) and point the harness at
//! the directory holding `c101.txt` …:
//!
//! ```text
//! cargo run --release --features ffi --example solomon -- <dir> [bks.csv] [methods]
//! ```
//!
//! `bks.csv` (optional) has lines `instance,vehicles,distance` -- best known
//! results under the usual hierarchical objective (vehicles first, then
//! distance); the gap columns are against it. `methods` is a comma list of
//! `nn,savings,ga,alns` (default all), each run once with seed 1.
//!
//! Mapping: the depot's due date is every vehicle's `max_duration` (return by
//! it), the fleet is the instance's vehicle count at its capacity
//! (`config.max_vehicles` too), distance and travel time are Euclidean in
//! double precision, as the benchmark defines them.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::time::Instant;

use serde_json::{json, Value};
use u_routing::ffi::{urouting_check_solution, urouting_free_string, urouting_solve_vrp};

struct Instance {
    name: String,
    vehicles: usize,
    capacity: f64,
    depot: (f64, f64),
    horizon: f64,
    customers: Vec<Value>,
}

fn parse(name: &str, text: &str) -> Option<Instance> {
    let rows: Vec<Vec<f64>> = text
        .lines()
        .map(|l| {
            l.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect::<Vec<f64>>()
        })
        .collect();
    // The vehicle line is the first with exactly two numbers; customer lines have seven.
    let fleet = rows.iter().find(|r| r.len() == 2)?;
    let table: Vec<&Vec<f64>> = rows.iter().filter(|r| r.len() == 7).collect();
    let depot = table.first()?;
    let customers = table[1..]
        .iter()
        .map(|r| {
            json!({
                "id": r[0] as i64, "x": r[1], "y": r[2], "demand": r[3],
                "time_window": [r[4], r[5]], "service_time": r[6],
            })
        })
        .collect();
    Some(Instance {
        name: name.to_string(),
        vehicles: fleet[0] as usize,
        capacity: fleet[1],
        depot: (depot[1], depot[2]),
        horizon: depot[5],
        customers,
    })
}

fn call(
    entry: unsafe extern "C" fn(*const std::os::raw::c_char, *mut *mut std::os::raw::c_char) -> i32,
    request: &Value,
) -> (i32, Value) {
    let text = CString::new(request.to_string()).expect("JSON has no NUL");
    let mut out = std::ptr::null_mut();
    let status = unsafe { entry(text.as_ptr(), &mut out) };
    let body = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    unsafe { urouting_free_string(out) };
    (status, serde_json::from_str(&body).unwrap_or(Value::Null))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: solomon <dir> [bks.csv] [methods]");
    let bks: HashMap<String, (f64, f64)> = args
        .next()
        .filter(|p| !p.is_empty())
        .map(|p| {
            std::fs::read_to_string(p)
                .expect("bks.csv")
                .lines()
                .filter_map(|l| {
                    let f: Vec<&str> = l.split(',').map(str::trim).collect();
                    Some((
                        f.first()?.to_lowercase(),
                        (f.get(1)?.parse().ok()?, f.get(2)?.parse().ok()?),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let methods: Vec<String> = args
        .next()
        .unwrap_or_else(|| "nn,savings,ga,alns".into())
        .split(',')
        .map(String::from)
        .collect();

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("instance directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("txt")))
        .collect();
    files.sort();

    println!("instance,method,routes,distance,unassigned,violations,ms,bks_vehicles,bks_distance,gap_pct");
    for path in files {
        let name = path
            .file_stem()
            .expect("a file name")
            .to_string_lossy()
            .to_lowercase();
        let text = std::fs::read_to_string(&path).expect("readable instance");
        let Some(inst) = parse(&name, &text) else {
            eprintln!("{name}: not a Solomon instance, skipped");
            continue;
        };
        let vehicles: Vec<Value> = (0..inst.vehicles)
            .map(|_| json!({ "capacity": inst.capacity, "max_duration": inst.horizon }))
            .collect();
        for method in &methods {
            // Only the settings the method reads: any other is refused
            // (`invalid_option`, `urouting_methods`).
            let config = if matches!(method.as_str(), "ga" | "alns") {
                json!({ "seed": 1, "max_vehicles": inst.vehicles })
            } else {
                json!({ "max_vehicles": inst.vehicles })
            };
            let request = json!({
                "depot_x": inst.depot.0, "depot_y": inst.depot.1,
                "customers": inst.customers, "vehicles": vehicles, "method": method,
                "config": config,
            });
            let started = Instant::now();
            let (status, solution) = call(urouting_solve_vrp, &request);
            let ms = started.elapsed().as_millis();
            if status != 0 {
                println!(
                    "{},{method},refused,,,,{ms},,,{}",
                    inst.name, solution["code"]
                );
                continue;
            }
            let (_, check) = call(
                urouting_check_solution,
                // The checker takes the plan's four keys, not the whole answer.
                &json!({ "problem": request, "solution": {
                    "routes": solution["routes"],
                    "route_vehicles": solution["route_vehicles"],
                    "unassigned": solution["unassigned"],
                    "total_distance": solution["total_distance"],
                } }),
            );
            let violations = check["violations"].as_array().map_or(usize::MAX, Vec::len);
            let routes = solution["routes"].as_array().map_or(0, Vec::len);
            let distance = solution["total_distance"].as_f64().unwrap_or(f64::NAN);
            let unassigned = solution["unassigned"].as_array().map_or(0, Vec::len);
            let (bv, bd) = bks
                .get(&inst.name)
                .or_else(|| bks.get(&format!("{}b", inst.name)))
                .copied()
                .unwrap_or((f64::NAN, f64::NAN));
            let gap = 100.0 * (distance / bd - 1.0);
            println!(
                "{},{method},{routes},{distance:.2},{unassigned},{violations},{ms},{bv},{bd},{gap:.1}",
                inst.name
            );
        }
    }
}
