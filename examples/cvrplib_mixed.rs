//! Measures a fleet of several kinds of vehicle on capacitated instances in the
//! CVRPLIB format (Uchoa et al.'s X set), against each kind alone.
//!
//! The instances are not in this repository: put the `.vrp` files of a CVRPLIB X
//! download into one directory. A mixed fleet is derived from each instance's own
//! capacity `Q`: a small kind (`Q/2`), a middle kind (`Q`) and a large kind
//! (`2Q`), with a fixed cost per route in proportion to the mean distance from the
//! depot to a customer `d`:
//!
//! - scenario `A`: fixed costs `0.4·4d`, `1.0·4d`, `2.2·4d`; the cost per distance
//!   is 1 for all;
//! - scenario `B`: fixed costs `0.3·4d`, `1.0·4d`, `2.5·4d` and a cost per distance
//!   of 0.8, 1.0 and 1.25.
//!
//! ```text
//! cargo run --release --features ffi --example cvrplib_mixed -- <dir> [A|B] [methods]
//! ```
//!
//! `methods` is a comma list of `savings,ga,alns` (default all). Each instance is
//! solved with each kind alone (`S`, `M`, `L`) and with all three (`SML`); the
//! output has the cost (`total_cost`), the routes and the kinds used, and the
//! checker's violations for every plan. A mixed fleet is never dearer than the best
//! of the three alone.

use std::ffi::{CStr, CString};
use std::time::Instant;

use serde_json::{json, Value};
use u_routing::ffi::{urouting_check_solution, urouting_free_string, urouting_solve_vrp};

/// An instance: the depot first, then the customers (`x`, `y`, demand), and
/// the capacity.
fn read(path: &std::path::Path) -> (Vec<(f64, f64, i64)>, i64) {
    let text = std::fs::read_to_string(path).expect("instance file is readable");
    let mut capacity = 0;
    let mut coords: Vec<(f64, f64)> = Vec::new();
    let mut demands: Vec<i64> = Vec::new();
    let mut section = "";
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("CAPACITY") {
            capacity = rest
                .trim_start_matches([' ', ':', '\t'])
                .trim()
                .parse()
                .expect("a capacity");
            continue;
        }
        if line.ends_with("SECTION") {
            section = line;
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match (section, fields.as_slice()) {
            ("NODE_COORD_SECTION", [_, x, y]) => coords.push((
                x.parse().expect("a coordinate"),
                y.parse().expect("a coordinate"),
            )),
            ("DEMAND_SECTION", [_, d]) => demands.push(d.parse().expect("a demand")),
            _ => {}
        }
    }
    let points = coords
        .iter()
        .zip(&demands)
        .map(|(&(x, y), &d)| (x, y, d))
        .collect();
    (points, capacity)
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(&args[0]);
    let scenario = args.get(1).map_or("A", String::as_str).to_string();
    let methods: Vec<String> = args
        .get(2)
        .map_or("savings,ga,alns", String::as_str)
        .split(',')
        .map(String::from)
        .collect();
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vrp"))
        .collect();
    names.sort();
    println!("instance,method,fleet,routes,kinds_used,unassigned,cost,violations,ms");
    for path in &names {
        let name = path.file_stem().and_then(|n| n.to_str()).expect("a name");
        let (points, capacity) = read(path);
        let n = points.len();
        let distances: Vec<Vec<f64>> = points
            .iter()
            .map(|a| {
                points
                    .iter()
                    .map(|b| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt().round())
                    .collect()
            })
            .collect();
        let mean_depot: f64 = distances[0][1..].iter().sum::<f64>() / (n - 1) as f64;
        let f = 4.0 * mean_depot;
        let customers: Vec<Value> = points[1..]
            .iter()
            .enumerate()
            .map(|(i, &(x, y, d))| json!({ "id": i + 1, "x": x, "y": y, "demand": d }))
            .collect();
        let cap = capacity as f64;
        let kinds: Vec<Value> = if scenario == "A" {
            vec![
                json!({ "capacity": (cap * 0.5).floor(), "fixed_cost": 0.4 * f }),
                json!({ "capacity": cap, "fixed_cost": 1.0 * f }),
                json!({ "capacity": (cap * 2.0).floor(), "fixed_cost": 2.2 * f }),
            ]
        } else {
            vec![
                json!({ "capacity": (cap * 0.5).floor(), "fixed_cost": 0.3 * f, "cost_per_distance": 0.8 }),
                json!({ "capacity": cap, "fixed_cost": 1.0 * f, "cost_per_distance": 1.0 }),
                json!({ "capacity": (cap * 2.0).floor(), "fixed_cost": 2.5 * f, "cost_per_distance": 1.25 }),
            ]
        };
        let fleets: Vec<(&str, Vec<Value>)> = vec![
            ("S", vec![kinds[0].clone()]),
            ("M", vec![kinds[1].clone()]),
            ("L", vec![kinds[2].clone()]),
            ("SML", kinds.clone()),
        ];
        for method in &methods {
            for (label, fleet) in &fleets {
                let mut config = if method == "savings" {
                    json!({})
                } else {
                    json!({ "seed": 1 })
                };
                if method == "ga" {
                    config["max_generations"] = json!(100);
                }
                if method == "alns" {
                    config["max_iterations"] = json!(2000);
                }
                let request = json!({
                    "depot_x": points[0].0, "depot_y": points[0].1,
                    "customers": customers, "vehicles": fleet, "method": method,
                    "config": config, "distances": distances,
                });
                let started = Instant::now();
                let (status, solution) = call(urouting_solve_vrp, &request);
                let ms = started.elapsed().as_millis();
                assert_eq!(status, 0, "{name}: {solution}");
                let (_, check) = call(
                    urouting_check_solution,
                    &json!({ "problem": request, "solution": {
                        "routes": solution["routes"],
                        "route_vehicles": solution["route_vehicles"],
                        "unassigned": solution["unassigned"],
                        "total_distance": solution["total_distance"],
                        "total_cost": solution["total_cost"],
                    } }),
                );
                let violations = check["violations"].as_array().map_or(usize::MAX, Vec::len);
                let routes = solution["routes"].as_array().map_or(0, Vec::len);
                let mut used: Vec<u64> = solution["route_vehicles"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                used.sort_unstable();
                used.dedup();
                let used = used
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("+");
                let unassigned = solution["unassigned"].as_array().map_or(0, Vec::len);
                println!(
                    "{name},{method},{label},{routes},{used},{unassigned},{:.1},{violations},{ms}",
                    solution["total_cost"].as_f64().unwrap_or(f64::NAN)
                );
            }
        }
    }
}
