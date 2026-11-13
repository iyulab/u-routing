//! Measures what `config.balance` costs: on capacitated instances in the
//! CVRPLIB format, the plan of each method without a balance, then with
//! `quantity: "distance"` at growing weights -- the cost added and the longest
//! route removed, in percent of the plan without one (the trade-off curve the
//! workload-balance literature reports, Matl et al. 2018).
//!
//! ```text
//! cargo run --release --features ffi --example cvrplib_balance -- <file.vrp>... [--methods savings,alns]
//! ```

use std::ffi::{CStr, CString};

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
    let mut files: Vec<String> = Vec::new();
    let mut methods = vec!["savings".to_string(), "alns".to_string()];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--methods" {
            methods = args
                .next()
                .expect("a method list")
                .split(',')
                .map(String::from)
                .collect();
        } else {
            files.push(a);
        }
    }
    println!("instance,method,weight,routes,cost,longest,cost_pct,longest_pct,feasible");
    for file in &files {
        let path = std::path::Path::new(file);
        let name = path.file_stem().and_then(|n| n.to_str()).expect("a name");
        let (points, capacity) = read(path);
        let distances: Vec<Vec<f64>> = points
            .iter()
            .map(|a| {
                points
                    .iter()
                    .map(|b| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt().round())
                    .collect()
            })
            .collect();
        let customers: Vec<Value> = points[1..]
            .iter()
            .enumerate()
            .map(|(i, &(x, y, d))| json!({ "id": i + 1, "x": x, "y": y, "demand": d }))
            .collect();
        for method in &methods {
            let mut base: Option<(f64, f64)> = None;
            for weight in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0] {
                let mut config = json!({});
                if method == "alns" {
                    config["seed"] = json!(1);
                }
                if weight > 0.0 {
                    config["balance"] = json!({ "quantity": "distance", "weight": weight });
                }
                let request = json!({
                    "depots": [{ "x": points[0].0, "y": points[0].1 }], "customers": customers,
                    "vehicles": [{ "capacity": capacity }], "method": method,
                    "config": config, "distances": distances,
                });
                let (status, out) = call(urouting_solve_vrp, &request);
                assert_eq!(status, 0, "{name} {method} {weight}: {out}");
                let cost = out["total_cost"].as_f64().expect("a cost");
                let longest = out["route_summaries"]
                    .as_array()
                    .expect("summaries")
                    .iter()
                    .map(|r| r["distance"].as_f64().expect("a distance"))
                    .fold(0.0, f64::max);
                let plan = json!({
                    "routes": out["routes"], "route_vehicles": out["route_vehicles"],
                    "unassigned": out["unassigned"], "total_cost": out["total_cost"],
                    "equity": out["equity"], "objective": out["objective"],
                });
                let (_, checked) = call(
                    urouting_check_solution,
                    &json!({ "problem": request, "solution": plan }),
                );
                let (c0, l0) = *base.get_or_insert((cost, longest));
                println!(
                    "{name},{method},{weight},{},{cost:.0},{longest:.0},{:+.2},{:+.2},{}",
                    out["routes"].as_array().map_or(0, Vec::len),
                    100.0 * (cost - c0) / c0,
                    100.0 * (longest - l0) / l0,
                    checked["feasible"]
                );
            }
        }
    }
}
