//! Measures the genetic algorithm on capacitated instances in the CVRPLIB
//! format (Uchoa et al.'s X set: `X-n101-k25.vrp` …), with the distances the
//! set defines -- Euclidean, rounded to the nearest integer -- so the cost is
//! comparable with the best known one.
//!
//! The instances are not in this repository. Put the `.vrp` files, and the
//! `.sol` files holding the best known solutions (a last line `Cost <n>`),
//! into one directory (the `X-*.vrp` / `X-*.sol` pairs of a CVRPLIB X download):
//!
//! ```text
//! cargo run --release --features ffi --example cvrplib -- <dir> [generations] [seeds]
//! ```
//!
//! Each instance goes through the C entry points a consumer calls --
//! `urouting_solve_vrp` with method `"ga"`, the rounded distances as
//! `distances` and as many vehicles as customers (the set fixes no fleet),
//! then `urouting_check_solution` on the answer. `generations` sets
//! `config.max_generations` (default: the service's); `seeds` defaults to `1`
//! (a comma list runs each). The cost is the answer's `total_distance`.

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
    let dir = std::path::Path::new(
        args.first()
            .expect("usage: cvrplib <dir> [generations] [seeds]"),
    );
    let generations: Option<usize> = args
        .get(1)
        .filter(|g| !g.is_empty())
        .map(|g| g.parse().expect("a count"));
    let seeds: Vec<u64> = args.get(2).map_or(vec![1], |s| {
        s.split(',').map(|v| v.parse().expect("a seed")).collect()
    });

    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("instance directory is readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vrp"))
        .collect();
    names.sort();

    println!("instance,seed,routes,cost,best,gap_pct,violations,ms");
    let mut gaps = Vec::new();
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
        let customers: Vec<Value> = points[1..]
            .iter()
            .enumerate()
            .map(|(i, &(x, y, d))| json!({ "id": i + 1, "x": x, "y": y, "demand": d }))
            .collect();
        // No fleet limit: as many vehicles as customers.
        let vehicles: Vec<Value> = (1..n).map(|_| json!({ "capacity": capacity })).collect();
        let best: Option<f64> = std::fs::read_to_string(path.with_extension("sol"))
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.trim().strip_prefix("Cost")?.trim().parse().ok())
            });
        for &seed in &seeds {
            let mut config = json!({ "seed": seed });
            if let Some(g) = generations {
                config["max_generations"] = json!(g);
            }
            let request = json!({
                "depot_x": points[0].0, "depot_y": points[0].1,
                "customers": customers, "vehicles": vehicles, "method": "ga",
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
            let cost = solution["total_distance"].as_f64().unwrap_or(f64::NAN);
            let gap = best.map(|b| 100.0 * (cost - b) / b);
            if let Some(g) = gap {
                gaps.push(g);
            }
            println!(
                "{name},{seed},{routes},{cost},{},{},{violations},{ms}",
                best.map_or(String::new(), |b| b.to_string()),
                gap.map_or(String::new(), |g| format!("{g:.2}")),
            );
        }
    }
    if !gaps.is_empty() {
        eprintln!(
            "ga: mean gap {:.2} % over {} runs",
            gaps.iter().sum::<f64>() / gaps.len() as f64,
            gaps.len()
        );
    }
}
