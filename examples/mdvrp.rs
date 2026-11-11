//! Measures the solver on Cordeau's multi-depot VRP instances (`p01` ... `p23`,
//! `pr01` ... `pr10`), through the C entry points a consumer calls
//! (`urouting_solve_vrp`, then `urouting_check_solution` on every answer).
//!
//! The instances are not in this repository. Download them (the zip of
//! instances and the zip of best known results on the "multiple depot VRP
//! instances" page of the University of Malaga's VRP web site) and point the
//! harness at the directories holding `p01` ... and `p01.res` ...:
//!
//! ```text
//! cargo run --release --features ffi --example mdvrp -- <instances> [best-known] [methods] [settings]
//! ```
//!
//! `best-known` (optional) holds one `<name>.res` per instance whose first
//! line is the best known total distance; the gap column is against it.
//! `methods` is a comma list of `nn,savings,ga,alns` (default all), each run
//! once with seed 1. `settings` (optional) is a JSON object merged into
//! `config` for `ga` and `alns`, e.g. `{"max_generations":300}`.
//!
//! Instance file: a first line `type m n t` -- `m` vehicles at each of the `t`
//! depots, `n` customers -- then one line `D Q` per depot (the longest a
//! route may take, 0 for no limit, and the capacity), then one line per customer
//! (`id x y service demand ...`), then one per depot (`id x y ...`).
//!
//! Mapping: every depot has one kind of vehicle of its capacity, `m` of them
//! (`count`), limited to `D` in duration when `D` is not 0 (service time counts
//! in a route's duration, only distance in its cost); distance and travel time
//! are Euclidean in double precision.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::time::Instant;

use serde_json::{json, Value};
use u_routing::ffi::{urouting_check_solution, urouting_free_string, urouting_solve_vrp};

struct Instance {
    name: String,
    per_depot: usize,
    /// `(max duration, capacity)` of each depot's vehicles.
    limits: Vec<(f64, f64)>,
    depots: Vec<(f64, f64)>,
    customers: Vec<Value>,
}

fn parse(name: &str, text: &str) -> Option<Instance> {
    let rows: Vec<Vec<f64>> = text
        .lines()
        .map(|l| {
            l.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .filter(|r: &Vec<f64>| !r.is_empty())
        .collect();
    let head = rows.first()?;
    let (per_depot, n, t) = (
        *head.get(1)? as usize,
        *head.get(2)? as usize,
        *head.get(3)? as usize,
    );
    let limits: Vec<(f64, f64)> = rows
        .get(1..=t)?
        .iter()
        .map(|r| Some((*r.first()?, *r.get(1)?)))
        .collect::<Option<_>>()?;
    let customers: Vec<Value> = rows
        .get(t + 1..t + 1 + n)?
        .iter()
        .map(|r| {
            json!({
                "id": r[0] as i64, "x": r[1], "y": r[2], "service_time": r[3], "demand": r[4],
            })
        })
        .collect();
    let depots: Vec<(f64, f64)> = rows
        .get(t + 1 + n..t + 1 + n + t)?
        .iter()
        .map(|r| (r[1], r[2]))
        .collect();
    Some(Instance {
        name: name.to_string(),
        per_depot,
        limits,
        depots,
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
        .expect("usage: mdvrp <instances> [best-known] [methods] [settings]");
    let best_dir = args.next().filter(|p| !p.is_empty());
    let methods: Vec<String> = args
        .next()
        .unwrap_or_else(|| "nn,savings,ga,alns".into())
        .split(',')
        .map(String::from)
        .collect();
    let settings: serde_json::Map<String, Value> = args
        .next()
        .map(|s| serde_json::from_str(&s).expect("settings: a JSON object"))
        .unwrap_or_default();

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("instance directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_none())
        .collect();
    files.sort();
    let best: HashMap<String, f64> = best_dir
        .map(|d| {
            std::fs::read_dir(d)
                .expect("best-known directory")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "res"))
                .filter_map(|p| {
                    let name = p.file_stem()?.to_string_lossy().to_lowercase();
                    let first = std::fs::read_to_string(&p).ok()?;
                    Some((name, first.lines().next()?.trim().parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_default();

    println!("instance,method,routes,distance,unassigned,violations,ms,bks_distance,gap_pct");
    for path in files {
        let name = path
            .file_stem()
            .expect("a file name")
            .to_string_lossy()
            .to_lowercase();
        let text = std::fs::read_to_string(&path).expect("readable instance");
        let Some(inst) = parse(&name, &text) else {
            eprintln!("{name}: not a Cordeau instance, skipped");
            continue;
        };
        let vehicles: Vec<Value> = inst
            .limits
            .iter()
            .enumerate()
            .map(|(depot, &(duration, capacity))| {
                let mut kind =
                    json!({ "capacity": capacity, "count": inst.per_depot, "depot": depot });
                if duration > 0.0 {
                    kind["max_duration"] = json!(duration);
                }
                kind
            })
            .collect();
        let depots: Vec<Value> = inst
            .depots
            .iter()
            .map(|&(x, y)| json!({ "x": x, "y": y }))
            .collect();
        for method in &methods {
            // Only the settings the method reads: any other is refused.
            let config = if matches!(method.as_str(), "ga" | "alns") {
                let mut config = json!({ "seed": 1 });
                config
                    .as_object_mut()
                    .expect("an object")
                    .extend(settings.clone());
                config
            } else {
                json!({})
            };
            let request = json!({
                "depots": depots, "customers": inst.customers, "vehicles": vehicles,
                "method": method, "config": config,
            });
            let started = Instant::now();
            let (status, solution) = call(urouting_solve_vrp, &request);
            let ms = started.elapsed().as_millis();
            if status != 0 {
                println!(
                    "{},{method},refused,,,,{ms},,{}",
                    inst.name, solution["code"]
                );
                continue;
            }
            let (_, check) = call(
                urouting_check_solution,
                // The checker takes the plan's keys, not the whole answer.
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
            let distance = solution["total_distance"].as_f64().unwrap_or(f64::NAN);
            let unassigned = solution["unassigned"].as_array().map_or(0, Vec::len);
            let bks = best.get(&inst.name).copied().unwrap_or(f64::NAN);
            let gap = 100.0 * (distance / bks - 1.0);
            println!(
                "{},{method},{routes},{distance:.2},{unassigned},{violations},{ms},{bks},{gap:.1}",
                inst.name
            );
        }
    }
}
