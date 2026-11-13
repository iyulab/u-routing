//! Measures the solver on Li & Lim's pickup-and-delivery instances with time windows, through the C
//! entry points a consumer calls (`urouting_solve_vrp`, then `urouting_check_solution` on every
//! answer) -- the counterpart of `pdptw`, which measures the library pieces directly.
//!
//! ```text
//! cargo run --release --features ffi --example pdptw_service -- <dir> [bks.csv] [methods] [filter] [settings]
//! ```
//!
//! `methods` is a comma list of `nn,savings,alns` (default all); `filter` keeps the instances whose
//! name starts with it; `settings` is a JSON object merged into `config` for `alns` (for example
//! `{"max_iterations":300,"seed":1}` or `{"time_limit_ms":2000}`). `bks.csv` has a header and lines
//! `instance,mark,vehicles,distance,…`; the gap columns are against it.
//!
//! Mapping: the depot's due date is every vehicle's `max_duration`; the fleet is the instance's
//! vehicle count at its capacity; a route costs 1000 in distance (`fixed_cost`), so that fewer routes
//! rank first, as the benchmark's hierarchical objective has it; a pickup carries `delivery`, the
//! delivery's demand is 0.

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
                .collect()
        })
        .filter(|r: &Vec<f64>| !r.is_empty())
        .collect();
    let head = rows.first()?;
    let (vehicles, capacity) = (*head.first()? as usize, *head.get(1)?);
    let nodes = rows.get(1..)?;
    let depot_row = nodes.first()?;
    let horizon = depot_row.get(5).copied()?;
    let mut customers = Vec::with_capacity(nodes.len());
    for r in nodes.iter().skip(1) {
        let (demand, ready, due, service) = (r[3], r[4], r[5], r[6]);
        let mut c = json!({
            "id": r[0] as u64, "x": r[1], "y": r[2], "demand": demand.max(0.0),
            "service_time": service, "time_window": [ready, due],
        });
        if demand > 0.0 {
            c["delivery"] = json!(r[8] as u64);
        }
        customers.push(c);
    }
    Some(Instance {
        name: name.to_string(),
        vehicles,
        capacity,
        depot: (depot_row[1], depot_row[2]),
        horizon,
        customers,
    })
}

fn call(
    entry: unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char) -> i32,
    request: &Value,
) -> (i32, Value) {
    let text = CString::new(request.to_string()).expect("no NUL in JSON");
    let mut out: *mut libc::c_char = std::ptr::null_mut();
    let status = unsafe { entry(text.as_ptr(), &mut out) };
    let answer = if out.is_null() {
        Value::Null
    } else {
        let s = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        unsafe { urouting_free_string(out) };
        serde_json::from_str(&s).unwrap_or(Value::Null)
    };
    (status, answer)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: pdptw_service <instances> [bks.csv] [methods] [filter] [settings]");
    let bks_path = args.next().filter(|p| !p.is_empty());
    let methods: Vec<String> = args
        .next()
        .map(|m| m.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| vec!["nn".into(), "savings".into(), "alns".into()]);
    let only = args.next().filter(|p| !p.is_empty());
    let settings: Value = args
        .next()
        .map(|s| serde_json::from_str(&s).expect("settings are JSON"))
        .unwrap_or_else(|| json!({}));

    let mut bks: HashMap<String, (usize, f64)> = HashMap::new();
    if let Some(path) = bks_path {
        for line in std::fs::read_to_string(path)
            .expect("best known file")
            .lines()
            .skip(1)
        {
            let f: Vec<&str> = line.split(',').collect();
            if let (Some(name), Some(v), Some(d)) = (f.first(), f.get(2), f.get(3)) {
                if let (Ok(v), Ok(d)) = (v.parse(), d.parse()) {
                    bks.insert(name.to_string(), (v, d));
                }
            }
        }
    }

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("instance directory")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();

    println!("instance,method,routes,distance,unassigned,violations,ms,bks_vehicles,bks_distance,gap_pct");
    for file in files {
        let name = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        if only
            .as_deref()
            .is_some_and(|prefix| !name.starts_with(prefix))
        {
            continue;
        }
        let Some(inst) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| parse(&name, &t))
        else {
            eprintln!("{name}: could not read");
            continue;
        };
        for method in &methods {
            let vehicles = json!([{
                "capacity": inst.capacity, "count": inst.vehicles,
                "max_duration": inst.horizon, "fixed_cost": 1000.0,
            }]);
            let config = if method == "alns" {
                settings.clone()
            } else {
                json!({})
            };
            let request = json!({
                "depots": [{ "x": inst.depot.0, "y": inst.depot.1 }],
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
            let (bv, bd, gap) = match bks.get(&inst.name) {
                Some(&(v, d)) => (
                    v.to_string(),
                    format!("{d:.2}"),
                    format!("{:.1}", 100.0 * (distance - d) / d),
                ),
                None => (String::new(), String::new(), String::new()),
            };
            println!(
                "{},{method},{routes},{distance:.2},{unassigned},{violations},{ms},{bv},{bd},{gap}",
                inst.name
            );
        }
    }
}
