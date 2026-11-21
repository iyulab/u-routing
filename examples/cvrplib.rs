//! Measures a search method on capacitated instances in the CVRPLIB format
//! (Uchoa et al.'s X set: `X-n101-k25.vrp` …), with the distances the set
//! defines -- Euclidean, rounded to the nearest integer -- so the cost is
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
//! `urouting_solve_vrp`, the rounded distances as `distances` and as many
//! vehicles as customers (the set fixes no fleet), then
//! `urouting_check_solution` on the answer. `generations` sets
//! `config.max_generations` (`ga`) or `config.max_iterations` (`alns`); with
//! a time budget below, give it a count large enough that the time stops the
//! run first (the `stop_reason` column shows what did). `seeds` defaults to `1`
//! (a comma list runs each). The cost is the answer's `total_distance`.
//!
//! Environment:
//! - `CVRPLIB_METHOD`: `ga` by default; `alns`; the one-pass methods
//!   (`savings` …) read neither a count nor a seed.
//! - `CVRPLIB_BUDGET_FACTOR` (`f`): the literature's time budget scaled by `f`
//!   -- `n × 240/100` seconds × `f` for `n` customers (Vidal 2022; Wouda et al.
//!   2024 use `f = 1`). Reducing `f` is a deviation from the protocol the
//!   published gaps were measured under; report it with the result.
//! - `CVRPLIB_TIME_MS`: one fixed `config.time_limit_ms` for every instance
//!   (ignored when a budget factor is given).
//! - `CVRPLIB_SUBSET` (`m`): a stratified subset of `m` instances -- the list
//!   sorted by customer count (ties by name), ranks `(k × (len − 1) + (m − 1) / 2) ÷ (m − 1)`
//!   for `k = 0 … m − 1`, so the smallest and the largest are always in.
//! - `CVRPLIB_ONLY`: comma list of name prefixes (`X-n502,X-n1001`), applied
//!   before the subset.
//!
//! The output is CSV (`instance,n,method,seed,routes,cost,best,gap_pct,violations,unassigned,ms,budget_ms,stop_reason`)
//! preceded by `# key=value` provenance lines, and followed by `# summary` lines:
//! the mean gap (mean over instances of the gap of the seed-mean cost -- the
//! literature's "mean gap"), the gap of the mean, by size stratum, with the
//! number of failed runs (violations or unassigned customers; left out of the
//! gaps) and of runs that overran the budget by more than 10 %.

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

/// One instance on disk, read once.
struct Instance {
    name: String,
    customers: usize,
    points: Vec<(f64, f64, i64)>,
    capacity: i64,
    best: Option<f64>,
}

/// One finished run.
struct Run {
    customers: usize,
    best: Option<f64>,
    cost: f64,
    ms: u128,
    budget_ms: Option<u64>,
    valid: bool,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The sample standard deviation (`NaN` below two values).
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

/// The summary lines: per size stratum and overall.
fn summarise(runs: &[(String, Run)]) {
    let failed = runs.iter().filter(|(_, r)| !r.valid).count();
    let overran = runs
        .iter()
        .filter(|(_, r)| r.budget_ms.is_some_and(|b| r.ms as f64 > 1.1 * b as f64))
        .count();
    println!(
        "# summary runs={} failed={failed} overran={overran} (ms > 1.1 x budget)",
        runs.len()
    );
    // Per instance: the seed-mean cost over the valid runs, the best known cost,
    // and the spread of the seeds' gaps.
    let mut names: Vec<&str> = runs.iter().map(|(n, _)| n.as_str()).collect();
    names.dedup();
    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new(); // (n, seed-mean cost, best, seed sd of gap)
    for name in names {
        let valid: Vec<&Run> = runs
            .iter()
            .filter(|(n, r)| n == name && r.valid)
            .map(|(_, r)| r)
            .collect();
        let Some(best) = valid.first().and_then(|r| r.best) else {
            continue;
        };
        let costs: Vec<f64> = valid.iter().map(|r| r.cost).collect();
        let gaps: Vec<f64> = costs.iter().map(|c| 100.0 * (c - best) / best).collect();
        rows.push((valid[0].customers, mean(&costs), best, std_dev(&gaps)));
    }
    type Keep = fn(usize) -> bool;
    let strata: [(&str, Keep); 4] = [
        ("n<300", |n| n < 300),
        ("300<=n<600", |n| (300..600).contains(&n)),
        ("n>=600", |n| n >= 600),
        ("all", |_| true),
    ];
    for (label, keep) in strata {
        let part: Vec<&(usize, f64, f64, f64)> = rows.iter().filter(|r| keep(r.0)).collect();
        if part.is_empty() {
            continue;
        }
        let gaps: Vec<f64> = part.iter().map(|r| 100.0 * (r.1 - r.2) / r.2).collect();
        let costs: f64 = part.iter().map(|r| r.1).sum();
        let bests: f64 = part.iter().map(|r| r.2).sum();
        let sds: Vec<f64> = part.iter().map(|r| r.3).filter(|s| s.is_finite()).collect();
        println!(
            "# summary stratum={label} instances={} mean_gap={:.3} gap_of_mean={:.3} max_instance_gap={:.3} mean_seed_sd={}",
            part.len(),
            mean(&gaps),
            100.0 * (costs / bests - 1.0),
            gaps.iter().cloned().fold(f64::MIN, f64::max),
            if sds.is_empty() {
                "NA".to_string()
            } else {
                format!("{:.3}", mean(&sds))
            },
        );
    }
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

    let method = std::env::var("CVRPLIB_METHOD").unwrap_or_else(|_| "ga".to_string());
    let time_limit_ms: Option<u64> = std::env::var("CVRPLIB_TIME_MS")
        .ok()
        .map(|v| v.parse().expect("a number of milliseconds"));
    let budget_factor: Option<f64> = std::env::var("CVRPLIB_BUDGET_FACTOR")
        .ok()
        .map(|v| v.parse().expect("a budget factor"));
    let subset: Option<usize> = std::env::var("CVRPLIB_SUBSET")
        .ok()
        .map(|v| v.parse().expect("a subset size"));
    let only: Vec<String> = std::env::var("CVRPLIB_ONLY")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();

    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("instance directory is readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vrp"))
        .filter(|p| {
            let stem = p.file_stem().and_then(|n| n.to_str()).unwrap_or("");
            only.is_empty() || only.iter().any(|prefix| stem.starts_with(prefix.as_str()))
        })
        .collect();
    paths.sort();
    let mut instances: Vec<Instance> = paths
        .iter()
        .map(|path| {
            let (points, capacity) = read(path);
            Instance {
                name: path
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .expect("a name")
                    .to_string(),
                customers: points.len() - 1,
                points,
                capacity,
                best: std::fs::read_to_string(path.with_extension("sol"))
                    .ok()
                    .and_then(|s| {
                        s.lines()
                            .find_map(|l| l.trim().strip_prefix("Cost")?.trim().parse().ok())
                    }),
            }
        })
        .collect();
    instances.sort_by(|a, b| (a.customers, &a.name).cmp(&(b.customers, &b.name)));
    if let Some(m) = subset.filter(|&m| m >= 1 && m < instances.len()) {
        let len = instances.len();
        let picked: Vec<usize> = (0..m)
            .map(|k| {
                if m == 1 {
                    0
                } else {
                    (k * (len - 1) + (m - 1) / 2) / (m - 1)
                }
            })
            .collect();
        instances = instances
            .into_iter()
            .enumerate()
            .filter(|(i, _)| picked.contains(i))
            .map(|(_, inst)| inst)
            .collect();
    }

    println!("# harness=cvrplib method={method} seeds={seeds:?} generations={generations:?}");
    println!(
        "# budget_factor={budget_factor:?} time_ms={time_limit_ms:?} subset={subset:?} instances={}",
        instances.len()
    );
    println!(
        "# os={} arch={} cpu={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_string())
    );
    println!("instance,n,method,seed,routes,cost,best,gap_pct,violations,unassigned,ms,budget_ms,stop_reason");
    let mut runs: Vec<(String, Run)> = Vec::new();
    for inst in &instances {
        let name = inst.name.as_str();
        let points = &inst.points;
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
        // No fleet limit: one kind of vehicle, as many of it as the plan needs.
        let vehicles: Vec<Value> = vec![json!({ "capacity": inst.capacity })];
        let budget_ms: Option<u64> = match budget_factor {
            Some(f) => Some(((inst.customers as f64 * 2400.0 * f).round() as u64).max(1)),
            None => time_limit_ms,
        };
        for &seed in &seeds {
            let mut config = json!({ "seed": seed });
            if let Some(g) = generations {
                match method.as_str() {
                    "ga" => config["max_generations"] = json!(g),
                    "alns" => config["max_iterations"] = json!(g),
                    _ => {}
                }
            }
            if let Some(ms) = budget_ms {
                config["time_limit_ms"] = json!(ms);
            }
            if !matches!(method.as_str(), "ga" | "alns") {
                config = json!({});
            }
            let request = json!({
                "depots": [{ "x": points[0].0, "y": points[0].1 }],
                "customers": customers, "vehicles": vehicles, "method": method,
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
            let unassigned = solution["unassigned"]
                .as_array()
                .map_or(usize::MAX, Vec::len);
            let routes = solution["routes"].as_array().map_or(0, Vec::len);
            let cost = solution["total_distance"].as_f64().unwrap_or(f64::NAN);
            let gap = inst.best.map(|b| 100.0 * (cost - b) / b);
            println!(
                "{name},{},{method},{seed},{routes},{cost},{},{},{violations},{unassigned},{ms},{},{}",
                inst.customers,
                inst.best.map_or(String::new(), |b| b.to_string()),
                gap.map_or(String::new(), |g| format!("{g:.2}")),
                budget_ms.map_or(String::new(), |b| b.to_string()),
                solution["stop_reason"].as_str().unwrap_or(""),
            );
            runs.push((
                name.to_string(),
                Run {
                    customers: inst.customers,
                    best: inst.best,
                    cost,
                    ms,
                    budget_ms,
                    valid: violations == 0 && unassigned == 0 && cost.is_finite(),
                },
            ));
        }
    }
    summarise(&runs);
}
