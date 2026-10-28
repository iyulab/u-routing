//! Measures the genetic algorithm on capacitated instances in the CVRPLIB
//! format (Uchoa et al.'s X set: `X-n101-k25.vrp` …), with the distances the
//! set defines -- Euclidean, rounded to the nearest integer -- so the cost is
//! comparable with the best known one.
//!
//! The instances are not in this repository. Put the `.vrp` files, and the
//! `.sol` files holding the best known solutions (a last line `Cost <n>`),
//! into one directory:
//!
//! ```text
//! cargo run --release --example cvrplib -- <dir> [generations] [seeds]
//! ```
//!
//! `generations` defaults to 200 (the service's default), `seeds` to `1`
//! (a comma list runs each). Every answer is checked: each customer served
//! once, no route over capacity.

use std::time::Instant;

use u_metaheur::ga::{GaConfig, GaRunner};
use u_routing::distance::DistanceMatrix;
use u_routing::evaluation::RouteLimits;
use u_routing::ga::{split, RoutingGaProblem};
use u_routing::local_search::{or_opt_improve, route_distance, two_opt_improve};
use u_routing::models::Customer;

/// An instance: the depot first, then the customers, and the capacity.
fn read(path: &std::path::Path) -> (Vec<Customer>, i32) {
    let text = std::fs::read_to_string(path).expect("instance file is readable");
    let mut capacity = 0;
    let mut coords: Vec<(f64, f64)> = Vec::new();
    let mut demands: Vec<i32> = Vec::new();
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
    let customers = coords
        .iter()
        .zip(&demands)
        .enumerate()
        .map(|(i, (&(x, y), &d))| {
            if i == 0 {
                Customer::depot(x, y)
            } else {
                Customer::new(i, x, y, d, 0.0)
            }
        })
        .collect();
    (customers, capacity)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(
        args.first()
            .expect("usage: cvrplib <dir> [generations] [seeds]"),
    );
    let generations: usize = args.get(1).map_or(200, |g| g.parse().expect("a count"));
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

    println!("instance,seed,routes,cost,best,gap_pct,ms");
    let mut gaps = Vec::new();
    for path in &names {
        let name = path.file_stem().and_then(|n| n.to_str()).expect("a name");
        let (customers, capacity) = read(path);
        let n = customers.len();
        let mut data = Vec::with_capacity(n * n);
        for a in &customers {
            for b in &customers {
                data.push(
                    ((a.x() - b.x()).powi(2) + (a.y() - b.y()).powi(2))
                        .sqrt()
                        .round(),
                );
            }
        }
        let dm = DistanceMatrix::from_data(n, data).expect("a square matrix");
        let best: Option<f64> = std::fs::read_to_string(path.with_extension("sol"))
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.trim().strip_prefix("Cost")?.trim().parse().ok())
            });
        for &seed in &seeds {
            let started = Instant::now();
            let problem = RoutingGaProblem::new(customers.clone(), dm.clone(), capacity);
            let config = GaConfig::default()
                .with_population_size(50)
                .with_max_generations(generations)
                .with_parallel(false)
                .with_seed(seed);
            let found = GaRunner::run(&problem, &config).expect("valid settings");
            let routes: Vec<Vec<usize>> =
                split(found.best.customers(), &customers, &dm, capacity, 0.0)
                    .routes
                    .iter()
                    .map(|r| {
                        let (r, _) = two_opt_improve(r, 0, &dm, &customers, &RouteLimits::NONE);
                        or_opt_improve(&r, 0, &dm, &customers, &RouteLimits::NONE).0
                    })
                    .collect();
            let ms = started.elapsed().as_millis();
            let mut served: Vec<usize> = routes.concat();
            served.sort_unstable();
            assert_eq!(
                served,
                (1..n).collect::<Vec<_>>(),
                "{name}: every customer once"
            );
            for r in &routes {
                let load: i32 = r.iter().map(|&c| customers[c].demand()).sum();
                assert!(load <= capacity, "{name}: a route over capacity");
            }
            let cost: f64 = routes.iter().map(|r| route_distance(r, 0, &dm)).sum();
            let gap = best.map(|b| 100.0 * (cost - b) / b);
            if let Some(g) = gap {
                gaps.push(g);
            }
            println!(
                "{name},{seed},{},{cost},{},{},{ms}",
                routes.len(),
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
