//! Measures the pickup-delivery constructions on the Li & Lim PDPTW instances through the Rust
//! API: nearest neighbour, savings, savings followed by pair relocation, and an adaptive large
//! neighbourhood search (every route is worth `ROUTE_COST` in distance, so vehicles come first
//! as in the benchmark's objective), each judged by `check_solution`.
//!
//! The instances are not in this repository. Download the 100-task set (SINTEF's "Li & Lim
//! benchmark" page, `pdp_100.zip`) and point the harness at the directory holding `lc101.txt` …:
//!
//! ```text
//! cargo run --release --example pdptw -- <dir> [bks.csv] [methods]
//! ```
//!
//! `bks.csv` (optional) has a header and lines `instance,mark,vehicles,distance,…` -- the best
//! known results under the usual hierarchical objective (vehicles first, then distance); the gap
//! columns are against it. `methods` is a comma list of `nn,savings,savings+pairs,alns` (default
//! the first three; `alns` runs `ALNS_ITERATIONS` iterations, 200 by default). A fourth argument
//! keeps only the instances whose name starts with it.
//!
//! Instance format: one header line (vehicles, capacity, speed), then one row per node (id, x, y,
//! demand, ready, due, service, pickup sibling, delivery sibling); node 0 is the depot, a pickup
//! has the positive demand and names its delivery, a delivery the negative demand. Mapping: the
//! depot's due time is each vehicle's `max_duration`, the fleet is the instance's vehicle count
//! at its capacity, distance and time are Euclidean in double precision, at the stated speed.

use std::collections::HashMap;
use std::time::Instant;

use u_metaheur::alns::{AlnsConfig, AlnsRunner};
use u_routing::alns::destroy::{RandomRemoval, RouteRemoval, ShawRemoval, WorstRemoval};
use u_routing::alns::repair::{GreedyInsertion, RegretInsertion};
use u_routing::alns::{Destroy, Repair, RoutingAlnsProblem};
use u_routing::constructive::{clarke_wright_savings, nearest_neighbor_tw};
use u_routing::distance::DistanceMatrix;
use u_routing::evaluation::{check_solution, Fleet, RouteEvaluator, RouteLimits};
use u_routing::local_search::pair_relocate_improve;
use u_routing::models::{Customer, Solution, TimeWindow, Vehicle};

/// What opening a route is worth in distance: a route outweighs any detour on these maps.
const ROUTE_COST: f64 = 1000.0;

struct Instance {
    name: String,
    customers: Vec<Customer>,
    vehicles: Vec<Vehicle>,
    vehicle: Vehicle,
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
    let (count, capacity) = (*head.first()? as usize, *head.get(1)? as i32);
    let nodes = rows.get(1..)?;
    let horizon = nodes.first()?.get(5).copied()?;
    let mut customers = Vec::with_capacity(nodes.len());
    for r in nodes {
        let id = *r.first()? as usize;
        if id == 0 {
            customers.push(Customer::depot(r[1], r[2]));
            continue;
        }
        let (demand, ready, due, service) = (r[3] as i32, r[4], r[5], r[6]);
        let (pickup_sibling, delivery_sibling) = (r[7] as usize, r[8] as usize);
        let mut c = Customer::new(id, r[1], r[2], demand.max(0), service)
            .with_time_window(TimeWindow::new(ready, due)?);
        if demand > 0 {
            c = c.pickup_of(delivery_sibling);
        } else {
            c = c.delivery_of(pickup_sibling);
        }
        customers.push(c);
    }
    let make = |id: usize| Vehicle::new(id, capacity).with_max_duration(horizon);
    Some(Instance {
        name: name.to_string(),
        customers,
        vehicles: (0..count).map(make).collect(),
        vehicle: make(0),
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: pdptw <instances> [bks.csv] [methods]");
    let bks_path = args.next().filter(|p| !p.is_empty());
    let only = args_filter();
    let methods: Vec<String> = args
        .next()
        .map(|m| m.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| vec!["nn".into(), "savings".into(), "savings+pairs".into()]);

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
    files.retain(|f| {
        f.file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|n| only.as_deref().is_none_or(|prefix| n.starts_with(prefix)))
    });

    println!("instance,method,routes,distance,unassigned,violations,ms,bks_vehicles,bks_distance,gap_pct");
    for file in files {
        let name = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let Some(i) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| parse(&name, &t))
        else {
            eprintln!("{name}: could not read");
            continue;
        };
        let dm = DistanceMatrix::from_customers(&i.customers);
        let fleet = Fleet {
            vehicles: &i.vehicles,
            counts: &[],
            max_routes: None,
        };
        for method in &methods {
            let start = Instant::now();
            let solution: Solution = match method.as_str() {
                "nn" => nearest_neighbor_tw(&i.customers, &dm, &i.vehicles),
                "savings" => clarke_wright_savings(&i.customers, &dm, &i.vehicle),
                "savings+pairs" => {
                    let s = clarke_wright_savings(&i.customers, &dm, &i.vehicle);
                    pair_relocate_improve(&s, &i.customers, &dm, &i.vehicle)
                }
                "alns" => alns(&i, &dm),
                other => panic!("unknown method {other}"),
            };
            let ms = start.elapsed().as_millis();
            let violations = check_solution(&i.customers, &dm, &fleet, &solution).len();
            let (bks_v, bks_d, gap) = match bks.get(&i.name) {
                Some(&(v, d)) => (
                    v.to_string(),
                    format!("{d:.2}"),
                    format!("{:.1}", 100.0 * (solution.total_distance() - d) / d),
                ),
                None => (String::new(), String::new(), String::new()),
            };
            println!(
                "{},{},{},{:.2},{},{},{},{},{},{}",
                i.name,
                method,
                solution.num_routes(),
                solution.total_distance(),
                solution.num_unassigned(),
                violations,
                ms,
                bks_v,
                bks_d,
                gap
            );
        }
    }
}

/// The instance-name prefix given as the fourth argument, if any.
fn args_filter() -> Option<String> {
    std::env::args().nth(4).filter(|p| !p.is_empty())
}

/// An adaptive large neighbourhood search from the problem's own seed, with every destroy and
/// repair operator, the vehicle's capacity and duration limit, and a price per route.
fn alns(i: &Instance, dm: &DistanceMatrix) -> Solution {
    let iterations: usize = std::env::var("ALNS_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let limits = RouteLimits::of(&i.vehicle);
    let capacity = i.vehicle.capacity();
    let problem = RoutingAlnsProblem::new(i.customers.clone(), dm.clone(), capacity)
        .with_route_cost(ROUTE_COST)
        .with_limits(limits);
    let destroy = vec![
        Destroy::Random(RandomRemoval),
        Destroy::Worst(WorstRemoval::new(dm.clone())),
        Destroy::Shaw(ShawRemoval::new(dm.clone(), i.customers.clone())),
        Destroy::Route(RouteRemoval),
    ];
    let repair = vec![
        Repair::Greedy(
            GreedyInsertion::new(dm.clone(), i.customers.clone(), capacity)
                .with_limits(limits)
                .with_route_cost(ROUTE_COST),
        ),
        Repair::Regret(
            RegretInsertion::new(dm.clone(), i.customers.clone(), capacity)
                .with_limits(limits)
                .with_route_cost(ROUTE_COST),
        ),
    ];
    let config = AlnsConfig::default()
        .with_max_iterations(iterations)
        .with_seed(1);
    let best = AlnsRunner::run(&problem, &destroy, &repair, &config)
        .expect("a run")
        .best;
    let evaluator = RouteEvaluator::new(&i.customers, dm, &i.vehicle);
    let mut solution = Solution::new();
    for route in best.routes().iter().filter(|r| !r.is_empty()) {
        solution.add_route(evaluator.build_route(route).0);
    }
    for &u in best.unassigned() {
        solution.add_unassigned(u);
    }
    solution.price(|_| &i.vehicle);
    solution
}
