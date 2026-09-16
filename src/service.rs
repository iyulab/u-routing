//! The solver entry point shared by the WebAssembly and C bindings.
//!
//! Both bindings take a problem as JSON and return a solution as JSON. What
//! differs between them is only the transport -- a JS object or a C string --
//! and the shape of a couple of top-level fields. The solver dispatch, the
//! defaults and the input checks live here once. They used to live in each
//! binding, and the C copy fell behind: it knew two of the four methods,
//! ignored solver settings, and reported whatever method name it was sent as
//! the one it had used.

use serde::{Deserialize, Serialize};

use crate::alns::destroy::RandomRemoval;
use crate::alns::repair::GreedyInsertion;
use crate::alns::RoutingAlnsProblem;
use crate::constructive::{clarke_wright_savings, nearest_neighbor, nearest_neighbor_tw};
use crate::distance::DistanceMatrix;
use crate::evaluation::has_time_windows;
use crate::ga::RoutingGaProblem;
use crate::ga::{split, split_tw};
use crate::local_search::{or_opt_improve, two_opt_improve};
use crate::models::{Customer, TimeWindow, Vehicle};
use u_metaheur::alns::{AlnsConfig, AlnsRunner};
use u_metaheur::ga::{GaConfig, GaRunner};

// ============================================================================
// Wire types shared by both bindings
// ============================================================================

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputCustomer {
    id: usize,
    x: f64,
    y: f64,
    #[serde(default)]
    demand: f64,
    #[serde(default)]
    service_time: f64,
    /// Optional time window as `[ready, due]`.
    #[serde(default)]
    time_window: Option<[f64; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputVehicle {
    #[serde(default = "default_capacity")]
    capacity: f64,
}

fn default_capacity() -> f64 {
    1e9
}

/// Optional solver configuration.
///
/// Fields are shared across methods; each method uses only the relevant ones.
/// Besides the per-method parameters, it carries limits the solver has to
/// respect whichever method runs -- `max_vehicles` is one.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputConfig {
    // --- GA parameters ---
    /// Population size for GA (default: 50).
    #[serde(default)]
    population_size: Option<usize>,
    /// Maximum generations for GA (default: 200).
    #[serde(default)]
    max_generations: Option<usize>,
    /// Mutation rate for GA in (0, 1] (default: 0.1).
    #[serde(default)]
    mutation_rate: Option<f64>,
    /// Elite ratio for GA in (0, 1] (default: 0.1).
    #[serde(default)]
    elite_ratio: Option<f64>,

    // --- ALNS parameters ---
    /// Maximum iterations for ALNS (default: 500).
    #[serde(default)]
    max_iterations: Option<usize>,

    // --- Shared ---
    /// Random seed for reproducibility.
    #[serde(default)]
    seed: Option<u64>,
    /// The number of routes the plan may use at most.
    ///
    /// The length of `vehicles` is not this number: it says which capacities
    /// exist, and for `"savings"`, `"ga"` and `"alns"` a single-entry list is
    /// how a caller states one capacity for an unbounded fleet. A caller with
    /// a fixed fleet says so here, and a plan that would need more routes is
    /// refused rather than returned as if the fleet could run it.
    #[serde(default)]
    max_vehicles: Option<usize>,
}

/// The method a request names when it names none.
pub(crate) fn default_method() -> String {
    Method::NearestNeighbor.name().to_string()
}

#[derive(Debug, Serialize)]
pub(crate) struct VrpOutput {
    pub(crate) routes: Vec<Vec<usize>>,
    pub(crate) total_distance: f64,
    pub(crate) num_vehicles: usize,
    pub(crate) method_used: String,
    /// Filled in by the binding: how to read a clock differs by target.
    pub(crate) computation_time_ms: f64,
    /// Customers (by input id) that no route serves -- for example when the
    /// fleet cannot carry the total demand. They would otherwise simply be
    /// missing from `routes`, and a partial plan would read as a complete one.
    pub(crate) unassigned: Vec<usize>,
}

// ============================================================================
// Methods
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    NearestNeighbor,
    Savings,
    Genetic,
    Alns,
}

impl Method {
    const ALL: [Method; 4] = [
        Method::NearestNeighbor,
        Method::Savings,
        Method::Genetic,
        Method::Alns,
    ];

    fn name(self) -> &'static str {
        match self {
            Method::NearestNeighbor => "nn",
            Method::Savings => "savings",
            Method::Genetic => "ga",
            Method::Alns => "alns",
        }
    }

    fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|m| m.name() == name)
            .ok_or_else(|| {
                let supported: Vec<String> = Self::ALL
                    .iter()
                    .map(|m| format!("\"{}\"", m.name()))
                    .collect();
                format!(
                    "unknown method '{name}'. Supported: {}",
                    supported.join(", ")
                )
            })
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Solves a capacitated VRP, with time windows where customers carry them.
///
/// `method` is checked before anything else, so a misspelt name is reported
/// even for a problem with no customers.
///
/// # Errors
///
/// An unknown method, a time window that is not a window, a demand or capacity
/// that is not a whole number of units in range, solver settings the chosen
/// method rejects, or a plan that needs more routes than `config.max_vehicles`
/// allows.
pub(crate) fn solve(
    depot: (f64, f64),
    input_customers: &[InputCustomer],
    input_vehicles: &[InputVehicle],
    method: &str,
    config: &InputConfig,
) -> Result<VrpOutput, String> {
    let method = Method::parse(method)?;
    if config.max_vehicles == Some(0) {
        return Err(
            "config.max_vehicles is 0, so no route may exist; give at least 1, \
             or leave it out to accept as many routes as the plan needs"
                .to_string(),
        );
    }
    let (customers, id_map) = build_customers(depot, input_customers)?;
    let vehicles = build_vehicles(input_vehicles)?;
    // Only nearest neighbour assigns routes to particular vehicles. The other
    // methods plan every route with one capacity, so a fleet of mixed
    // capacities is not a problem they can state -- it used to be solved as a
    // fleet of the first vehicle's capacity.
    if method != Method::NearestNeighbor {
        let first = vehicles[0].capacity();
        if let Some(other) = vehicles.iter().find(|v| v.capacity() != first) {
            return Err(format!(
                "method \"{}\" plans every route with one vehicle capacity, but the \
                 fleet has capacities {first} and {}; give every vehicle the same \
                 capacity, or use \"nn\", which reads each vehicle",
                method.name(),
                other.capacity()
            ));
        }
    }

    if customers.len() <= 1 {
        return Ok(VrpOutput {
            routes: vec![],
            total_distance: 0.0,
            num_vehicles: 0,
            method_used: method.name().to_string(),
            computation_time_ms: 0.0,
            unassigned: Vec::new(),
        });
    }

    let dm = DistanceMatrix::from_customers(&customers);
    // The capacity every route of savings, GA and ALNS is planned with.
    let capacity = vehicles[0].capacity();

    let mut output = match method {
        Method::NearestNeighbor => solve_nn(&customers, &dm, &vehicles, &id_map),
        Method::Savings => solve_savings(&customers, &dm, &vehicles, &id_map),
        Method::Genetic => solve_ga(&customers, &dm, capacity, &id_map, config)?,
        Method::Alns => solve_alns(&customers, &dm, capacity, &id_map, config)?,
    };
    // Checked against the number the output reports, so the crate refuses
    // exactly when the caller comparing `num_vehicles` to its own fleet would
    // have. Applied to every method alike: `"nn"` cannot exceed the vehicle
    // list, but it can exceed a smaller `max_vehicles`.
    if let Some(max) = config.max_vehicles {
        if output.num_vehicles > max {
            return Err(format!(
                "method \"{}\" needs {} routes to serve these customers, but \
                 config.max_vehicles is {max}; raise max_vehicles, raise the \
                 vehicle capacity, or leave max_vehicles out to accept the plan",
                method.name(),
                output.num_vehicles
            ));
        }
    }
    // Derived from the routes rather than from each solver's own bookkeeping,
    // so it holds for every method alike.
    let served: std::collections::HashSet<usize> =
        output.routes.iter().flatten().copied().collect();
    output.unassigned = id_map
        .iter()
        .copied()
        .filter(|id| !served.contains(id))
        .collect();
    Ok(output)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Builds the internal customer list and ID mapping from input.
///
/// Returns `(customers, id_map)` where `customers[0]` is the depot and
/// `id_map[i]` is the original customer ID for internal index `i+1`.
///
/// A time window the model cannot represent is an error. It used to be
/// dropped, which solved the problem without the constraint the caller had
/// asked for and reported that as success.
fn build_customers(
    depot: (f64, f64),
    input_customers: &[InputCustomer],
) -> Result<(Vec<Customer>, Vec<usize>), String> {
    let mut customers: Vec<Customer> = Vec::with_capacity(input_customers.len() + 1);
    customers.push(Customer::depot(depot.0, depot.1));

    let mut id_map: Vec<usize> = Vec::with_capacity(input_customers.len());

    for ic in input_customers {
        let demand = whole_units(ic.demand, || format!("customer {}: demand", ic.id))?;
        let idx = customers.len();
        id_map.push(ic.id);
        let mut c = Customer::new(idx, ic.x, ic.y, demand, ic.service_time);
        if let Some([ready, due]) = ic.time_window {
            let tw = TimeWindow::new(ready, due).ok_or_else(|| {
                format!(
                    "customer {}: time_window [{ready}, {due}] is not a window \
                     (it needs ready <= due)",
                    ic.id
                )
            })?;
            c = c.with_time_window(tw);
        }
        customers.push(c);
    }

    Ok((customers, id_map))
}

/// Converts internal route indices back to original customer IDs.
fn map_routes(routes: &[Vec<usize>], id_map: &[usize]) -> Vec<Vec<usize>> {
    routes
        .iter()
        .map(|route| {
            route
                .iter()
                .map(|&internal_idx| id_map[internal_idx - 1])
                .collect()
        })
        .collect()
}

/// Applies intra-route 2-opt + or-opt local search to improve routes. With
/// time windows the moves only reorder a route in ways that keep every
/// customer on time.
fn apply_local_search(
    routes: &[Vec<usize>],
    dm: &DistanceMatrix,
    customers: &[Customer],
) -> (Vec<Vec<usize>>, f64) {
    let mut improved_routes = Vec::with_capacity(routes.len());
    let mut total = 0.0;
    for route in routes {
        let (r1, _) = two_opt_improve(route, 0, dm, customers);
        let (r2, dist) = or_opt_improve(&r1, 0, dm, customers);
        total += dist;
        improved_routes.push(r2);
    }
    (improved_routes, total)
}

/// A demand or capacity as the whole number of units the model stores.
///
/// The model counts in `i32` and the wire carries JSON numbers. A value the
/// model would have to round, or clamp into range, is refused: solving a
/// demand of `2.4` as `2` answers a different problem from the one asked, and
/// reports it as solved.
fn whole_units(value: f64, what: impl FnOnce() -> String) -> Result<i32, String> {
    if value.fract() == 0.0 && (0.0..=f64::from(i32::MAX)).contains(&value) {
        Ok(value as i32)
    } else {
        Err(format!(
            "{} is {value}; it must be a whole number of units from 0 to {} -- \
             scale the unit (kilograms to grams, say) to keep a fractional amount",
            what(),
            i32::MAX
        ))
    }
}

/// Builds the vehicle list from input, falling back to a single unlimited vehicle.
fn build_vehicles(input_vehicles: &[InputVehicle]) -> Result<Vec<Vehicle>, String> {
    if input_vehicles.is_empty() {
        return Ok(vec![Vehicle::new(0, i32::MAX)]);
    }
    input_vehicles
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let capacity = whole_units(v.capacity, || format!("vehicle {i}: capacity"))?;
            Ok(Vehicle::new(i, capacity))
        })
        .collect()
}

// ============================================================================
// Solver methods
// ============================================================================

fn solve_nn(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicles: &[Vehicle],
    id_map: &[usize],
) -> VrpOutput {
    let solution = if has_time_windows(customers) {
        nearest_neighbor_tw(customers, dm, vehicles)
    } else {
        nearest_neighbor(customers, dm, vehicles)
    };
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    let mapped = map_routes(&routes, id_map);
    VrpOutput {
        total_distance: solution.total_distance(),
        num_vehicles: mapped.len(),
        routes: mapped,
        method_used: Method::NearestNeighbor.name().to_string(),
        computation_time_ms: 0.0,
        unassigned: Vec::new(),
    }
}

fn solve_savings(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicles: &[Vehicle],
    id_map: &[usize],
) -> VrpOutput {
    let vehicle_template = &vehicles[0];
    let solution = clarke_wright_savings(customers, dm, vehicle_template);
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    let mapped = map_routes(&routes, id_map);
    VrpOutput {
        total_distance: solution.total_distance(),
        num_vehicles: mapped.len(),
        routes: mapped,
        method_used: Method::Savings.name().to_string(),
        computation_time_ms: 0.0,
        unassigned: Vec::new(),
    }
}

fn solve_ga(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    id_map: &[usize],
    cfg: &InputConfig,
) -> Result<VrpOutput, String> {
    let problem = RoutingGaProblem::new(customers.to_vec(), dm.clone(), capacity);

    // Sequential on every target: rayon is unavailable in WebAssembly, and a
    // seed should reproduce the same routes whichever binding runs it.
    let mut ga_config = GaConfig::default()
        .with_population_size(cfg.population_size.unwrap_or(50))
        .with_max_generations(cfg.max_generations.unwrap_or(200))
        .with_parallel(false);

    if let Some(mr) = cfg.mutation_rate {
        ga_config = ga_config.with_mutation_rate(mr);
    }
    if let Some(er) = cfg.elite_ratio {
        ga_config = ga_config.with_elite_ratio(er);
    }
    if let Some(seed) = cfg.seed {
        ga_config = ga_config.with_seed(seed);
    }

    ga_config
        .validate()
        .map_err(|e| format!("GA config error: {}", e))?;

    let ga_result =
        GaRunner::run(&problem, &ga_config).map_err(|e| format!("GA execution error: {}", e))?;

    // Split the best individual into routes the way its fitness was computed.
    // 2-opt and or-opt reorder a route without regard to time, so a problem
    // with windows keeps the split routes as they are.
    // 2-opt and or-opt only take moves that keep every customer on time, so
    // the split routes can be polished either way.
    let tour = ga_result.best.customers();
    let routes = if has_time_windows(customers) {
        split_tw(tour, customers, dm, capacity).routes
    } else {
        split(tour, customers, dm, capacity).routes
    };
    let (improved_routes, total_distance) = apply_local_search(&routes, dm, customers);
    let mapped = map_routes(&improved_routes, id_map);

    Ok(VrpOutput {
        num_vehicles: mapped.len(),
        total_distance,
        routes: mapped,
        method_used: Method::Genetic.name().to_string(),
        computation_time_ms: 0.0,
        unassigned: Vec::new(),
    })
}

fn solve_alns(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    id_map: &[usize],
    cfg: &InputConfig,
) -> Result<VrpOutput, String> {
    let problem = RoutingAlnsProblem::new(customers.to_vec(), dm.clone(), capacity);

    let destroy_ops = vec![RandomRemoval];
    let repair_ops = vec![GreedyInsertion::new(
        dm.clone(),
        customers.to_vec(),
        capacity,
    )];

    let mut alns_config =
        AlnsConfig::default().with_max_iterations(cfg.max_iterations.unwrap_or(500));

    if let Some(seed) = cfg.seed {
        alns_config = alns_config.with_seed(seed);
    }

    alns_config
        .validate()
        .map_err(|e| format!("ALNS config error: {}", e))?;

    let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &alns_config)
        .map_err(|e| format!("ALNS execution error: {}", e))?;

    // Apply local search to improve ALNS result
    let alns_routes: Vec<Vec<usize>> = result.best.routes().to_vec();
    let (improved_routes, total_distance) = apply_local_search(&alns_routes, dm, customers);
    let mapped = map_routes(&improved_routes, id_map);

    Ok(VrpOutput {
        num_vehicles: mapped.len(),
        total_distance,
        routes: mapped,
        method_used: Method::Alns.name().to_string(),
        computation_time_ms: 0.0,
        unassigned: Vec::new(),
    })
}

// ============================================================================
// Tests (native — exercise the solver functions directly)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMatrix;
    use crate::models::Customer;

    /// Helper: build a small test problem with N customers around the origin.
    fn test_customers(n: usize) -> (Vec<Customer>, DistanceMatrix, Vec<usize>) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        let mut id_map = Vec::new();
        for i in 1..=n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            customers.push(Customer::new(
                i,
                angle.cos() * 10.0,
                angle.sin() * 10.0,
                5,
                0.0,
            ));
            id_map.push(i);
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm, id_map)
    }

    fn customer(json: serde_json::Value) -> InputCustomer {
        serde_json::from_value(json).expect("valid customer")
    }

    fn vehicle(json: serde_json::Value) -> InputVehicle {
        serde_json::from_value(json).expect("valid vehicle")
    }

    // ---- entry point ----

    #[test]
    fn an_unknown_method_is_rejected_even_with_no_customers() {
        let err = solve((0.0, 0.0), &[], &[], "tabu", &InputConfig::default())
            .expect_err("unknown method");
        assert!(err.contains("tabu") && err.contains("\"alns\""), "{err}");
    }

    #[test]
    fn no_customers_is_an_empty_solution() {
        let out = solve((0.0, 0.0), &[], &[], "ga", &InputConfig::default()).expect("empty");
        assert!(out.routes.is_empty());
        assert_eq!(out.method_used, "ga");
    }

    #[test]
    fn an_inverted_time_window_names_its_customer() {
        let customers = [customer(serde_json::json!({
            "id": 42, "x": 1.0, "y": 1.0, "time_window": [9.0, 3.0]
        }))];
        let err = solve((0.0, 0.0), &customers, &[], "nn", &InputConfig::default())
            .expect_err("inverted window");
        assert!(err.contains("customer 42"), "{err}");
    }

    #[test]
    fn a_valid_time_window_is_kept() {
        let customers = [customer(serde_json::json!({
            "id": 1, "x": 1.0, "y": 1.0, "time_window": [0.0, 100.0]
        }))];
        let out =
            solve((0.0, 0.0), &customers, &[], "nn", &InputConfig::default()).expect("solvable");
        assert_eq!(out.routes, vec![vec![1]]);
    }

    /// 0.3.3 rounded a demand of 2.4 to 2 and solved that problem instead.
    #[test]
    fn a_fractional_demand_is_refused_not_rounded() {
        let customers = [customer(serde_json::json!({
            "id": 7, "x": 1.0, "y": 1.0, "demand": 2.4
        }))];
        let err = solve((0.0, 0.0), &customers, &[], "nn", &InputConfig::default())
            .expect_err("fractional demand");
        assert!(err.contains("customer 7: demand is 2.4"), "{err}");
    }

    #[test]
    fn demand_and_capacity_must_be_whole_units_in_range() {
        for bad in [-1.0, 0.5, 3e9] {
            let customers = [customer(serde_json::json!({
                "id": 1, "x": 1.0, "y": 1.0, "demand": bad
            }))];
            let err = solve((0.0, 0.0), &customers, &[], "nn", &InputConfig::default())
                .expect_err("demand out of whole units");
            assert!(err.contains("customer 1: demand"), "{bad}: {err}");

            // Checked before the empty-problem shortcut, so a fleet is
            // validated even with nobody to serve.
            let vehicles = [vehicle(serde_json::json!({ "capacity": bad }))];
            let err = solve((0.0, 0.0), &[], &vehicles, "nn", &InputConfig::default())
                .expect_err("capacity out of whole units");
            assert!(err.contains("vehicle 0: capacity"), "{bad}: {err}");
        }

        let customers = [customer(serde_json::json!({
            "id": 1, "x": 1.0, "y": 1.0, "demand": 0.0
        }))];
        let vehicles = [vehicle(serde_json::json!({ "capacity": 10.0 }))];
        assert!(solve(
            (0.0, 0.0),
            &customers,
            &vehicles,
            "nn",
            &InputConfig::default()
        )
        .is_ok());
    }

    fn fleet(capacities: &[f64]) -> Vec<InputVehicle> {
        capacities
            .iter()
            .map(|c| vehicle(serde_json::json!({ "capacity": c })))
            .collect()
    }

    fn ring(n: usize) -> Vec<InputCustomer> {
        (1..=n)
            .map(|i| {
                let a = i as f64;
                customer(serde_json::json!({
                    "id": i, "x": a.cos() * 10.0, "y": a.sin() * 10.0, "demand": 5.0
                }))
            })
            .collect()
    }

    fn quick() -> InputConfig {
        InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            max_iterations: Some(20),
            seed: Some(1),
            ..InputConfig::default()
        }
    }

    // ---- max_vehicles ----

    /// Negative control for the whole group below: with no `max_vehicles`,
    /// a single-entry `vehicles` list still means "one capacity", not "one
    /// vehicle". That is what the README documents and what its own example
    /// relies on, so the fix must not quietly turn the list length into a cap.
    #[test]
    fn a_single_capacity_entry_still_plans_as_many_routes_as_demand_needs() {
        let customers = ring(8);
        let one_capacity = fleet(&[10.0]);
        for method in ["savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &one_capacity, method, &quick()).expect(method);
            assert!(
                out.num_vehicles > one_capacity.len(),
                "{method}: expected more routes than the list length, got {}",
                out.num_vehicles
            );
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
        }
    }

    /// A caller that states its fleet gets a refusal naming both numbers,
    /// in the shape the mixed-capacity refusal already uses.
    #[test]
    fn a_plan_needing_more_routes_than_max_vehicles_is_refused() {
        let customers = ring(8);
        let one_capacity = fleet(&[10.0]);
        let limited = InputConfig {
            max_vehicles: Some(2),
            ..quick()
        };
        for method in ["savings", "ga", "alns"] {
            let err =
                solve((0.0, 0.0), &customers, &one_capacity, method, &limited).expect_err(method);
            assert!(
                err.contains("max_vehicles is 2") && err.contains(method),
                "{method}: {err}"
            );
        }
    }

    /// The number in the refusal is the number the output would have
    /// reported, so a caller comparing `num_vehicles` to its own fleet and
    /// the crate refusing never disagree.
    #[test]
    fn the_refusal_names_the_route_count_the_output_would_have_reported() {
        let customers = ring(8);
        let one_capacity = fleet(&[10.0]);
        for method in ["savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &one_capacity, method, &quick()).expect(method);
            let needed = out.num_vehicles;
            let limited = InputConfig {
                max_vehicles: Some(needed - 1),
                ..quick()
            };
            let err =
                solve((0.0, 0.0), &customers, &one_capacity, method, &limited).expect_err(method);
            assert!(
                err.contains(&format!("needs {needed} routes")),
                "{method}: expected the refusal to name {needed}, got: {err}"
            );
        }
    }

    /// A limit the plan fits is not a refusal -- the constraint is "at most",
    /// not "exactly".
    #[test]
    fn a_max_vehicles_the_plan_fits_is_accepted() {
        let customers = ring(8);
        let one_capacity = fleet(&[10.0]);
        let roomy = InputConfig {
            max_vehicles: Some(8),
            ..quick()
        };
        for method in ["savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &one_capacity, method, &roomy).expect(method);
            assert!(out.num_vehicles <= 8, "{method}: {}", out.num_vehicles);
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
        }
    }

    /// `"nn"` cannot exceed its vehicle list, but it can exceed a smaller
    /// `max_vehicles` -- so the limit is read by every method alike rather
    /// than being a three-method special case.
    #[test]
    fn max_vehicles_is_read_by_nearest_neighbour_too() {
        let customers = ring(8);
        let four = fleet(&[10.0, 10.0, 10.0, 10.0]);
        let unlimited = solve((0.0, 0.0), &customers, &four, "nn", &quick()).expect("nn");
        assert!(unlimited.num_vehicles > 1, "{}", unlimited.num_vehicles);

        let limited = InputConfig {
            max_vehicles: Some(1),
            ..quick()
        };
        let err =
            solve((0.0, 0.0), &customers, &four, "nn", &limited).expect_err("nn over the limit");
        assert!(
            err.contains("max_vehicles is 1") && err.contains("\"nn\""),
            "{err}"
        );
    }

    /// Zero routes cannot serve a customer, so it is refused where it is
    /// stated rather than turning every solve into a route-count failure.
    #[test]
    fn a_max_vehicles_of_zero_is_refused_as_stated() {
        let zero = InputConfig {
            max_vehicles: Some(0),
            ..quick()
        };
        let err = solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), "ga", &zero).expect_err("zero");
        assert!(err.contains("max_vehicles is 0"), "{err}");
    }

    /// Only nearest neighbour reads each vehicle. The other methods plan with
    /// one capacity, and took the first vehicle's for the whole fleet: a
    /// fleet of 10 and 100 was solved as two vehicles of 10.
    #[test]
    fn methods_that_plan_one_capacity_refuse_a_mixed_fleet() {
        let customers = ring(4);
        let mixed = fleet(&[10.0, 100.0]);
        for method in ["savings", "ga", "alns"] {
            let err = solve((0.0, 0.0), &customers, &mixed, method, &quick()).expect_err(method);
            assert!(
                err.contains(&format!("\"{method}\"")) && err.contains("capacit"),
                "{method}: {err}"
            );
        }
        assert!(solve((0.0, 0.0), &customers, &mixed, "nn", &quick()).is_ok());

        let uniform = fleet(&[20.0, 20.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            assert!(
                solve((0.0, 0.0), &customers, &uniform, method, &quick()).is_ok(),
                "{method}"
            );
        }
    }

    /// Two customers beside the depot whose windows cannot share a route:
    /// serving customer 1 lasts until t = 6, and customer 2 closes at t = 3.
    fn clashing_windows() -> Vec<InputCustomer> {
        vec![
            customer(serde_json::json!({
                "id": 1, "x": 1.0, "y": 0.0, "service_time": 5.0, "time_window": [0.0, 2.0]
            })),
            customer(serde_json::json!({
                "id": 2, "x": 0.0, "y": 1.0, "service_time": 5.0, "time_window": [0.0, 3.0]
            })),
        ]
    }

    /// Customers reached after their window closed, driving each route from
    /// the depot at time 0 with travel time equal to distance.
    fn late_arrivals(out: &VrpOutput, customers: &[InputCustomer]) -> Vec<usize> {
        let by_id = |id: usize| customers.iter().find(|c| c.id == id).expect("known id");
        let mut late = Vec::new();
        for route in &out.routes {
            let (mut x, mut y, mut t) = (0.0_f64, 0.0_f64, 0.0_f64);
            for &id in route {
                let c = by_id(id);
                t += ((c.x - x).powi(2) + (c.y - y).powi(2)).sqrt();
                if let Some([ready, due]) = c.time_window {
                    if t > due + 1e-9 {
                        late.push(id);
                    }
                    t = t.max(ready);
                }
                t += c.service_time;
                (x, y) = (c.x, c.y);
            }
        }
        late
    }

    #[test]
    fn every_method_keeps_time_windows() {
        let customers = clashing_windows();
        let two = fleet(&[100.0, 100.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &two, method, &quick()).expect(method);
            assert!(
                late_arrivals(&out, &customers).is_empty(),
                "{method}: {:?}",
                out.routes
            );
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
        }
    }

    /// Eight customers on a ring whose windows open in ring order but close
    /// tightly, so the shortest tour (the ring) is the only one that is on
    /// time and every method has to find it with its windows, not against
    /// them. The local search then has plenty of tempting reversals that
    /// would shorten nothing and make someone late.
    fn ordered_ring() -> Vec<InputCustomer> {
        (1..=8)
            .map(|i| {
                let angle = std::f64::consts::TAU * (i - 1) as f64 / 8.0;
                // Walking the ring, customer i is reached after about i - 1
                // chords of length 2·sin(π/8) ≈ 0.765 plus the radius.
                let due = 1.0 + 0.766 * (i - 1) as f64 + 0.3;
                customer(serde_json::json!({
                    "id": i, "x": angle.cos(), "y": angle.sin(), "service_time": 0.0,
                    "time_window": [0.0, due]
                }))
            })
            .collect()
    }

    #[test]
    fn every_method_keeps_windows_on_a_ring_that_tempts_reversals() {
        let customers = ordered_ring();
        let fleet = fleet(&[100.0, 100.0, 100.0, 100.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &fleet, method, &quick()).expect(method);
            assert!(
                late_arrivals(&out, &customers).is_empty(),
                "{method}: {:?}",
                out.routes
            );
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
        }
    }

    /// A customer no route can reach on time is reported unassigned rather
    /// than served late, by every method.
    #[test]
    fn an_unreachable_window_is_reported_unassigned_not_served_late() {
        let mut customers = clashing_windows();
        customers.push(customer(serde_json::json!({
            "id": 3, "x": 10.0, "y": 0.0, "service_time": 0.0, "time_window": [0.0, 5.0]
        })));
        let fleet = fleet(&[100.0, 100.0, 100.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &fleet, method, &quick()).expect(method);
            assert!(
                late_arrivals(&out, &customers).is_empty(),
                "{method}: {:?}",
                out.routes
            );
            assert_eq!(out.unassigned, vec![3], "{method}: {:?}", out.routes);
        }
    }

    /// Each refusal is read verbatim by a caller, so a line continuation that
    /// lost its backslash shows up as a run of spaces mid-sentence.
    #[test]
    fn refusals_read_as_one_sentence() {
        let fraction = [customer(serde_json::json!({
            "id": 1, "x": 1.0, "y": 0.0, "demand": 0.5
        }))];
        let over_fleet = InputConfig {
            max_vehicles: Some(1),
            ..quick()
        };
        let no_fleet = InputConfig {
            max_vehicles: Some(0),
            ..quick()
        };
        let errors = [
            solve((0.0, 0.0), &ring(2), &fleet(&[10.0, 100.0]), "ga", &quick())
                .expect_err("mixed fleet"),
            solve((0.0, 0.0), &fraction, &[], "nn", &quick()).expect_err("fractional demand"),
            solve((0.0, 0.0), &ring(8), &fleet(&[10.0]), "ga", &over_fleet)
                .expect_err("over max_vehicles"),
            solve((0.0, 0.0), &ring(8), &fleet(&[10.0]), "ga", &no_fleet)
                .expect_err("zero max_vehicles"),
        ];
        for e in errors {
            assert!(!e.contains("  "), "{e}");
        }
    }

    // ---- GA: valid minimal input ----

    #[test]
    fn ga_valid_minimal() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method_used, "ga");
        assert!(!output.routes.is_empty());
        assert!(output.total_distance > 0.0);
    }

    // ---- GA: population_size too small ----

    #[test]
    fn ga_population_size_too_small() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(1),
            max_generations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "population_size=1 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.contains("population_size"),
            "error should mention population_size: {}",
            err
        );
    }

    #[test]
    fn ga_population_size_zero() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(0),
            max_generations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "population_size=0 should fail validation");
    }

    // ---- GA: max_generations zero ----

    #[test]
    fn ga_zero_generations() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "max_generations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.contains("max_generations"),
            "error should mention max_generations: {}",
            err
        );
    }

    // ---- GA: elite_ratio too high ----

    #[test]
    fn ga_elite_ratio_fills_population() {
        let (customers, dm, id_map) = test_customers(3);
        // elite_ratio is clamped to 1.0, so with pop=2 all are elite → validation error
        let cfg = InputConfig {
            population_size: Some(2),
            max_generations: Some(5),
            elite_ratio: Some(1.5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(
            result.is_err(),
            "elite_ratio filling entire population should fail"
        );
    }

    // ---- GA: single customer ----

    #[test]
    fn ga_single_customer() {
        let (customers, dm, id_map) = test_customers(1);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with 1 customer should succeed");
        let output = result.unwrap();
        assert_eq!(output.routes.len(), 1);
    }

    // ---- GA: mutation_rate clamped (not an error, just verifies no panic) ----

    #[test]
    fn ga_extreme_mutation_rate() {
        let (customers, dm, id_map) = test_customers(3);
        // mutation_rate > 1.0 is clamped by GaConfig::with_mutation_rate
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            mutation_rate: Some(5.0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(
            result.is_ok(),
            "clamped mutation_rate should not cause error"
        );
    }

    // ---- ALNS: valid minimal input ----

    #[test]
    fn alns_valid_minimal() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            max_iterations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_alns(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "ALNS with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method_used, "alns");
        assert!(!output.routes.is_empty());
    }

    // ---- ALNS: zero iterations ----

    #[test]
    fn alns_zero_iterations() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig {
            max_iterations: Some(0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_alns(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_err(), "max_iterations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.contains("max_iterations"),
            "error should mention max_iterations: {}",
            err
        );
    }

    // ---- Default config (no config provided) ----

    #[test]
    fn ga_default_config() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with default config should succeed");
    }

    #[test]
    fn alns_default_config() {
        let (customers, dm, id_map) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_alns(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "ALNS with default config should succeed");
    }

    // ---- GA: larger problem to stress-test GA operators ----

    #[test]
    fn ga_larger_problem() {
        let (customers, dm, id_map) = test_customers(20);
        let cfg = InputConfig {
            population_size: Some(30),
            max_generations: Some(20),
            seed: Some(123),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 1000, &id_map, &cfg);
        assert!(
            result.is_ok(),
            "GA with 20 customers should succeed: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert!(!output.routes.is_empty());
        assert!(output.total_distance > 0.0);
    }

    // ---- GA: tight capacity forces many routes ----

    #[test]
    fn ga_tight_capacity() {
        let (customers, dm, id_map) = test_customers(10);
        // Each customer has demand=5, capacity=5 forces one customer per route
        let cfg = InputConfig {
            population_size: Some(20),
            max_generations: Some(10),
            seed: Some(99),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 5, &id_map, &cfg);
        assert!(
            result.is_ok(),
            "GA with tight capacity should succeed: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert_eq!(output.routes.len(), 10, "each customer needs its own route");
    }

    // ---- GA: two customers (minimal crossover) ----

    #[test]
    fn ga_two_customers() {
        let (customers, dm, id_map) = test_customers(2);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &id_map, &cfg);
        assert!(result.is_ok(), "GA with 2 customers should succeed");
    }
}

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn nested_structs_reject_unknown_keys() {
        assert_rejects_unknown::<super::InputCustomer>(json!({
            "id": 1, "x": 0.0, "y": 0.0, "priority": 2
        }));
        assert_rejects_unknown::<super::InputVehicle>(json!({
            "capacity": 10.0, "speed": 1.0
        }));
        assert_rejects_unknown::<super::InputConfig>(json!({
            "population_size": 60, "generations": 100
        }));
    }
}
