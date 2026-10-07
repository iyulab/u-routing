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
use serde_json::json;

use crate::alns::destroy::RandomRemoval;
use crate::alns::repair::GreedyInsertion;
use crate::alns::RoutingAlnsProblem;
use crate::constructive::{clarke_wright_savings, nearest_neighbor, nearest_neighbor_tw};
use crate::distance::DistanceMatrix;
use crate::evaluation::has_time_windows;
use crate::fleet::limit_routes;
use crate::ga::RoutingGaProblem;
use crate::ga::{split, split_tw};
use crate::local_search::{or_opt_improve, two_opt_improve};
use crate::models::{Customer, TimeWindow, Vehicle};
use u_metaheur::alns::{AlnsConfig, AlnsRunner};
use u_metaheur::ga::{GaConfig, GaRunner};

// ============================================================================
// Refusals
// ============================================================================

/// A refusal on its way to a caller: readable text for people, and `fields` --
/// `code` first among them -- for programs.
///
/// `code` is a stable name for the reason and the other fields are the values
/// behind it (which customer, which positions, which setting). The message is
/// free to change; a program that branches on it, or pulls a number out of it,
/// breaks when it does. Both bindings carry the same pair: the WebAssembly one
/// copies `fields` onto the thrown `Error`, the C one writes them next to
/// `"error"` in its error body.
#[derive(Debug)]
pub(crate) struct ServiceError {
    pub(crate) message: String,
    pub(crate) fields: serde_json::Value,
}

impl ServiceError {
    fn new(code: &str, message: String, mut fields: serde_json::Value) -> Self {
        let mut all = serde_json::Map::new();
        all.insert("code".into(), json!(code));
        if let Some(extra) = fields.as_object_mut() {
            all.append(extra);
        }
        ServiceError {
            message,
            fields: serde_json::Value::Object(all),
        }
    }

    /// The input is not the shape the solver takes: a wrong type, a missing or
    /// unknown key. `parameter` names the argument.
    pub(crate) fn malformed_input(parameter: &str, message: String) -> Self {
        Self::new(
            "malformed_input",
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A NaN or ±Infinity where a finite number belongs; `fields` carries
    /// `parameter` and `index`. Only JavaScript can send one: JSON has none.
    #[cfg(feature = "wasm")]
    pub(crate) fn value_not_finite(message: String, fields: serde_json::Value) -> Self {
        Self::new("value_not_finite", message, fields)
    }

    /// A failure inside the library rather than a refusal of the input.
    #[cfg(feature = "ffi")]
    pub(crate) fn internal(message: &str) -> Self {
        Self::new("internal", message.to_string(), json!({}))
    }

    /// The stable reason, as the `code` field carries it.
    #[cfg(test)]
    pub(crate) fn code(&self) -> &str {
        self.fields["code"]
            .as_str()
            .expect("every refusal carries a code")
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

// ============================================================================
// Wire types shared by both bindings
// ============================================================================

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputCustomer {
    id: usize,
    x: f64,
    y: f64,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    demand: f64,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    service_time: f64,
    /// Optional time window as `[ready, due]`.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "[number, number] | null"))]
    time_window: Option<[f64; 2]>,
}

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputVehicle {
    #[serde(default = "default_capacity")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
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
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputConfig {
    // --- GA parameters ---
    /// Population size for GA (default: 50).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    population_size: Option<usize>,
    /// Maximum generations for GA (default: 200).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    max_generations: Option<usize>,
    /// Mutation rate for GA in (0, 1] (default: 0.1).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    mutation_rate: Option<f64>,
    /// Elite ratio for GA in (0, 1] (default: 0.1).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    elite_ratio: Option<f64>,

    // --- ALNS parameters ---
    /// Maximum iterations for ALNS (default: 500).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    max_iterations: Option<usize>,

    // --- Shared ---
    /// Random seed for reproducibility.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    seed: Option<u64>,
    /// The number of routes the plan may use at most -- the fixed fleet.
    ///
    /// The length of `vehicles` is not this number: it says which capacities
    /// exist, and for `"savings"`, `"ga"` and `"alns"` a single-entry list is
    /// how a caller states one capacity for an unbounded fleet. A caller with
    /// a fixed fleet says so here, and every method keeps to it: a plan that
    /// would need more routes has its lightest routes emptied into the others
    /// where capacity and time windows allow, and the customers that fit
    /// nowhere are reported in `unassigned`.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    max_vehicles: Option<usize>,
}

/// The method a request names when it names none.
pub(crate) fn default_method() -> String {
    Method::NearestNeighbor.name().to_string()
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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

    fn parse(name: &str) -> Result<Self, ServiceError> {
        Self::ALL
            .into_iter()
            .find(|m| m.name() == name)
            .ok_or_else(|| {
                let expected: Vec<&str> = Self::ALL.iter().map(|m| m.name()).collect();
                let quoted: Vec<String> = expected.iter().map(|m| format!("\"{m}\"")).collect();
                ServiceError::new(
                    "unknown_option",
                    format!("unknown method '{name}'. Supported: {}", quoted.join(", ")),
                    json!({ "parameter": "method", "got": name, "expected": expected }),
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
/// that is not a whole number of units in range, or solver settings the chosen
/// method rejects. Each carries its own `code` (see [`ServiceError`]).
pub(crate) fn solve(
    depot: (f64, f64),
    input_customers: &[InputCustomer],
    input_vehicles: &[InputVehicle],
    method: &str,
    config: &InputConfig,
) -> Result<VrpOutput, ServiceError> {
    let method = Method::parse(method)?;
    if config.max_vehicles == Some(0) {
        return Err(ServiceError::new(
            "invalid_option",
            "config.max_vehicles is 0, so no route may exist; give at least 1, \
             or leave it out to accept as many routes as the plan needs"
                .to_string(),
            json!({ "parameter": "max_vehicles", "value": 0 }),
        ));
    }
    let (customers, id_map) = build_customers(depot, input_customers)?;
    let vehicles = build_vehicles(input_vehicles)?;
    // Only nearest neighbour assigns routes to particular vehicles. The other
    // methods plan every route with one capacity, so a fleet of mixed
    // capacities is not a problem they can state -- it used to be solved as a
    // fleet of the first vehicle's capacity.
    if method != Method::NearestNeighbor {
        let first = vehicles[0].capacity();
        if let Some((index, other)) = vehicles
            .iter()
            .enumerate()
            .find(|(_, v)| v.capacity() != first)
        {
            return Err(ServiceError::new(
                "mixed_fleet",
                format!(
                    "method \"{}\" plans every route with one vehicle capacity, but the \
                     fleet has capacities {first} and {}; give every vehicle the same \
                     capacity, or use \"nn\", which reads each vehicle",
                    method.name(),
                    other.capacity()
                ),
                json!({
                    "method": method.name(),
                    "capacities": [first, other.capacity()],
                    "index": index,
                }),
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

    let plan = match method {
        Method::NearestNeighbor => solve_nn(&customers, &dm, &vehicles),
        Method::Savings => solve_savings(&customers, &dm, &vehicles),
        Method::Genetic => solve_ga(&customers, &dm, capacity, config)?,
        Method::Alns => solve_alns(&customers, &dm, capacity, config)?,
    };
    // A fixed fleet is kept by every method alike: `"nn"` cannot exceed its
    // vehicle list, but it can exceed a smaller `max_vehicles`.
    let plan = match config.max_vehicles {
        Some(max) if plan.routes.len() > max => {
            let limited = limit_routes(plan.routes, plan.capacities, max, &dm, &customers);
            let (routes, total_distance) = apply_local_search(&limited.routes, &dm, &customers);
            Plan {
                routes,
                capacities: limited.capacities,
                total_distance,
                method,
            }
        }
        _ => plan,
    };
    let routes = map_routes(&plan.routes, id_map.as_slice());
    // Derived from the routes rather than from each solver's own bookkeeping,
    // so it holds for every method alike.
    let served: std::collections::HashSet<usize> = routes.iter().flatten().copied().collect();
    let unassigned = id_map
        .iter()
        .copied()
        .filter(|id| !served.contains(id))
        .collect();
    Ok(VrpOutput {
        num_vehicles: routes.len(),
        total_distance: plan.total_distance,
        routes,
        method_used: plan.method.name().to_string(),
        computation_time_ms: 0.0,
        unassigned,
    })
}

/// A method's plan in internal customer indices, before it is mapped to the
/// caller's ids.
#[derive(Debug)]
struct Plan {
    routes: Vec<Vec<usize>>,
    /// The capacity of each route, in the same order.
    capacities: Vec<i32>,
    total_distance: f64,
    method: Method,
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
///
/// So is an `id` given to two customers. The output names customers by `id`
/// alone, so a route through both would read `[1, 1]` and the caller could not
/// tell which point was visited when.
fn build_customers(
    depot: (f64, f64),
    input_customers: &[InputCustomer],
) -> Result<(Vec<Customer>, Vec<usize>), ServiceError> {
    let mut customers: Vec<Customer> = Vec::with_capacity(input_customers.len() + 1);
    customers.push(Customer::depot(depot.0, depot.1));

    let mut id_map: Vec<usize> = Vec::with_capacity(input_customers.len());
    let mut first_at: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(input_customers.len());

    for (position, ic) in input_customers.iter().enumerate() {
        if let Some(first) = first_at.insert(ic.id, position) {
            return Err(ServiceError::new(
                "duplicate_id",
                format!(
                    "customer {}: the id is given twice, at positions {first} and \
                     {position} of customers (counting from 0); routes and unassigned \
                     name customers by id, so every customer needs its own",
                    ic.id
                ),
                json!({ "id": ic.id, "first": first, "second": position }),
            ));
        }
        let demand = whole_units(ic.demand, "demand", position, Some(ic.id), || {
            format!("customer {}: demand", ic.id)
        })?;
        let idx = customers.len();
        id_map.push(ic.id);
        let mut c = Customer::new(idx, ic.x, ic.y, demand, ic.service_time);
        if let Some([ready, due]) = ic.time_window {
            let tw = TimeWindow::new(ready, due).ok_or_else(|| {
                ServiceError::new(
                    "invalid_time_window",
                    format!(
                        "customer {}: time_window [{ready}, {due}] is not a window \
                         (it needs ready <= due)",
                        ic.id
                    ),
                    json!({ "id": ic.id, "index": position, "ready": ready, "due": due }),
                )
            })?;
            c = c.with_time_window(tw);
        }
        customers.push(c);
    }

    Ok((customers, id_map))
}

/// Solver settings the method's runner refused (a population of 1, zero
/// generations or iterations). The runner reports these in its own words, so
/// the refusal carries its text and names the method and the `config` it read.
fn settings_refused(method: Method, e: impl std::fmt::Display) -> ServiceError {
    ServiceError::new(
        "invalid_option",
        format!("method \"{}\": config refused: {e}", method.name()),
        json!({ "parameter": "config", "method": method.name() }),
    )
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
        let (r1, _) = two_opt_improve(
            route,
            0,
            dm,
            customers,
            &crate::evaluation::RouteLimits::NONE,
        );
        let (r2, dist) =
            or_opt_improve(&r1, 0, dm, customers, &crate::evaluation::RouteLimits::NONE);
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
///
/// The refusal names the field (`"demand"` or `"capacity"`), the position of
/// the entry in its list, and the customer's `id` when there is one.
fn whole_units(
    value: f64,
    parameter: &str,
    index: usize,
    id: Option<usize>,
    what: impl FnOnce() -> String,
) -> Result<i32, ServiceError> {
    if value.fract() == 0.0 && (0.0..=f64::from(i32::MAX)).contains(&value) {
        Ok(value as i32)
    } else {
        Err(ServiceError::new(
            "not_whole_units",
            format!(
                "{} is {value}; it must be a whole number of units from 0 to {} -- \
                 scale the unit (kilograms to grams, say) to keep a fractional amount",
                what(),
                i32::MAX
            ),
            // A NaN or an infinity is not a JSON number; it crosses as `null`.
            json!({ "parameter": parameter, "index": index, "id": id, "value": value }),
        ))
    }
}

/// Builds the vehicle list from input, falling back to a single unlimited vehicle.
fn build_vehicles(input_vehicles: &[InputVehicle]) -> Result<Vec<Vehicle>, ServiceError> {
    if input_vehicles.is_empty() {
        return Ok(vec![Vehicle::new(0, i32::MAX)]);
    }
    input_vehicles
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let capacity = whole_units(v.capacity, "capacity", i, None, || {
                format!("vehicle {i}: capacity")
            })?;
            Ok(Vehicle::new(i, capacity))
        })
        .collect()
}

// ============================================================================
// Solver methods
// ============================================================================

fn solve_nn(customers: &[Customer], dm: &DistanceMatrix, vehicles: &[Vehicle]) -> Plan {
    let solution = if has_time_windows(customers) {
        nearest_neighbor_tw(customers, dm, vehicles)
    } else {
        nearest_neighbor(customers, dm, vehicles)
    };
    Plan {
        routes: solution.routes().iter().map(|r| r.customer_ids()).collect(),
        // Nearest neighbour fills each vehicle with its own capacity.
        capacities: solution
            .routes()
            .iter()
            .map(|r| vehicles[r.vehicle_id()].capacity())
            .collect(),
        total_distance: solution.total_distance(),
        method: Method::NearestNeighbor,
    }
}

fn solve_savings(customers: &[Customer], dm: &DistanceMatrix, vehicles: &[Vehicle]) -> Plan {
    let vehicle_template = &vehicles[0];
    let solution = clarke_wright_savings(customers, dm, vehicle_template);
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    Plan {
        capacities: vec![vehicle_template.capacity(); routes.len()],
        routes,
        total_distance: solution.total_distance(),
        method: Method::Savings,
    }
}

fn solve_ga(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    cfg: &InputConfig,
) -> Result<Plan, ServiceError> {
    let problem = RoutingGaProblem::new(customers.to_vec(), dm.clone(), capacity);

    // Sequential on every target: rayon is unavailable in WebAssembly, and a
    // seed should reproduce the same routes whichever binding runs it.
    let mut ga_config = GaConfig::default()
        .with_population_size(cfg.population_size.unwrap_or(50))
        .with_max_generations(cfg.max_generations.unwrap_or(200))
        .with_parallel(false);

    // Both are rates in (0, 1]: a value outside is refused, not clamped.
    for (parameter, value) in [
        ("config.mutation_rate", cfg.mutation_rate),
        ("config.elite_ratio", cfg.elite_ratio),
    ] {
        if let Some(v) = value.filter(|v| !(*v > 0.0 && *v <= 1.0)) {
            return Err(ServiceError::new(
                "parameter_out_of_range",
                format!("{parameter} must be in (0, 1], got {v}"),
                json!({ "parameter": parameter, "min": 0.0, "max": 1.0, "got": v }),
            ));
        }
    }
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
        .map_err(|e| settings_refused(Method::Genetic, e))?;

    let ga_result =
        GaRunner::run(&problem, &ga_config).map_err(|e| settings_refused(Method::Genetic, e))?;

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
    let (routes, total_distance) = apply_local_search(&routes, dm, customers);
    Ok(Plan {
        capacities: vec![capacity; routes.len()],
        routes,
        total_distance,
        method: Method::Genetic,
    })
}

fn solve_alns(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    cfg: &InputConfig,
) -> Result<Plan, ServiceError> {
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
        .map_err(|e| settings_refused(Method::Alns, e))?;

    let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &alns_config)
        .map_err(|e| settings_refused(Method::Alns, e))?;

    // Apply local search to improve ALNS result
    let alns_routes: Vec<Vec<usize>> = result.best.routes().to_vec();
    let (routes, total_distance) = apply_local_search(&alns_routes, dm, customers);
    Ok(Plan {
        capacities: vec![capacity; routes.len()],
        routes,
        total_distance,
        method: Method::Alns,
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
    fn test_customers(n: usize) -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for i in 1..=n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            customers.push(Customer::new(
                i,
                angle.cos() * 10.0,
                angle.sin() * 10.0,
                5,
                0.0,
            ));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
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
        assert!(
            err.message.contains("tabu") && err.message.contains("\"alns\""),
            "{err}"
        );
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
        assert!(err.message.contains("customer 42"), "{err}");
    }

    #[test]
    fn a_repeated_customer_id_is_refused_naming_both_positions() {
        let customers = [
            customer(serde_json::json!({ "id": 1, "x": 1.0, "y": 1.0 })),
            customer(serde_json::json!({ "id": 7, "x": 2.0, "y": 0.0 })),
            customer(serde_json::json!({ "id": 1, "x": 2.0, "y": 2.0 })),
        ];
        for method in ["nn", "savings", "ga", "alns"] {
            let err = solve((0.0, 0.0), &customers, &[], method, &InputConfig::default())
                .expect_err("a repeated id cannot be told apart in the routes");
            assert!(err.message.contains("customer 1"), "{method}: {err}");
            assert!(err.message.contains("0 and 2"), "{method}: {err}");
        }
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
        assert!(err.message.contains("customer 7: demand is 2.4"), "{err}");
    }

    #[test]
    fn demand_and_capacity_must_be_whole_units_in_range() {
        for bad in [-1.0, 0.5, 3e9] {
            let customers = [customer(serde_json::json!({
                "id": 1, "x": 1.0, "y": 1.0, "demand": bad
            }))];
            let err = solve((0.0, 0.0), &customers, &[], "nn", &InputConfig::default())
                .expect_err("demand out of whole units");
            assert!(err.message.contains("customer 1: demand"), "{bad}: {err}");

            // Checked before the empty-problem shortcut, so a fleet is
            // validated even with nobody to serve.
            let vehicles = [vehicle(serde_json::json!({ "capacity": bad }))];
            let err = solve((0.0, 0.0), &[], &vehicles, "nn", &InputConfig::default())
                .expect_err("capacity out of whole units");
            assert!(err.message.contains("vehicle 0: capacity"), "{bad}: {err}");
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

    /// The #301 report, reproduced: six customers of demand 10, two vehicles
    /// of 20. Without a limit, savings, GA and ALNS open a third route.
    fn six_of_ten() -> Vec<InputCustomer> {
        (1..=6)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / 6.0;
                customer(serde_json::json!({
                    "id": i, "x": a.cos() * 10.0, "y": a.sin() * 10.0, "demand": 10.0
                }))
            })
            .collect()
    }

    /// A caller that states its fleet gets a plan the fleet can run: at most
    /// that many routes, and the customers it cannot carry in `unassigned` --
    /// for every method, which is the acceptance test #301 asked for.
    #[test]
    fn every_method_keeps_to_a_fixed_fleet_and_reports_what_does_not_fit() {
        let two = fleet(&[20.0, 20.0]);
        let fixed = InputConfig {
            max_vehicles: Some(2),
            ..quick()
        };
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &six_of_ten(), &two, method, &fixed).expect(method);
            assert!(
                out.num_vehicles <= 2,
                "{method}: {} routes",
                out.num_vehicles
            );
            assert_eq!(out.num_vehicles, out.routes.len());
            assert_eq!(out.unassigned.len(), 2, "{method}: {:?}", out.unassigned);
            let served: usize = out.routes.iter().map(Vec::len).sum();
            assert_eq!(served + out.unassigned.len(), 6, "{method}");
            for route in &out.routes {
                assert!(route.len() * 10 <= 20, "{method}: over capacity {route:?}");
            }
        }
    }

    /// Unlimited, the same request still needs a third route -- so the test
    /// above measures the limit, not a problem that happened to fit.
    #[test]
    fn without_the_limit_the_same_request_opens_more_routes() {
        let two = fleet(&[20.0, 20.0]);
        for method in ["savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &six_of_ten(), &two, method, &quick()).expect(method);
            assert_eq!(out.num_vehicles, 3, "{method}");
            assert!(out.unassigned.is_empty(), "{method}");
        }
    }

    /// Emptying a route moves its customers into spare room elsewhere before
    /// it gives any up.
    #[test]
    fn spare_capacity_is_used_before_a_customer_is_left_out() {
        let roomy = fleet(&[30.0]);
        let fixed = InputConfig {
            max_vehicles: Some(2),
            ..quick()
        };
        for method in ["savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &six_of_ten(), &roomy, method, &fixed).expect(method);
            assert!(out.num_vehicles <= 2, "{method}: {}", out.num_vehicles);
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
        }
    }

    /// A limit the plan fits leaves the plan as it was -- the constraint is
    /// "at most", not "exactly".
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
            let free =
                solve((0.0, 0.0), &customers, &one_capacity, method, &quick()).expect(method);
            assert_eq!(out.routes, free.routes, "{method}");
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
        }
    }

    /// A fixed fleet with time windows: the routes that remain still reach
    /// every customer on time.
    #[test]
    fn a_fixed_fleet_keeps_time_windows() {
        let customers: Vec<InputCustomer> = (1..=6)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / 6.0;
                customer(serde_json::json!({
                    "id": i, "x": a.cos() * 10.0, "y": a.sin() * 10.0, "demand": 1.0,
                    "time_window": [0.0, 25.0]
                }))
            })
            .collect();
        let fixed = InputConfig {
            max_vehicles: Some(2),
            ..quick()
        };
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve(
                (0.0, 0.0),
                &customers,
                &fleet(&[100.0, 100.0]),
                method,
                &fixed,
            )
            .expect(method);
            assert!(out.num_vehicles <= 2, "{method}");
            let (internal, _) = build_customers((0.0, 0.0), &customers).expect("valid");
            let dm = DistanceMatrix::from_customers(&internal);
            for route in &out.routes {
                // ids are 1..=6 and equal the internal indices here
                assert!(
                    crate::evaluation::route_feasible(
                        route,
                        0,
                        &dm,
                        &internal,
                        &crate::evaluation::RouteLimits::NONE
                    ),
                    "{method}: late on {route:?}"
                );
            }
        }
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
        assert!(err.message.contains("max_vehicles is 0"), "{err}");
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
                err.message.contains(&format!("\"{method}\"")) && err.message.contains("capacit"),
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
        let no_fleet = InputConfig {
            max_vehicles: Some(0),
            ..quick()
        };
        let errors = [
            solve((0.0, 0.0), &ring(2), &fleet(&[10.0, 100.0]), "ga", &quick())
                .expect_err("mixed fleet"),
            solve((0.0, 0.0), &fraction, &[], "nn", &quick()).expect_err("fractional demand"),
            solve((0.0, 0.0), &ring(8), &fleet(&[10.0]), "ga", &no_fleet)
                .expect_err("zero max_vehicles"),
        ];
        for e in errors {
            assert!(!e.message.contains("  "), "{e}");
        }
    }

    /// Every refusal names its reason as a `code` and carries the values
    /// behind it, so a caller branches on the code rather than on the text.
    #[test]
    fn every_refusal_carries_its_code_and_values() {
        let one = |json: serde_json::Value| [customer(json)];
        let cases: Vec<(ServiceError, &str, serde_json::Value)> = vec![
            (
                solve((0.0, 0.0), &[], &[], "tabu", &quick()).expect_err("method"),
                "unknown_option",
                serde_json::json!({ "parameter": "method", "got": "tabu" }),
            ),
            (
                solve(
                    (0.0, 0.0),
                    &[
                        customer(serde_json::json!({ "id": 5, "x": 1.0, "y": 0.0 })),
                        customer(serde_json::json!({ "id": 5, "x": 2.0, "y": 0.0 })),
                    ],
                    &[],
                    "nn",
                    &quick(),
                )
                .expect_err("duplicate"),
                "duplicate_id",
                serde_json::json!({ "id": 5, "first": 0, "second": 1 }),
            ),
            (
                solve(
                    (0.0, 0.0),
                    &one(serde_json::json!({ "id": 9, "x": 1.0, "y": 0.0, "demand": 0.5 })),
                    &[],
                    "nn",
                    &quick(),
                )
                .expect_err("fraction"),
                "not_whole_units",
                serde_json::json!({ "parameter": "demand", "index": 0, "id": 9, "value": 0.5 }),
            ),
            (
                solve((0.0, 0.0), &[], &fleet(&[10.0, 2.5]), "nn", &quick())
                    .expect_err("capacity"),
                "not_whole_units",
                serde_json::json!({ "parameter": "capacity", "index": 1, "id": null }),
            ),
            (
                solve(
                    (0.0, 0.0),
                    &one(serde_json::json!({ "id": 3, "x": 1.0, "y": 0.0, "time_window": [4.0, 1.0] })),
                    &[],
                    "nn",
                    &quick(),
                )
                .expect_err("window"),
                "invalid_time_window",
                serde_json::json!({ "id": 3, "index": 0, "ready": 4.0, "due": 1.0 }),
            ),
            (
                solve((0.0, 0.0), &ring(2), &fleet(&[10.0, 100.0]), "ga", &quick())
                    .expect_err("mixed"),
                "mixed_fleet",
                serde_json::json!({ "method": "ga", "capacities": [10, 100], "index": 1 }),
            ),
            (
                solve(
                    (0.0, 0.0),
                    &ring(8),
                    &fleet(&[10.0]),
                    "ga",
                    &InputConfig {
                        max_vehicles: Some(0),
                        ..quick()
                    },
                )
                .expect_err("zero"),
                "invalid_option",
                serde_json::json!({ "parameter": "max_vehicles", "value": 0 }),
            ),
            (
                solve(
                    (0.0, 0.0),
                    &ring(4),
                    &fleet(&[10.0]),
                    "ga",
                    &InputConfig {
                        population_size: Some(1),
                        ..quick()
                    },
                )
                .expect_err("settings"),
                "invalid_option",
                serde_json::json!({ "parameter": "config", "method": "ga" }),
            ),
        ];
        for (err, code, expected) in cases {
            assert_eq!(err.code(), code, "{err}");
            for (key, value) in expected.as_object().expect("fields") {
                assert_eq!(&err.fields[key], value, "{code}.{key}: {}", err.fields);
            }
        }
    }

    // ---- GA: valid minimal input ----

    #[test]
    fn ga_valid_minimal() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(result.is_ok(), "GA with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method, Method::Genetic);
        assert!(!output.routes.is_empty());
        assert!(output.total_distance > 0.0);
    }

    // ---- GA: population_size too small ----

    #[test]
    fn ga_population_size_too_small() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(1),
            max_generations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(result.is_err(), "population_size=1 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("population_size"),
            "error should mention population_size: {}",
            err
        );
    }

    #[test]
    fn ga_population_size_zero() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(0),
            max_generations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(result.is_err(), "population_size=0 should fail validation");
    }

    // ---- GA: max_generations zero ----

    #[test]
    fn ga_zero_generations() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(result.is_err(), "max_generations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("max_generations"),
            "error should mention max_generations: {}",
            err
        );
    }

    // ---- GA: elite_ratio too high ----

    #[test]
    fn ga_elite_ratio_fills_population() {
        let (customers, dm) = test_customers(3);
        // elite_ratio 1.0 with pop=2 makes every individual elite → validation error
        let cfg = InputConfig {
            population_size: Some(2),
            max_generations: Some(5),
            elite_ratio: Some(1.0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(
            result.is_err(),
            "elite_ratio filling entire population should fail"
        );
    }

    // ---- GA: single customer ----

    #[test]
    fn ga_single_customer() {
        let (customers, dm) = test_customers(1);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(result.is_ok(), "GA with 1 customer should succeed");
        let output = result.unwrap();
        assert_eq!(output.routes.len(), 1);
    }

    // ---- GA: rates outside (0, 1] are refused, not clamped ----

    #[test]
    fn ga_rates_outside_0_1_are_refused() {
        let (customers, dm) = test_customers(3);
        for (mutation_rate, elite_ratio, parameter, got) in [
            (Some(5.0), None, "config.mutation_rate", 5.0),
            (Some(0.0), None, "config.mutation_rate", 0.0),
            (None, Some(1.5), "config.elite_ratio", 1.5),
        ] {
            let cfg = InputConfig {
                population_size: Some(10),
                max_generations: Some(5),
                mutation_rate,
                elite_ratio,
                seed: Some(42),
                ..InputConfig::default()
            };
            let err = solve_ga(&customers, &dm, 100, &cfg).expect_err("out of (0, 1]");
            assert_eq!(
                err.fields,
                json!({ "code": "parameter_out_of_range", "parameter": parameter,
                        "min": 0.0, "max": 1.0, "got": got })
            );
        }
    }

    // ---- ALNS: valid minimal input ----

    #[test]
    fn alns_valid_minimal() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            max_iterations: Some(10),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_alns(&customers, &dm, 100, &cfg);
        assert!(result.is_ok(), "ALNS with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method, Method::Alns);
        assert!(!output.routes.is_empty());
    }

    // ---- ALNS: zero iterations ----

    #[test]
    fn alns_zero_iterations() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            max_iterations: Some(0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_alns(&customers, &dm, 100, &cfg);
        assert!(result.is_err(), "max_iterations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("max_iterations"),
            "error should mention max_iterations: {}",
            err
        );
    }

    // ---- Default config (no config provided) ----

    #[test]
    fn ga_default_config() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_ga(&customers, &dm, 100, &cfg);
        assert!(result.is_ok(), "GA with default config should succeed");
    }

    #[test]
    fn alns_default_config() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_alns(&customers, &dm, 100, &cfg);
        assert!(result.is_ok(), "ALNS with default config should succeed");
    }

    // ---- GA: larger problem to stress-test GA operators ----

    #[test]
    fn ga_larger_problem() {
        let (customers, dm) = test_customers(20);
        let cfg = InputConfig {
            population_size: Some(30),
            max_generations: Some(20),
            seed: Some(123),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 1000, &cfg);
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
        let (customers, dm) = test_customers(10);
        // Each customer has demand=5, capacity=5 forces one customer per route
        let cfg = InputConfig {
            population_size: Some(20),
            max_generations: Some(10),
            seed: Some(99),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 5, &cfg);
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
        let (customers, dm) = test_customers(2);
        let cfg = InputConfig {
            population_size: Some(10),
            max_generations: Some(5),
            seed: Some(42),
            ..InputConfig::default()
        };
        let result = solve_ga(&customers, &dm, 100, &cfg);
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
