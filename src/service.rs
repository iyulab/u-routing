//! The solver entry point shared by the WebAssembly and C bindings.
//!
//! Both bindings take a problem as JSON and return a solution as JSON. What
//! differs between them is only the transport -- a JS object or a C string --
//! and the shape of a couple of top-level fields. The solver dispatch, the
//! defaults and the input checks live here once. They used to live in each
//! binding, and the C copy fell behind: it knew two of the four methods,
//! ignored solver settings, and reported whatever method name it was sent as
//! the one it had used.

use std::ops::{Bound, RangeBounds};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::alns::destroy::RandomRemoval;
use crate::alns::repair::GreedyInsertion;
use crate::alns::RoutingAlnsProblem;
use crate::constructive::{clarke_wright_savings, nearest_neighbor, nearest_neighbor_tw};
use crate::distance::DistanceMatrix;
use crate::evaluation::{has_time_windows, route_checks_needed, RouteLimits};
use crate::fleet::limit_routes;
use crate::ga::RoutingGaProblem;
use crate::ga::{split, split_tw};
use crate::local_search::{or_opt_improve, two_opt_improve};
use crate::models::{Customer, TimeWindow, Vehicle};
use u_metaheur::alns::{AlnsConfig, AlnsRunner};
use u_metaheur::ga::{GaConfig, GaRunner};
use u_metaheur::ConfigError;

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

/// The bounds a `parameter_out_of_range` refusal reports. A closed (included)
/// end is `min` / `max`, an open (excluded) end `exclusive_min` /
/// `exclusive_max`; a side holds at most one of its two.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bounds {
    min: Option<f64>,
    max: Option<f64>,
    exclusive_min: Option<f64>,
    exclusive_max: Option<f64>,
}

impl Bounds {
    /// Reads a range's ends: `2.0..` is "at least 2",
    /// `(Excluded(0.0), Included(1.0))` is `(0, 1]`.
    fn of(range: impl RangeBounds<f64>) -> Self {
        let split = |bound: Bound<&f64>| match bound {
            Bound::Included(&v) => (Some(v), None),
            Bound::Excluded(&v) => (None, Some(v)),
            Bound::Unbounded => (None, None),
        };
        let (min, exclusive_min) = split(range.start_bound());
        let (max, exclusive_max) = split(range.end_bound());
        Self {
            min,
            max,
            exclusive_min,
            exclusive_max,
        }
    }

    /// Writes the bounds into a refusal's fields: `min` and `max` always
    /// (`null` on a side that is open or unbounded), `exclusive_min` and
    /// `exclusive_max` only when that end is open.
    fn insert_into(self, fields: &mut serde_json::Value) {
        fields["min"] = json!(self.min);
        fields["max"] = json!(self.max);
        for (key, open) in [
            ("exclusive_min", self.exclusive_min),
            ("exclusive_max", self.exclusive_max),
        ] {
            if let Some(value) = open {
                fields[key] = json!(value);
            }
        }
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Where in the argument `param` a request stopped deserializing, as the
/// refusal reports it: the field (`customers[1].x`, `config.seed`) and, when
/// the failure sits in an array, its position there. Both bindings name a
/// malformed value this way.
pub(crate) fn failure_site(
    param: &str,
    path: &serde_path_to_error::Path,
) -> (String, Option<usize>) {
    use serde_path_to_error::Segment;
    let segments: Vec<&Segment> = path.iter().collect();
    let index = segments.iter().rev().find_map(|s| match s {
        Segment::Seq { index } => Some(*index),
        _ => None,
    });
    // A trailing `[i]` is the index, not part of the name.
    let named = match segments.last() {
        Some(Segment::Seq { .. }) => &segments[..segments.len() - 1],
        _ => &segments[..],
    };
    let mut name = String::new();
    for segment in named {
        match segment {
            Segment::Seq { index } => name.push_str(&format!("[{index}]")),
            Segment::Map { key } | Segment::Enum { variant: key } => {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(key);
            }
            Segment::Unknown => name.push_str(".?"),
        }
    }
    // A top-level array argument (`data[1][2]`) or a failure at the root
    // (a missing field, a syntax error) is named by the argument itself.
    if name.is_empty() || name.starts_with('[') {
        name.insert_str(0, param);
    }
    (name, index)
}

/// A refusal of a value that does not deserialize, at the place it stopped:
/// `parameter` and, inside an array, `index` (see [`failure_site`]).
pub(crate) fn malformed_at<E: std::fmt::Display>(
    param: &str,
    error: &serde_path_to_error::Error<E>,
) -> ServiceError {
    let (parameter, index) = failure_site(param, error.path());
    let mut err = ServiceError::malformed_input(
        &parameter,
        format!("{param}: {}: {}", error.path(), error.inner()),
    );
    if let Some(i) = index {
        err.fields["index"] = json!(i);
    }
    err
}

// ============================================================================
// Whole numbers
// ============================================================================

/// A whole number exactly as the caller wrote it: any integer, or a number
/// with no fractional part.
///
/// Counts, ids and the seed are read through this rather than straight into
/// an unsigned type. As a `usize`, `population_size: -1` failed as a type
/// mismatch ("invalid value: integer `-1`, expected usize",
/// `malformed_input`) while `1` was refused by the range check -- two answers
/// to one mistake. Read signed and wide, every whole number reaches a range
/// check, which names the accepted range ([`whole_in`], [`runner_count`]). A
/// fractional part is refused here: whether a count is whole is a matter of
/// its type, so `2.5` is `malformed_input`.
struct Whole(i128);

impl<'de> Deserialize<'de> for Whole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WholeVisitor;

        impl serde::de::Visitor<'_> for WholeVisitor {
            type Value = Whole;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a whole number")
            }

            fn visit_i64<E>(self, v: i64) -> Result<Whole, E> {
                Ok(Whole(v.into()))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Whole, E> {
                Ok(Whole(v.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Whole, E> {
                if v.fract() == 0.0 {
                    // `as` saturates only past ±1.7e38, far outside every
                    // range a setting accepts.
                    Ok(Whole(v as i128))
                } else {
                    Err(E::custom(format_args!("must be a whole number, got {v}")))
                }
            }
        }

        deserializer.deserialize_any(WholeVisitor)
    }
}

/// Reads a whole-number field (`#[serde(deserialize_with = "whole")]`); see
/// [`Whole`].
fn whole<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i128, D::Error> {
    Whole::deserialize(deserializer).map(|w| w.0)
}

/// Reads an optional whole-number field: `null` is no value.
fn optional_whole<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i128>, D::Error> {
    Option::<Whole>::deserialize(deserializer).map(|w| w.map(|w| w.0))
}

/// A whole number from the request as the `T` the solver takes.
///
/// Below `min` -- a negative value as much as a too-small one -- it is
/// refused as `parameter_out_of_range` with `min` and no upper bound; above
/// the largest `T` (`max`, the type's own limit) with both bounds.
fn whole_in<T: TryFrom<i128>>(
    parameter: &str,
    value: i128,
    min: i128,
    max: i128,
) -> Result<T, ServiceError> {
    let (bounds, rule) = if value < min {
        (Bounds::of(min as f64..), format!("at least {min}"))
    } else {
        match T::try_from(value) {
            Ok(v) => return Ok(v),
            Err(_) => (
                Bounds::of(min as f64..=max as f64),
                format!("from {min} to {max}"),
            ),
        }
    };
    let mut fields = json!({ "parameter": parameter, "got": value as f64 });
    bounds.insert_into(&mut fields);
    Err(ServiceError::new(
        "parameter_out_of_range",
        format!("{parameter} must be {rule}, got {value}"),
        fields,
    ))
}

/// A count from `config` as the `usize` the method's runner takes.
///
/// The runner checks its own counts (a population of at least 2, at least one
/// generation or iteration), and its refusal is the one reported
/// ([`settings_refused`]). A negative count cannot reach that check as a
/// `usize`, so it is put to it as 0 -- below every count a runner accepts --
/// and refused with the caller's number as `got`: `-1` and `1` are refused
/// alike, with the runner's bounds. `check` validates a runner config that
/// differs from the default only in this count.
fn runner_count(
    method: Method,
    parameter: &str,
    value: i128,
    check: impl Fn(usize) -> Result<(), ConfigError>,
) -> Result<usize, ServiceError> {
    if let Ok(count) = usize::try_from(value) {
        return Ok(count);
    }
    let name = parameter.trim_start_matches("config.");
    if value < 0 {
        if let Err(mut e) = check(0) {
            if let ConfigError::OutOfRange { parameter, got, .. } = &mut e {
                if *parameter == name {
                    *got = value as f64;
                    return Err(settings_refused(method, e));
                }
            }
        }
    }
    // A count past what the target holds, or one the runner sets no minimum
    // for: the type is the limit.
    let mut e = whole_in::<usize>(parameter, value, 0, usize::MAX as i128)
        .expect_err("a value no usize holds is outside 0..=usize::MAX");
    e.fields["method"] = json!(method.name());
    e.message = format!("method \"{}\": {}", method.name(), e.message);
    Err(e)
}

/// The seed from `config`, for the methods that read one: a whole number from
/// 0 to 2^64 - 1.
fn seed(method: Method, cfg: &InputConfig) -> Result<Option<u64>, ServiceError> {
    cfg.seed
        .map(|seed| {
            whole_in("config.seed", seed, 0, u64::MAX.into()).map_err(|mut e| {
                e.fields["method"] = json!(method.name());
                e.message = format!("method \"{}\": {}", method.name(), e.message);
                e
            })
        })
        .transpose()
}

// ============================================================================
// Wire types shared by both bindings
// ============================================================================

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputCustomer {
    /// A whole number from 0, the caller's own label; routes and
    /// `unassigned` name the customer by it. Read signed so a negative id is
    /// a range refusal: [`Whole`].
    #[serde(deserialize_with = "whole")]
    #[cfg_attr(feature = "wasm", tsify(type = "number"))]
    id: i128,
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
    /// Longest total distance one route of this vehicle may travel, depot to
    /// depot. Absent or `null`: no limit.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    max_distance: Option<f64>,
    /// Latest return to the depot, on the clock that starts at 0 (travel time
    /// equals distance, plus waiting and service). Absent or `null`: no limit.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    max_duration: Option<f64>,
}

fn default_capacity() -> f64 {
    1e9
}

/// Optional solver configuration.
///
/// Fields are shared across methods; each method uses only the relevant ones.
/// Counts and the seed are whole numbers read signed, so a negative one is
/// refused by the range check like a too-small one ([`Whole`]).
/// Besides the per-method parameters, it carries limits the solver has to
/// respect whichever method runs -- `max_vehicles` is one.
#[derive(Deserialize, Default)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputConfig {
    // --- GA parameters ---
    /// Population size for GA (default: 50).
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    population_size: Option<i128>,
    /// Maximum generations for GA (default: 200).
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    max_generations: Option<i128>,
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
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    max_iterations: Option<i128>,

    // --- Shared ---
    /// Random seed for reproducibility: a whole number from 0 to 2^64 - 1.
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    seed: Option<i128>,
    /// The number of routes the plan may use at most -- the fixed fleet.
    ///
    /// The length of `vehicles` is not this number: it says which capacities
    /// exist, and for `"savings"`, `"ga"` and `"alns"` a single-entry list is
    /// how a caller states one capacity for an unbounded fleet. A caller with
    /// a fixed fleet says so here, and every method keeps to it: a plan that
    /// would need more routes has its lightest routes emptied into the others
    /// where capacity and time windows allow, and the customers that fit
    /// nowhere are reported in `unassigned`.
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    max_vehicles: Option<i128>,
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
    // At least one route, or no limit at all: 0 is a range refusal like -1.
    let max_vehicles = config
        .max_vehicles
        .map(|n| {
            whole_in::<usize>("config.max_vehicles", n, 1, usize::MAX as i128).map_err(|mut e| {
                e.message
                    .push_str(" -- leave it out to accept as many routes as the plan needs");
                e
            })
        })
        .transpose()?;
    let (customers, id_map) = build_customers(depot, input_customers)?;
    let vehicles = build_vehicles(input_vehicles)?;
    // Only nearest neighbour assigns routes to particular vehicles. The other
    // methods plan every route with one vehicle -- one capacity and one set of
    // limits -- so a fleet that differs in either is not a problem they can
    // state (it used to be solved as a fleet of the first vehicle).
    if method != Method::NearestNeighbor {
        refuse_mixed_fleet(method, &vehicles)?;
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
    // The capacity and limits every route of savings, GA and ALNS is planned
    // with (one vehicle type -- checked above).
    let capacity = vehicles[0].capacity();
    let limits = RouteLimits::of(&vehicles[0]);

    let plan = match method {
        Method::NearestNeighbor => solve_nn(&customers, &dm, &vehicles),
        Method::Savings => solve_savings(&customers, &dm, &vehicles),
        Method::Genetic => solve_ga(&customers, &dm, capacity, limits, config)?,
        Method::Alns => solve_alns(&customers, &dm, capacity, limits, config)?,
    };
    // A fixed fleet is kept by every method alike: `"nn"` cannot exceed its
    // vehicle list, but it can exceed a smaller `max_vehicles`.
    let plan = match max_vehicles {
        Some(max) if plan.routes.len() > max => {
            let limited = limit_routes(
                plan.routes,
                plan.capacities,
                plan.limits,
                max,
                &dm,
                &customers,
            );
            let (routes, total_distance) =
                apply_local_search(&limited.routes, &limited.limits, &dm, &customers);
            Plan {
                routes,
                capacities: limited.capacities,
                limits: limited.limits,
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
    /// The distance and duration limits of each route, in the same order.
    limits: Vec<RouteLimits>,
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
        let id: usize = whole_in("id", ic.id, 0, usize::MAX as i128).map_err(|mut e| {
            e.fields["index"] = json!(position);
            e.message = format!(
                "customer at position {position} of customers: {}",
                e.message
            );
            e
        })?;
        if let Some(first) = first_at.insert(id, position) {
            return Err(ServiceError::new(
                "duplicate_id",
                format!(
                    "customer {id}: the id is given twice, at positions {first} and \
                     {position} of customers (counting from 0); routes and unassigned \
                     name customers by id, so every customer needs its own"
                ),
                json!({ "id": id, "first": first, "second": position }),
            ));
        }
        let demand = whole_units(ic.demand, "demand", position, Some(id), || {
            format!("customer {id}: demand")
        })?;
        let idx = customers.len();
        id_map.push(id);
        let mut c = Customer::new(idx, ic.x, ic.y, demand, ic.service_time);
        if let Some([ready, due]) = ic.time_window {
            let tw = TimeWindow::new(ready, due).ok_or_else(|| {
                ServiceError::new(
                    "invalid_time_window",
                    format!(
                        "customer {id}: time_window [{ready}, {due}] is not a window \
                         (it needs ready <= due)"
                    ),
                    json!({ "id": id, "index": position, "ready": ready, "due": due }),
                )
            })?;
            c = c.with_time_window(tw);
        }
        customers.push(c);
    }

    Ok((customers, id_map))
}

/// Solver settings the method's runner refused (a population of 1, zero
/// generations or iterations), named by the `config` field the runner checked:
/// a number outside its range is `parameter_out_of_range` with the bounds as
/// the runner reported them (open ends as `exclusive_min` / `exclusive_max`),
/// a setting wrong only beside the others is `invalid_option`.
fn settings_refused(method: Method, e: ConfigError) -> ServiceError {
    let parameter = format!("config.{}", e.parameter());
    let message = format!("method \"{}\": config.{e}", method.name());
    match e {
        ConfigError::OutOfRange {
            min,
            max,
            exclusive_min,
            exclusive_max,
            got,
            ..
        } => {
            let mut fields = json!({ "parameter": parameter, "method": method.name(), "got": got });
            Bounds {
                min,
                max,
                exclusive_min,
                exclusive_max,
            }
            .insert_into(&mut fields);
            ServiceError::new("parameter_out_of_range", message, fields)
        }
        _ => ServiceError::new(
            "invalid_option",
            message,
            json!({ "parameter": parameter, "method": method.name() }),
        ),
    }
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
    limits: &[RouteLimits],
    dm: &DistanceMatrix,
    customers: &[Customer],
) -> (Vec<Vec<usize>>, f64) {
    let mut improved_routes = Vec::with_capacity(routes.len());
    let mut total = 0.0;
    for (route, limits) in routes.iter().zip(limits) {
        let (r1, _) = two_opt_improve(route, 0, dm, customers, limits);
        let (r2, dist) = or_opt_improve(&r1, 0, dm, customers, limits);
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
/// reports it as solved. A fraction is `not_whole_units` -- the unit is the
/// thing to change -- and a whole number outside `[0, i32::MAX]` is
/// `parameter_out_of_range` with those bounds.
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
    let accepted = 0.0..=f64::from(i32::MAX);
    if value.fract() != 0.0 {
        return Err(ServiceError::new(
            "not_whole_units",
            format!(
                "{} is {value}; it must be a whole number of units -- scale the unit \
                 (kilograms to grams, say) to keep a fractional amount",
                what(),
            ),
            json!({ "parameter": parameter, "index": index, "id": id, "value": value }),
        ));
    }
    if !accepted.contains(&value) {
        let mut fields = json!({ "parameter": parameter, "index": index, "id": id, "got": value });
        Bounds::of(accepted).insert_into(&mut fields);
        return Err(ServiceError::new(
            "parameter_out_of_range",
            format!(
                "{} is {value}; it must be from 0 to {} units",
                what(),
                i32::MAX
            ),
            fields,
        ));
    }
    Ok(value as i32)
}

/// Builds the vehicle list from input, falling back to a single unlimited vehicle.
/// `mixed_fleet` at the first vehicle that differs from the first one in
/// capacity, `max_distance` or `max_duration`, naming that field.
fn refuse_mixed_fleet(method: Method, vehicles: &[Vehicle]) -> Result<(), ServiceError> {
    let first = &vehicles[0];
    for (index, other) in vehicles.iter().enumerate().skip(1) {
        let differs = [
            ("capacity", json!(first.capacity()), json!(other.capacity())),
            (
                "max_distance",
                json!(first.max_distance()),
                json!(other.max_distance()),
            ),
            (
                "max_duration",
                json!(first.max_duration()),
                json!(other.max_duration()),
            ),
        ]
        .into_iter()
        .find(|(_, a, b)| a != b);
        if let Some((parameter, a, b)) = differs {
            let plural = if parameter == "capacity" {
                "capacities"
            } else {
                parameter
            };
            return Err(ServiceError::new(
                "mixed_fleet",
                format!(
                    "method \"{}\" plans every route with one vehicle, but the fleet has \
                     {plural} {a} and {b}; give every vehicle the same {parameter}, or use \
                     \"nn\", which reads each vehicle",
                    method.name(),
                ),
                json!({
                    "method": method.name(),
                    "parameter": parameter,
                    "values": [a, b],
                    "capacities": [first.capacity(), other.capacity()],
                    "index": index,
                }),
            ));
        }
    }
    Ok(())
}

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
            let mut vehicle = Vehicle::new(i, capacity);
            for (parameter, value) in [
                ("max_distance", v.max_distance),
                ("max_duration", v.max_duration),
            ] {
                let Some(limit) = value else { continue };
                // JSON has no infinity, so only a non-positive limit reaches here:
                // no route could keep it, and "no limit" is said by leaving it out.
                if limit.is_nan() || limit <= 0.0 {
                    let mut fields = json!({ "parameter": parameter, "index": i, "got": limit });
                    Bounds::of((Bound::Excluded(0.0), Bound::Unbounded)).insert_into(&mut fields);
                    return Err(ServiceError::new(
                        "parameter_out_of_range",
                        format!("vehicle {i}: {parameter} must be > 0, got {limit}"),
                        fields,
                    ));
                }
                vehicle = if parameter == "max_distance" {
                    vehicle.with_max_distance(limit)
                } else {
                    vehicle.with_max_duration(limit)
                };
            }
            Ok(vehicle)
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
        limits: solution
            .routes()
            .iter()
            .map(|r| RouteLimits::of(&vehicles[r.vehicle_id()]))
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
        limits: vec![RouteLimits::of(vehicle_template); routes.len()],
        routes,
        total_distance: solution.total_distance(),
        method: Method::Savings,
    }
}

fn solve_ga(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    limits: RouteLimits,
    cfg: &InputConfig,
) -> Result<Plan, ServiceError> {
    let problem =
        RoutingGaProblem::new(customers.to_vec(), dm.clone(), capacity).with_limits(limits);

    // Sequential on every target: rayon is unavailable in WebAssembly, and a
    // seed should reproduce the same routes whichever binding runs it.
    let method = Method::Genetic;
    let population_size = match cfg.population_size {
        Some(n) => runner_count(method, "config.population_size", n, |n| {
            GaConfig::default().with_population_size(n).validate()
        })?,
        None => 50,
    };
    let max_generations = match cfg.max_generations {
        Some(n) => runner_count(method, "config.max_generations", n, |n| {
            GaConfig::default().with_max_generations(n).validate()
        })?,
        None => 200,
    };
    let mut ga_config = GaConfig::default()
        .with_population_size(population_size)
        .with_max_generations(max_generations)
        .with_parallel(false);

    // Rates, refused rather than clamped: a mutation rate in (0, 1]; an elite
    // ratio in (0, 1), since a ratio of 1 makes every individual an elite.
    for (parameter, value, accepted, text) in [
        (
            "config.mutation_rate",
            cfg.mutation_rate,
            (Bound::Excluded(0.0), Bound::Included(1.0)),
            "(0, 1]",
        ),
        (
            "config.elite_ratio",
            cfg.elite_ratio,
            (Bound::Excluded(0.0), Bound::Excluded(1.0)),
            "(0, 1)",
        ),
    ] {
        if let Some(v) = value.filter(|v| !accepted.contains(v)) {
            let mut fields = json!({ "parameter": parameter, "got": v });
            Bounds::of(accepted).insert_into(&mut fields);
            return Err(ServiceError::new(
                "parameter_out_of_range",
                format!("{parameter} must be in {text}, got {v}"),
                fields,
            ));
        }
    }
    if let Some(mr) = cfg.mutation_rate {
        ga_config = ga_config.with_mutation_rate(mr);
    }
    if let Some(er) = cfg.elite_ratio {
        ga_config = ga_config.with_elite_ratio(er);
    }
    if let Some(seed) = seed(method, cfg)? {
        ga_config = ga_config.with_seed(seed);
    }

    ga_config
        .validate()
        .map_err(|e| settings_refused(Method::Genetic, e))?;

    let ga_result =
        GaRunner::run(&problem, &ga_config).map_err(|e| settings_refused(Method::Genetic, e))?;

    // Split the best individual into routes the way its fitness was computed.
    // 2-opt and or-opt only take moves that keep every customer on time and
    // the route within the limits, so the split routes can be polished.
    let tour = ga_result.best.customers();
    let routes = if route_checks_needed(customers, &limits) {
        split_tw(tour, customers, dm, capacity, &limits).routes
    } else {
        split(tour, customers, dm, capacity).routes
    };
    let route_limits = vec![limits; routes.len()];
    let (routes, total_distance) = apply_local_search(&routes, &route_limits, dm, customers);
    Ok(Plan {
        capacities: vec![capacity; routes.len()],
        limits: route_limits,
        routes,
        total_distance,
        method: Method::Genetic,
    })
}

fn solve_alns(
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
    limits: RouteLimits,
    cfg: &InputConfig,
) -> Result<Plan, ServiceError> {
    let problem =
        RoutingAlnsProblem::new(customers.to_vec(), dm.clone(), capacity).with_limits(limits);

    let destroy_ops = vec![RandomRemoval];
    let repair_ops =
        vec![GreedyInsertion::new(dm.clone(), customers.to_vec(), capacity).with_limits(limits)];

    let method = Method::Alns;
    let max_iterations = match cfg.max_iterations {
        Some(n) => runner_count(method, "config.max_iterations", n, |n| {
            AlnsConfig::default().with_max_iterations(n).validate()
        })?,
        None => 500,
    };
    let mut alns_config = AlnsConfig::default().with_max_iterations(max_iterations);

    if let Some(seed) = seed(method, cfg)? {
        alns_config = alns_config.with_seed(seed);
    }

    alns_config
        .validate()
        .map_err(|e| settings_refused(Method::Alns, e))?;

    let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &alns_config)
        .map_err(|e| settings_refused(Method::Alns, e))?;

    // Apply local search to improve ALNS result
    let alns_routes: Vec<Vec<usize>> = result.best.routes().to_vec();
    let route_limits = vec![limits; alns_routes.len()];
    let (routes, total_distance) = apply_local_search(&alns_routes, &route_limits, dm, customers);
    Ok(Plan {
        capacities: vec![capacity; routes.len()],
        limits: route_limits,
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

    // ---- vehicle max_distance / max_duration ----

    /// Every route of every method keeps the vehicle's limits -- checked by the
    /// evaluator, which prices each route on its own -- and a customer no route
    /// can reach within them is reported unassigned, with or without a fixed
    /// fleet size.
    #[test]
    fn every_method_keeps_the_vehicle_limits() {
        let mut customers = ring(8);
        // Out and back is 200: beyond any route of this fleet.
        customers.push(customer(serde_json::json!({
            "id": 9, "x": 100.0, "y": 0.0, "demand": 5.0, "service_time": 1.0
        })));
        let limited = |extra: serde_json::Value| {
            let mut v = serde_json::json!({ "capacity": 1000.0 });
            v.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            vec![vehicle(v.clone()), vehicle(v)]
        };
        let (internal, _) = build_customers((0.0, 0.0), &customers).expect("valid");
        let dm = DistanceMatrix::from_customers(&internal);
        for (limit, fleet) in [
            (
                "max_distance",
                limited(serde_json::json!({ "max_distance": 45.0 })),
            ),
            (
                "max_duration",
                limited(serde_json::json!({ "max_duration": 47.0 })),
            ),
        ] {
            // Built from the numbers, not from the code under test.
            let oracle = match limit {
                "max_distance" => Vehicle::new(0, 1000).with_max_distance(45.0),
                _ => Vehicle::new(0, 1000).with_max_duration(47.0),
            };
            for max_vehicles in [None, Some(3)] {
                let config = InputConfig {
                    max_vehicles,
                    ..quick()
                };
                for method in ["nn", "savings", "ga", "alns"] {
                    let out = solve((0.0, 0.0), &customers, &fleet, method, &config)
                        .unwrap_or_else(|e| panic!("{method}: {e:?}"));
                    assert!(out.unassigned.contains(&9), "{method} {limit}: {out:?}");
                    let evaluator = crate::evaluation::RouteEvaluator::new(&internal, &dm, &oracle);
                    for route in &out.routes {
                        // ids 1..=9 are the internal indices here
                        let (_, violations) = evaluator.build_route(route);
                        assert!(
                            violations.is_empty(),
                            "{method} {limit} {max_vehicles:?}: {violations:?} on {route:?}"
                        );
                    }
                    if let Some(max) = max_vehicles {
                        assert!(out.routes.len() as i128 <= max, "{method}: {out:?}");
                    }
                    let served: usize = out.routes.iter().map(Vec::len).sum();
                    assert_eq!(served + out.unassigned.len(), 9, "{method} lost a customer");
                }
            }
        }
    }

    #[test]
    fn a_limit_is_refused_when_no_route_could_keep_it() {
        let customers = ring(3);
        let fleet = vec![vehicle(
            serde_json::json!({ "capacity": 10.0, "max_distance": 0.0 }),
        )];
        let e = solve((0.0, 0.0), &customers, &fleet, "nn", &quick()).expect_err("0 distance");
        // A limit must be > 0: zero itself is refused, so the bound is open.
        assert_eq!(
            e.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "max_distance", "index": 0,
                    "min": null, "max": null, "exclusive_min": 0.0, "got": 0.0 })
        );
    }

    /// The methods that plan every route alike need one vehicle type; nearest
    /// neighbour reads each vehicle and takes a mixed fleet.
    #[test]
    fn a_fleet_mixed_in_its_limits_is_refused_by_the_one_vehicle_methods() {
        let customers = ring(6);
        let fleet = vec![
            vehicle(serde_json::json!({ "capacity": 100.0, "max_duration": 60.0 })),
            vehicle(serde_json::json!({ "capacity": 100.0 })),
        ];
        for method in ["savings", "ga", "alns"] {
            let e = solve((0.0, 0.0), &customers, &fleet, method, &quick()).expect_err(method);
            assert_eq!(e.fields["code"], "mixed_fleet", "{method}");
            assert_eq!(e.fields["parameter"], "max_duration", "{method}");
            assert_eq!(e.fields["index"], 1, "{method}");
        }
        assert!(solve((0.0, 0.0), &customers, &fleet, "nn", &quick()).is_ok());
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
    /// stated rather than turning every solve into a route-count failure --
    /// by the same range check as a negative count, whatever the method.
    #[test]
    fn a_max_vehicles_below_one_is_refused_as_stated() {
        for (got, method) in [(0, "ga"), (-1, "nn")] {
            let config = InputConfig {
                max_vehicles: Some(got),
                ..quick()
            };
            let err =
                solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), method, &config).expect_err("below 1");
            assert_eq!(
                err.fields,
                serde_json::json!({ "code": "parameter_out_of_range",
                    "parameter": "config.max_vehicles", "min": 1.0, "max": null,
                    "got": got as f64 })
            );
            assert!(
                err.message.starts_with(&format!(
                    "config.max_vehicles must be at least 1, got {got}"
                )),
                "{err}"
            );
        }
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
        let by_id = |id: usize| {
            customers
                .iter()
                .find(|c| c.id == id as i128)
                .expect("known id")
        };
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
                    &one(serde_json::json!({ "id": 9, "x": 1.0, "y": 0.0, "demand": -3.0 })),
                    &[],
                    "nn",
                    &quick(),
                )
                .expect_err("negative"),
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "demand", "index": 0, "id": 9,
                                    "min": 0.0, "max": f64::from(i32::MAX), "got": -3.0 }),
            ),
            (
                solve((0.0, 0.0), &[], &fleet(&[10.0, 3e9]), "nn", &quick())
                    .expect_err("too large"),
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "capacity", "index": 1, "id": null,
                                    "min": 0.0, "max": f64::from(i32::MAX), "got": 3e9 }),
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
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "config.max_vehicles", "min": 1.0,
                                    "max": null, "got": 0.0 }),
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
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "config.population_size", "method": "ga",
                                    "min": 2.0, "max": null, "got": 1.0 }),
            ),
            (
                solve(
                    (0.0, 0.0),
                    &ring(4),
                    &fleet(&[10.0]),
                    "alns",
                    &InputConfig {
                        max_iterations: Some(0),
                        ..quick()
                    },
                )
                .expect_err("no iterations"),
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "config.max_iterations", "method": "alns",
                                    "min": 1.0, "got": 0.0 }),
            ),
        ];
        for (err, code, expected) in cases {
            assert_eq!(err.code(), code, "{err}");
            for (key, value) in expected.as_object().expect("fields") {
                assert_eq!(&err.fields[key], value, "{code}.{key}: {}", err.fields);
            }
        }
    }

    // ---- whole numbers ----

    /// A count is a whole number read signed: a negative one reaches the
    /// runner's own check and is refused with the runner's bounds, the same
    /// refusal a too-small positive one gets.
    #[test]
    fn a_negative_count_is_refused_like_a_too_small_one() {
        let refused = |method: &str, config: InputConfig| {
            solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), method, &config)
                .expect_err("out of range")
                .fields
        };
        let expected = |parameter: &str, method: &str, min: f64, got: f64| {
            serde_json::json!({ "code": "parameter_out_of_range", "parameter": parameter,
                "method": method, "min": min, "max": null, "got": got })
        };
        for got in [-1, 1] {
            let config = InputConfig {
                population_size: Some(got),
                ..quick()
            };
            assert_eq!(
                refused("ga", config),
                expected("config.population_size", "ga", 2.0, got as f64)
            );
        }
        for got in [-1, 0] {
            let config = InputConfig {
                max_generations: Some(got),
                ..quick()
            };
            assert_eq!(
                refused("ga", config),
                expected("config.max_generations", "ga", 1.0, got as f64)
            );
            let config = InputConfig {
                max_iterations: Some(got),
                ..quick()
            };
            assert_eq!(
                refused("alns", config),
                expected("config.max_iterations", "alns", 1.0, got as f64)
            );
        }
        let err = solve(
            (0.0, 0.0),
            &ring(4),
            &fleet(&[10.0]),
            "ga",
            &InputConfig {
                population_size: Some(-1),
                ..quick()
            },
        )
        .expect_err("negative");
        assert_eq!(
            err.message,
            "method \"ga\": config.population_size must be at least 2, got -1"
        );
    }

    /// A seed has no range of its own beyond its type: 0 to 2^64 - 1.
    #[test]
    fn a_negative_seed_is_out_of_range() {
        for method in ["ga", "alns"] {
            let config = InputConfig {
                seed: Some(-1),
                ..quick()
            };
            let err = solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), method, &config)
                .expect_err("negative seed");
            assert_eq!(
                err.fields,
                serde_json::json!({ "code": "parameter_out_of_range", "parameter": "config.seed",
                    "method": method, "min": 0.0, "max": null, "got": -1.0 })
            );
            let config = InputConfig {
                seed: Some(u64::MAX.into()),
                ..quick()
            };
            assert!(
                solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), method, &config).is_ok(),
                "{method}: the largest seed"
            );
        }
        let err = whole_in::<u64>("config.seed", i128::from(u64::MAX) + 1, 0, u64::MAX.into())
            .expect_err("past u64");
        assert_eq!(err.fields["max"], serde_json::json!(u64::MAX as f64));
        assert_eq!(err.fields["min"], serde_json::json!(0.0));
    }

    /// An id is a label from 0 up; a negative one is out of that range, at
    /// the customer's position.
    #[test]
    fn a_negative_id_is_out_of_range_at_its_customer() {
        let customers = [
            customer(serde_json::json!({ "id": 1, "x": 1.0, "y": 0.0 })),
            customer(serde_json::json!({ "id": -1, "x": 0.0, "y": 1.0 })),
        ];
        let err = solve((0.0, 0.0), &customers, &[], "nn", &quick()).expect_err("negative id");
        assert_eq!(
            err.fields,
            serde_json::json!({ "code": "parameter_out_of_range", "parameter": "id",
                "index": 1, "min": 0.0, "max": null, "got": -1.0 })
        );
    }

    /// Whole numbers read as themselves, a fractional one does not read at
    /// all: whether a count is whole is a matter of its type.
    #[test]
    fn whole_numbers_read_as_themselves_and_fractions_do_not() {
        let config: InputConfig = serde_json::from_value(serde_json::json!({
            "population_size": 12, "max_generations": 3.0, "max_iterations": 7,
            "seed": u64::MAX, "max_vehicles": null,
        }))
        .expect("whole numbers");
        assert_eq!(config.population_size, Some(12));
        assert_eq!(config.max_generations, Some(3));
        assert_eq!(config.max_iterations, Some(7));
        assert_eq!(config.seed, Some(u64::MAX.into()));
        assert_eq!(config.max_vehicles, None);
        for field in [
            "population_size",
            "max_generations",
            "max_iterations",
            "seed",
            "max_vehicles",
        ] {
            let err = serde_json::from_value::<InputConfig>(serde_json::Value::Object(
                [(field.to_string(), serde_json::json!(2.5))]
                    .into_iter()
                    .collect(),
            ))
            .err()
            .expect("a fraction");
            assert!(
                err.to_string().contains("must be a whole number, got 2.5"),
                "{field}: {err}"
            );
            let err = serde_json::from_value::<InputConfig>(serde_json::Value::Object(
                [(field.to_string(), serde_json::json!("2"))]
                    .into_iter()
                    .collect(),
            ))
            .err()
            .expect("a string");
            assert!(
                err.to_string().contains("expected a whole number"),
                "{field}: {err}"
            );
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
        assert!(result.is_err(), "max_generations=0 should fail validation");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("max_generations"),
            "error should mention max_generations: {}",
            err
        );
    }

    // ---- GA: elite_ratio too high ----

    /// A ratio of 1 makes every individual an elite in any population: a
    /// range refusal whose upper end is open, not a combination.
    #[test]
    fn ga_elite_ratio_fills_population() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig {
            population_size: Some(2),
            max_generations: Some(5),
            elite_ratio: Some(1.0),
            seed: Some(42),
            ..InputConfig::default()
        };
        let err = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg)
            .expect_err("elite_ratio filling entire population");
        assert_eq!(
            err.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "config.elite_ratio",
                    "min": null, "max": null, "exclusive_min": 0.0, "exclusive_max": 1.0,
                    "got": 1.0 })
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        ] {
            let cfg = InputConfig {
                population_size: Some(10),
                max_generations: Some(5),
                mutation_rate,
                elite_ratio,
                seed: Some(42),
                ..InputConfig::default()
            };
            let err =
                solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg).expect_err("out of (0, 1]");
            // (0, 1]: the lower end is open, the upper closed.
            assert_eq!(
                err.fields,
                json!({ "code": "parameter_out_of_range", "parameter": parameter,
                        "min": null, "max": 1.0, "exclusive_min": 0.0, "got": got })
            );
        }
    }

    /// An open end the runner reports stays open on the wire: (0, 1] is
    /// `exclusive_min` beside `max`, never `min: 0`.
    #[test]
    fn runner_bounds_are_forwarded_with_their_open_ends() {
        let open = AlnsConfig::default()
            .with_reaction_factor(0.0)
            .validate()
            .expect_err("a reaction factor of zero");
        let err = settings_refused(Method::Alns, open);
        assert_eq!(
            err.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "config.reaction_factor",
                    "method": "alns", "min": null, "max": 1.0, "exclusive_min": 0.0,
                    "got": 0.0 })
        );
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
        let result = solve_alns(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        let result = solve_alns(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
        assert!(result.is_ok(), "GA with default config should succeed");
    }

    #[test]
    fn alns_default_config() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_alns(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 1000, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 5, RouteLimits::NONE, &cfg);
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
        let result = solve_ga(&customers, &dm, 100, RouteLimits::NONE, &cfg);
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
