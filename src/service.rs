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

use crate::alns::destroy::{RandomRemoval, RouteRemoval, ShawRemoval, WorstRemoval};
use crate::alns::repair::{GreedyInsertion, RegretInsertion};
use crate::alns::{Destroy, Repair, RoutingAlnsProblem};
use crate::constructive::{clarke_wright_savings, nearest_neighbor, nearest_neighbor_tw};
use crate::distance::DistanceMatrix;
use crate::evaluation::{has_time_windows, route_checks_needed, RouteEvaluator, RouteLimits};
use crate::fleet::limit_routes;
use crate::ga::RoutingGaProblem;
use crate::ga::{split, split_tw};
use crate::local_search::{or_opt_improve, two_opt_improve};
use crate::models::{Customer, TimeWindow, Vehicle};
use u_metaheur::alns::{AlnsConfig, AlnsRunner};
use u_metaheur::ga::{GaConfig, GaRunner};
use u_metaheur::{ConfigError, StopReason};

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
/// Every reason the solver refuses a request with: the `code` both bindings
/// carry, and the fields that can stand beside it. A refusal is built from one
/// of these ([`ServiceError::new`]), so [`codes`] -- the list a caller reads --
/// is the list the solver refuses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Code {
    MalformedInput,
    /// Only JavaScript can send a NaN or an infinity (JSON has none); the
    /// .NET client refuses one with this code while it assembles the request.
    ValueNotFinite,
    UnknownOption,
    InvalidOption,
    ParameterOutOfRange,
    DimensionMismatch,
    DuplicateId,
    InvalidTimeWindow,
    NotWholeUnits,
    MixedFleet,
    Internal,
}

impl Code {
    pub(crate) const ALL: [Code; 11] = [
        Code::MalformedInput,
        Code::ValueNotFinite,
        Code::UnknownOption,
        Code::InvalidOption,
        Code::ParameterOutOfRange,
        Code::DimensionMismatch,
        Code::DuplicateId,
        Code::InvalidTimeWindow,
        Code::NotWholeUnits,
        Code::MixedFleet,
        Code::Internal,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Code::MalformedInput => "malformed_input",
            Code::ValueNotFinite => "value_not_finite",
            Code::UnknownOption => "unknown_option",
            Code::InvalidOption => "invalid_option",
            Code::ParameterOutOfRange => "parameter_out_of_range",
            Code::DimensionMismatch => "dimension_mismatch",
            Code::DuplicateId => "duplicate_id",
            Code::InvalidTimeWindow => "invalid_time_window",
            Code::NotWholeUnits => "not_whole_units",
            Code::MixedFleet => "mixed_fleet",
            Code::Internal => "internal",
        }
    }

    /// The fields a refusal with this code can carry besides `code`.
    pub(crate) fn fields(self) -> &'static [&'static str] {
        match self {
            Code::MalformedInput | Code::ValueNotFinite => &["parameter", "index"],
            Code::UnknownOption => &["parameter", "got", "expected"],
            Code::InvalidOption => &["parameter", "method"],
            Code::ParameterOutOfRange => &[
                "parameter",
                "index",
                "id",
                "method",
                "min",
                "max",
                "exclusive_min",
                "exclusive_max",
                "got",
            ],
            Code::DimensionMismatch => &["parameter", "expected", "got"],
            Code::DuplicateId => &["id", "first", "second"],
            Code::InvalidTimeWindow => &["id", "index", "ready", "due"],
            Code::NotWholeUnits => &["parameter", "index", "id", "value"],
            Code::MixedFleet => &["method", "parameter", "values", "capacities", "index"],
            Code::Internal => &[],
        }
    }
}

#[derive(Debug)]
pub(crate) struct ServiceError {
    pub(crate) message: String,
    pub(crate) fields: serde_json::Value,
}

impl ServiceError {
    fn new(code: Code, message: String, mut fields: serde_json::Value) -> Self {
        if let Some(extra) = fields.as_object() {
            debug_assert!(
                extra.keys().all(|k| code.fields().contains(&k.as_str())),
                "{} refusal with undeclared fields: {:?}",
                code.name(),
                extra.keys().collect::<Vec<_>>()
            );
        }
        let mut all = serde_json::Map::new();
        all.insert("code".into(), json!(code.name()));
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
            Code::MalformedInput,
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A NaN or ±Infinity where a finite number belongs; `fields` carries
    /// `parameter` and `index`. Only JavaScript can send one: JSON has none.
    #[cfg(feature = "wasm")]
    pub(crate) fn value_not_finite(message: String, fields: serde_json::Value) -> Self {
        Self::new(Code::ValueNotFinite, message, fields)
    }

    /// A failure inside the library rather than a refusal of the input.
    pub(crate) fn internal(message: &str) -> Self {
        Self::new(Code::Internal, message.to_string(), json!({}))
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
        Code::ParameterOutOfRange,
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

/// The time limit from `config`, for the methods that read one: a whole
/// number of milliseconds from 1 to 2^64 - 1.
fn time_limit(method: Method, cfg: &InputConfig) -> Result<Option<u64>, ServiceError> {
    cfg.time_limit_ms
        .map(|ms| {
            whole_in("config.time_limit_ms", ms, 1, u64::MAX.into()).map_err(|mut e| {
                e.fields["method"] = json!(method.name());
                e.message = format!("method \"{}\": {}", method.name(), e.message);
                e
            })
        })
        .transpose()
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
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
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
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
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
    /// Cost of using this vehicle for a route, whatever its length (`>= 0`,
    /// default 0). Every method minimizes the plan's `total_cost`, so a fixed
    /// cost trades distance for fewer routes.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    fixed_cost: f64,
    /// Cost per unit of distance this vehicle travels (`> 0`, default 1).
    #[serde(default = "default_cost_per_distance")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    cost_per_distance: f64,
}

/// No limit: the most the model stores, which is also the most one customer
/// may ask for -- so a vehicle without a capacity takes any customer, as the
/// vehicle of a request with no `vehicles` does.
fn default_capacity() -> f64 {
    f64::from(i32::MAX)
}

fn default_cost_per_distance() -> f64 {
    1.0
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
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
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
    /// Wall-clock budget in milliseconds for GA and ALNS: a whole number, at
    /// least 1. A run that spends it returns the best plan found by then,
    /// `stop_reason: "time_limit"`. Unset, a run ends on its own budget.
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    time_limit_ms: Option<i128>,
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

/// One route's figures, as the evaluator reads the route: the depot at both
/// ends, the route's vehicle, the customers' service times and time windows.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
pub(crate) struct RouteSummary {
    /// The route's length, depot to depot.
    pub(crate) distance: f64,
    /// The demand the route carries: the sum of its customers' demands.
    pub(crate) load: i64,
    /// The route's vehicle's `fixed_cost` plus `cost_per_distance` × `distance`.
    pub(crate) cost: f64,
    /// When the route is back at the depot, leaving it at time 0: travel
    /// (one unit of distance is one unit of time), waiting for time windows to
    /// open and service times. Equal to `distance` for a route with neither.
    pub(crate) duration: f64,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
pub(crate) struct VrpOutput {
    pub(crate) routes: Vec<Vec<usize>>,
    /// The vehicle that drives each route: its index in the request's
    /// `vehicles`, in the order of `routes`. `"nn"` gives each route its own
    /// vehicle from the list; `"savings"`, `"ga"` and `"alns"` drive every
    /// route with the request's single vehicle type, `0`. A request with no
    /// `vehicles` is solved with one vehicle of unlimited capacity, also `0`.
    pub(crate) route_vehicles: Vec<usize>,
    /// What each route comes to, in the order of `routes` -- the figures the
    /// plan's totals are sums of: `total_distance` is the sum of their
    /// `distance` and `total_cost` of their `cost`.
    pub(crate) route_summaries: Vec<RouteSummary>,
    pub(crate) total_distance: f64,
    /// What the plan costs: over its routes, the route's vehicle's
    /// `fixed_cost` plus `cost_per_distance` × the route's distance. Equal to
    /// `total_distance` with the default costs.
    pub(crate) total_cost: f64,
    pub(crate) num_vehicles: usize,
    pub(crate) method_used: String,
    /// Filled in by the binding: how to read a clock differs by target.
    pub(crate) computation_time_ms: f64,
    /// Customers (by input id) that no route serves -- for example when the
    /// fleet cannot carry the total demand. They would otherwise simply be
    /// missing from `routes`, and a partial plan would read as a complete one.
    pub(crate) unassigned: Vec<usize>,
    /// Why the method stopped: `"completed"` for `"nn"` and `"savings"`, which
    /// build the plan in one pass; `"iteration_limit"` when `"ga"` or `"alns"`
    /// spent its generation or iteration budget; `"no_improvement"` when
    /// `"ga"` went its stall budget without a better plan; `"time_limit"`
    /// when `config.time_limit_ms` ran out first.
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "\"completed\" | \"optimal\" | \"target_reached\" | \"iteration_limit\" | \"no_improvement\" | \"time_limit\" | \"cancelled\""
        )
    )]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "schema::stop_reason")
    )]
    pub(crate) stop_reason: StopReason,
}

// ============================================================================
// Methods
// ============================================================================

/// The settings a request's `config` can carry, in the order of
/// [`InputConfig`]'s fields. Which of them a method reads is
/// [`Method::reads_setting`].
pub(crate) const SETTINGS: [&str; 8] = [
    "population_size",
    "max_generations",
    "mutation_rate",
    "elite_ratio",
    "max_iterations",
    "seed",
    "time_limit_ms",
    "max_vehicles",
];

/// A refusal reason: the `code` and the fields that can stand beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RefusalCode {
    pub(crate) code: String,
    pub(crate) fields: Vec<String>,
}

/// A violation the checker reports: its `code`, the fields beside it, and the
/// functions that report it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct ViolationCode {
    pub(crate) code: String,
    pub(crate) fields: Vec<String>,
    #[cfg_attr(feature = "wasm", tsify(type = "\"check_vrp_solution\"[]"))]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "schema::checkers")
    )]
    pub(crate) checkers: Vec<String>,
}

/// What the solver can answer with besides a result: every refusal `code`
/// and every checker violation `code`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct Codes {
    pub(crate) refusals: Vec<RefusalCode>,
    pub(crate) violations: Vec<ViolationCode>,
}

/// The vocabularies, from the tables the refusals and violations are built
/// from: [`Code`] and [`PlanViolation::examples`].
pub(crate) fn codes() -> Codes {
    Codes {
        refusals: Code::ALL
            .into_iter()
            .map(|c| RefusalCode {
                code: c.name().to_string(),
                fields: c.fields().iter().map(|f| f.to_string()).collect(),
            })
            .collect(),
        violations: PlanViolation::examples()
            .iter()
            .map(|v| {
                let value = serde_json::to_value(v).expect("a violation serializes");
                let object = value.as_object().expect("a violation is an object");
                ViolationCode {
                    code: object["code"].as_str().expect("tagged").to_string(),
                    fields: object.keys().filter(|k| *k != "code").cloned().collect(),
                    checkers: vec!["check_vrp_solution".to_string()],
                }
            })
            .collect(),
    }
}

/// One method and the `config` settings it reads; a request giving any other
/// setting with this method is refused (`invalid_option`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct MethodInfo {
    /// The method, as a request's `method` names it.
    #[cfg_attr(
        feature = "wasm",
        tsify(type = "\"nn\" | \"savings\" | \"ga\" | \"alns\"")
    )]
    #[cfg_attr(all(test, feature = "wasm"), schemars(schema_with = "schema::method"))]
    pub(crate) name: String,
    /// The settings it reads, in [`SETTINGS`] order.
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "(\"population_size\" | \"max_generations\" | \"mutation_rate\" | \"elite_ratio\" | \"max_iterations\" | \"seed\" | \"time_limit_ms\" | \"max_vehicles\")[]"
        )
    )]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "schema::settings")
    )]
    pub(crate) reads: Vec<String>,
}

/// Every method, with the settings each reads -- made from the same table as
/// the refusal of a setting the method does not read, so the two cannot
/// disagree.
pub(crate) fn methods() -> Vec<MethodInfo> {
    Method::ALL
        .into_iter()
        .map(|m| MethodInfo {
            name: m.name().to_string(),
            reads: SETTINGS
                .into_iter()
                .filter(|s| m.reads_setting(s))
                .map(String::from)
                .collect(),
        })
        .collect()
}

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

    /// Whether this method reads `setting` (one of [`SETTINGS`]). This is the
    /// one table both the refusal of a setting the method does not read and
    /// the published catalog ([`methods`]) come from.
    ///
    /// - `population_size`, `max_generations`, `mutation_rate`, `elite_ratio`:
    ///   the genetic algorithm.
    /// - `max_iterations`: ALNS.
    /// - `seed`, `time_limit_ms`: the two search methods (GA, ALNS).
    /// - `max_vehicles`: every method -- the fixed fleet all of them keep to.
    fn reads_setting(self, setting: &str) -> bool {
        use Method::*;
        match setting {
            "population_size" | "max_generations" | "mutation_rate" | "elite_ratio" => {
                self == Genetic
            }
            "max_iterations" => self == Alns,
            "seed" | "time_limit_ms" => matches!(self, Genetic | Alns),
            "max_vehicles" => true,
            _ => false,
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
                    Code::UnknownOption,
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
    refuse_unread_settings(method, config)?;
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
    let vehicles = build_vehicles(input_vehicles)?;
    // The largest load any vehicle takes: a customer asking for more rides on
    // no route at all, whatever the method -- reported unassigned up front
    // rather than left for each method to notice (savings and ALNS opened a
    // route per customer and never looked; GA's split folded the tour into one).
    let largest_load = vehicles
        .iter()
        .map(Vehicle::capacity)
        .max()
        .expect("build_vehicles returns at least one vehicle");
    let BuiltCustomers {
        customers,
        id_map,
        all_ids,
    } = build_customers(depot, input_customers, largest_load)?;
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
            route_vehicles: vec![],
            route_summaries: vec![],
            total_distance: 0.0,
            total_cost: 0.0,
            num_vehicles: 0,
            method_used: method.name().to_string(),
            computation_time_ms: 0.0,
            unassigned: all_ids,
            stop_reason: StopReason::Completed,
        });
    }

    let dm = DistanceMatrix::from_customers(&customers);

    let plan = match method {
        Method::NearestNeighbor => solve_nn(&customers, &dm, &vehicles),
        Method::Savings => solve_savings(&customers, &dm, &vehicles),
        Method::Genetic => solve_ga(&customers, &dm, &vehicles[0], config)?,
        Method::Alns => solve_alns(&customers, &dm, &vehicles[0], config)?,
    };
    // A fixed fleet is kept by every method alike: `"nn"` cannot exceed its
    // vehicle list, but it can exceed a smaller `max_vehicles`.
    let plan = match max_vehicles {
        Some(max) if plan.routes.len() > max => {
            let limited = limit_routes(plan.routes, plan.vehicles, &vehicles, max, &dm, &customers);
            let limits: Vec<RouteLimits> = limited
                .vehicles
                .iter()
                .map(|&v| RouteLimits::of(&vehicles[v]))
                .collect();
            let (routes, _) = apply_local_search(&limited.routes, &limits, &dm, &customers);
            Plan {
                routes,
                vehicles: limited.vehicles,
                method,
                stop_reason: plan.stop_reason,
            }
        }
        _ => plan,
    };
    let route_summaries: Vec<RouteSummary> = plan
        .routes
        .iter()
        .zip(&plan.vehicles)
        .map(|(route, &v)| {
            let (built, _) = RouteEvaluator::new(&customers, &dm, &vehicles[v]).build_route(route);
            let distance = route_distance(route, &dm);
            RouteSummary {
                distance,
                load: built.total_load().into(),
                cost: vehicles[v].route_cost(distance),
                duration: built.total_duration(),
            }
        })
        .collect();
    let total_cost = route_summaries.iter().map(|r| r.cost).sum();
    let total_distance = route_summaries.iter().map(|r| r.distance).sum();
    let routes = map_routes(&plan.routes, id_map.as_slice());
    // Derived from the routes rather than from each solver's own bookkeeping,
    // so it holds for every method alike.
    let served: std::collections::HashSet<usize> = routes.iter().flatten().copied().collect();
    let unassigned = all_ids
        .iter()
        .copied()
        .filter(|id| !served.contains(id))
        .collect();
    Ok(VrpOutput {
        num_vehicles: routes.len(),
        total_distance,
        total_cost,
        routes,
        route_vehicles: plan.vehicles,
        route_summaries,
        method_used: plan.method.name().to_string(),
        computation_time_ms: 0.0,
        unassigned,
        stop_reason: plan.stop_reason,
    })
}

// ============================================================================
// Checking a plan
// ============================================================================

/// A plan to check, in the shape `solve` returns one: customer ids per route
/// (the depot implied at both ends), the vehicle of each route, the
/// customers reported unassigned and, optionally, the total distance reported.
#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputSolution {
    /// Customer ids per route, as `solve` reports them.
    #[cfg_attr(feature = "wasm", tsify(type = "number[][]"))]
    routes: Vec<Vec<i128>>,
    /// The vehicle of each route: its index in the request's `vehicles`.
    /// Absent: `0` for every route.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number[] | null"))]
    route_vehicles: Option<Vec<i128>>,
    /// Customers reported as served by no route.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number[]"))]
    unassigned: Vec<i128>,
    /// The plan's total distance as reported. Absent: not checked.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    total_distance: Option<f64>,
    /// The plan's total cost as reported (see `solve`'s `total_cost`).
    /// Absent: not checked.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    total_cost: Option<f64>,
}

/// One way a plan breaks its problem. `route` is an index into the plan's
/// `routes`; `customer` and `vehicle` are as the request and plan name them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(tag = "code", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum PlanViolation {
    /// The route carries more than its vehicle's capacity.
    CapacityExceeded {
        route: usize,
        load: i64,
        capacity: i64,
    },
    /// The route reaches `customer` after its window closes.
    TimeWindowViolated {
        route: usize,
        customer: i64,
        arrival: f64,
        due: f64,
    },
    /// The route is longer than its vehicle's `max_distance`.
    MaxDistanceExceeded {
        route: usize,
        distance: f64,
        max_distance: f64,
    },
    /// The route returns later than its vehicle's `max_duration`.
    MaxDurationExceeded {
        route: usize,
        duration: f64,
        max_duration: f64,
    },
    /// The route visits an id that is no customer of the request.
    UnknownCustomer { route: usize, customer: i64 },
    /// `customer` is visited a second time, on `route`.
    CustomerVisitedTwice { route: usize, customer: i64 },
    /// `customer` is on no route and not in `unassigned`.
    CustomerMissing { customer: i64 },
    /// `customer` is in `unassigned` but a route serves it, it is listed
    /// twice, or it is no customer of the request.
    WronglyUnassigned { customer: i64 },
    /// The route's vehicle is not in the request's `vehicles`.
    UnknownVehicle { route: usize, vehicle: i64 },
    /// `"nn"` drives each vehicle once; this route uses one a second time.
    VehicleReused { route: usize, vehicle: i64 },
    /// More routes than `config.max_vehicles`.
    TooManyRoutes { routes: usize, max_routes: usize },
    /// The reported `total_distance` differs from the routes' own.
    DistanceMismatch { reported: f64, actual: f64 },
    /// The reported `total_cost` differs from what the routes cost with
    /// their vehicles.
    CostMismatch { reported: f64, actual: f64 },
}

impl PlanViolation {
    /// One of each violation, for the published vocabulary ([`codes`]): its
    /// code and fields are read off these, so they are the enum's.
    pub(crate) fn examples() -> Vec<Self> {
        vec![
            Self::CapacityExceeded {
                route: 0,
                load: 0,
                capacity: 0,
            },
            Self::TimeWindowViolated {
                route: 0,
                customer: 0,
                arrival: 0.0,
                due: 0.0,
            },
            Self::MaxDistanceExceeded {
                route: 0,
                distance: 0.0,
                max_distance: 0.0,
            },
            Self::MaxDurationExceeded {
                route: 0,
                duration: 0.0,
                max_duration: 0.0,
            },
            Self::UnknownCustomer {
                route: 0,
                customer: 0,
            },
            Self::CustomerVisitedTwice {
                route: 0,
                customer: 0,
            },
            Self::CustomerMissing { customer: 0 },
            Self::WronglyUnassigned { customer: 0 },
            Self::UnknownVehicle {
                route: 0,
                vehicle: 0,
            },
            Self::VehicleReused {
                route: 0,
                vehicle: 0,
            },
            Self::TooManyRoutes {
                routes: 0,
                max_routes: 0,
            },
            Self::DistanceMismatch {
                reported: 0.0,
                actual: 0.0,
            },
            Self::CostMismatch {
                reported: 0.0,
                actual: 0.0,
            },
        ]
    }
}

/// Whether a plan keeps every rule of its problem, and each rule it breaks.
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
pub(crate) struct CheckOutput {
    /// No violations.
    pub(crate) feasible: bool,
    /// Every rule the plan breaks; empty when `feasible`.
    pub(crate) violations: Vec<PlanViolation>,
}

/// Checks `solution` against the problem a `solve` request states, read as
/// `solve` reads it -- the same `method` (it decides whether each vehicle
/// drives one route, `"nn"`, or is a type any number of routes use) and the
/// same refusals. `config.max_vehicles` bounds the routes; the method's search
/// settings change nothing here but are judged as `solve` judges them, so a
/// request `solve` takes is a request `check` takes.
pub(crate) fn check(
    depot: (f64, f64),
    input_customers: &[InputCustomer],
    input_vehicles: &[InputVehicle],
    method: &str,
    config: &InputConfig,
    solution: &InputSolution,
) -> Result<CheckOutput, ServiceError> {
    use crate::evaluation::{check_solution, Fleet, VehicleUse};
    use crate::models::{Route, Solution, ViolationType, Visit};

    let method = Method::parse(method)?;
    refuse_unread_settings(method, config)?;
    let max_routes = config
        .max_vehicles
        .map(|n| whole_in::<usize>("config.max_vehicles", n, 1, usize::MAX as i128))
        .transpose()?;
    let BuiltCustomers {
        customers, id_map, ..
    } = build_customers(depot, input_customers, i32::MAX)?;
    let vehicles = build_vehicles(input_vehicles)?;
    if method != Method::NearestNeighbor {
        refuse_mixed_fleet(method, &vehicles)?;
    }
    if let Some(route_vehicles) = &solution.route_vehicles {
        if route_vehicles.len() != solution.routes.len() {
            return Err(ServiceError::new(
                Code::DimensionMismatch,
                format!(
                    "solution.route_vehicles has {} entries for {} routes -- one per route",
                    route_vehicles.len(),
                    solution.routes.len()
                ),
                json!({ "parameter": "solution.route_vehicles",
                        "expected": solution.routes.len(), "got": route_vehicles.len() }),
            ));
        }
    }

    // Caller ids <-> internal indices. An id no customer has gets an index
    // past the last customer, which the checker reports as unknown.
    let internal_of: std::collections::HashMap<i128, usize> = id_map
        .iter()
        .enumerate()
        .map(|(i, &id)| (id as i128, i + 1))
        .collect();
    let mut caller_of: Vec<i128> = std::iter::once(-1)
        .chain(id_map.iter().map(|&id| id as i128))
        .collect();
    let mut internal = |id: i128| -> usize {
        internal_of.get(&id).copied().unwrap_or_else(|| {
            caller_of.push(id);
            caller_of.len() - 1
        })
    };

    let dm = DistanceMatrix::from_customers(&customers);
    let mut plan = Solution::new();
    let mut actual_total = Some(0.0);
    let mut actual_cost = Some(0.0);
    for (r, ids) in solution.routes.iter().enumerate() {
        let vehicle = solution
            .route_vehicles
            .as_ref()
            .map_or(0, |v| v[r])
            .try_into()
            .unwrap_or(usize::MAX);
        let mut route = Route::new(vehicle);
        let internal_ids: Vec<usize> = ids.iter().map(|&id| internal(id)).collect();
        for &customer_id in &internal_ids {
            route.push_visit(Visit {
                customer_id,
                arrival_time: 0.0,
                departure_time: 0.0,
                load_after: 0,
            });
        }
        // The distance is the route's own (the plan reports only a total, which
        // is compared below), so the checker finds no per-route mismatch.
        if internal_ids.iter().all(|&c| c < customers.len()) {
            let (mut prev, mut d) = (0, 0.0);
            for &c in &internal_ids {
                d += dm.get(prev, c);
                prev = c;
            }
            d += dm.get(prev, 0);
            route.set_total_distance(d);
            actual_total = actual_total.map(|t| t + d);
            actual_cost = match (actual_cost, vehicles.get(vehicle)) {
                (Some(c), Some(v)) => Some(c + v.route_cost(d)),
                _ => None,
            };
        } else {
            actual_total = None;
            actual_cost = None;
        }
        plan.add_route(route);
    }
    for &id in &solution.unassigned {
        let i = internal(id);
        plan.add_unassigned(i);
    }

    let fleet = Fleet {
        vehicles: &vehicles,
        usage: if method == Method::NearestNeighbor {
            VehicleUse::OneRouteEach
        } else {
            VehicleUse::AnyNumber
        },
        max_routes,
    };
    let caller = |internal: usize| -> i64 { caller_of[internal] as i64 };
    let vehicle_given = |route: usize| -> i64 {
        solution
            .route_vehicles
            .as_ref()
            .map_or(0, |v| v[route] as i64)
    };
    let mut violations: Vec<PlanViolation> = check_solution(&customers, &dm, &fleet, &plan)
        .into_iter()
        .filter_map(|v| {
            Some(match v.kind {
                ViolationType::CapacityExceeded {
                    route_index,
                    load,
                    capacity,
                } => PlanViolation::CapacityExceeded {
                    route: route_index,
                    load: load.into(),
                    capacity: capacity.into(),
                },
                ViolationType::TimeWindowViolated {
                    route_index,
                    customer_id,
                    arrival,
                    due,
                } => PlanViolation::TimeWindowViolated {
                    route: route_index,
                    customer: caller(customer_id),
                    arrival,
                    due,
                },
                ViolationType::MaxDistanceExceeded {
                    route_index,
                    distance,
                    max_distance,
                } => PlanViolation::MaxDistanceExceeded {
                    route: route_index,
                    distance,
                    max_distance,
                },
                ViolationType::MaxDurationExceeded {
                    route_index,
                    duration,
                    max_duration,
                } => PlanViolation::MaxDurationExceeded {
                    route: route_index,
                    duration,
                    max_duration,
                },
                ViolationType::UnknownCustomer {
                    route_index,
                    customer_id,
                } => PlanViolation::UnknownCustomer {
                    route: route_index,
                    customer: caller(customer_id),
                },
                ViolationType::CustomerVisitedTwice {
                    customer_id,
                    route_index,
                } => PlanViolation::CustomerVisitedTwice {
                    route: route_index,
                    customer: caller(customer_id),
                },
                ViolationType::CustomerMissing { customer_id } => PlanViolation::CustomerMissing {
                    customer: caller(customer_id),
                },
                ViolationType::WronglyUnassigned { customer_id } => {
                    PlanViolation::WronglyUnassigned {
                        customer: caller(customer_id),
                    }
                }
                ViolationType::UnknownVehicle { route_index, .. } => {
                    PlanViolation::UnknownVehicle {
                        route: route_index,
                        vehicle: vehicle_given(route_index),
                    }
                }
                ViolationType::VehicleReused { route_index, .. } => PlanViolation::VehicleReused {
                    route: route_index,
                    vehicle: vehicle_given(route_index),
                },
                ViolationType::TooManyRoutes { routes, max_routes } => {
                    PlanViolation::TooManyRoutes { routes, max_routes }
                }
                // Each route's distance was set to its own above.
                ViolationType::DistanceMismatch { .. } => return None,
            })
        })
        .collect();
    if let (Some(reported), Some(actual)) = (solution.total_distance, actual_total) {
        if (reported - actual).abs() > 1e-6 * actual.abs().max(1.0) {
            violations.push(PlanViolation::DistanceMismatch { reported, actual });
        }
    }
    if let (Some(reported), Some(actual)) = (solution.total_cost, actual_cost) {
        if (reported - actual).abs() > 1e-6 * actual.abs().max(1.0) {
            violations.push(PlanViolation::CostMismatch { reported, actual });
        }
    }
    Ok(CheckOutput {
        feasible: violations.is_empty(),
        violations,
    })
}

/// A method's plan in internal customer indices, before it is mapped to the
/// caller's ids.
#[derive(Debug)]
struct Plan {
    routes: Vec<Vec<usize>>,
    /// The vehicle of each route -- its index in the request's fleet -- in the
    /// same order. Capacity and limits come from the vehicle.
    vehicles: Vec<usize>,
    method: Method,
    /// Why the method stopped.
    stop_reason: StopReason,
}

// ============================================================================
// Internal helpers
// ============================================================================
/// The customers as the solvers see them, and the ids that name them.
struct BuiltCustomers {
    /// Depot first, then every customer a vehicle can carry.
    customers: Vec<Customer>,
    /// The caller's id of `customers[i + 1]`.
    id_map: Vec<usize>,
    /// Every customer's id, in the order given -- what `unassigned` is read from.
    all_ids: Vec<usize>,
}

/// Builds the internal customer list and ID mapping from input, leaving out
/// (but still validating) every customer asking for more than `largest_load`.
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
    largest_load: i32,
) -> Result<BuiltCustomers, ServiceError> {
    let mut customers: Vec<Customer> = Vec::with_capacity(input_customers.len() + 1);
    customers.push(Customer::depot(depot.0, depot.1));

    // `id_map` names the customers a solver sees (every one a vehicle can
    // carry); `all_ids` names all of them, in the order given.
    let mut id_map: Vec<usize> = Vec::with_capacity(input_customers.len());
    let mut all_ids: Vec<usize> = Vec::with_capacity(input_customers.len());
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
                Code::DuplicateId,
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
        all_ids.push(id);
        let idx = customers.len();
        let mut c = Customer::new(idx, ic.x, ic.y, demand, ic.service_time);
        if let Some([ready, due]) = ic.time_window {
            let tw = TimeWindow::new(ready, due).ok_or_else(|| {
                ServiceError::new(
                    Code::InvalidTimeWindow,
                    format!(
                        "customer {id}: time_window [{ready}, {due}] is not a window \
                         (it needs ready <= due)"
                    ),
                    json!({ "id": id, "index": position, "ready": ready, "due": due }),
                )
            })?;
            c = c.with_time_window(tw);
        }
        if demand <= largest_load {
            id_map.push(id);
            customers.push(c);
        }
    }

    // The model adds loads in `i32`. Every route's load is part of the sum of
    // the demands of the customers a solver sees, so that sum fitting is what
    // keeps every addition in range -- a vehicle without a capacity (the most
    // the model stores) included. One more than that is refused here, and not
    // left to wrap in a build without overflow checks.
    let total: f64 = customers.iter().map(|c| f64::from(c.demand())).sum();
    let accepted = 0.0..=f64::from(i32::MAX);
    if !accepted.contains(&total) {
        let mut fields = json!({ "parameter": "total_demand", "got": total });
        Bounds::of(accepted).insert_into(&mut fields);
        return Err(ServiceError::new(
            Code::ParameterOutOfRange,
            format!(
                "the demands of the customers a vehicle can carry add up to {total}; the                  model counts a route's load as a whole number of units up to {}, so they                  must add up to no more -- use a larger unit (kilograms for grams, say)",
                i32::MAX
            ),
            fields,
        ));
    }

    Ok(BuiltCustomers {
        customers,
        id_map,
        all_ids,
    })
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
            ServiceError::new(Code::ParameterOutOfRange, message, fields)
        }
        _ => ServiceError::new(
            Code::InvalidOption,
            message,
            json!({ "parameter": parameter, "method": method.name() }),
        ),
    }
}

/// Refuses a setting the method does not read, naming the methods that do.
/// Run without it, the request would claim a setting that changed nothing; it
/// used to be accepted and ignored.
fn refuse_unread_settings(method: Method, cfg: &InputConfig) -> Result<(), ServiceError> {
    let given = [
        cfg.population_size.is_some(),
        cfg.max_generations.is_some(),
        cfg.mutation_rate.is_some(),
        cfg.elite_ratio.is_some(),
        cfg.max_iterations.is_some(),
        cfg.seed.is_some(),
        cfg.time_limit_ms.is_some(),
        cfg.max_vehicles.is_some(),
    ];
    for (setting, present) in SETTINGS.into_iter().zip(given) {
        if present && !method.reads_setting(setting) {
            let names: Vec<&str> = Method::ALL
                .into_iter()
                .filter(|m| m.reads_setting(setting))
                .map(Method::name)
                .collect();
            return Err(ServiceError::new(
                Code::InvalidOption,
                format!(
                    "config.{setting} is read by {} -- method \"{}\" does not read it; leave it out",
                    names.join(", "),
                    method.name()
                ),
                json!({ "parameter": format!("config.{setting}"), "method": method.name() }),
            ));
        }
    }
    Ok(())
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

/// Depot, `route`'s customers in order, depot: the length of the round trip.
fn route_distance(route: &[usize], dm: &DistanceMatrix) -> f64 {
    let mut prev = 0;
    let mut d = 0.0;
    for &c in route {
        d += dm.get(prev, c);
        prev = c;
    }
    d + dm.get(prev, 0)
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
            Code::NotWholeUnits,
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
            Code::ParameterOutOfRange,
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
/// capacity, `fixed_cost`, `cost_per_distance`, `max_distance` or
/// `max_duration`, naming that field.
fn refuse_mixed_fleet(method: Method, vehicles: &[Vehicle]) -> Result<(), ServiceError> {
    let first = &vehicles[0];
    for (index, other) in vehicles.iter().enumerate().skip(1) {
        let differs = [
            ("capacity", json!(first.capacity()), json!(other.capacity())),
            (
                "fixed_cost",
                json!(first.fixed_cost()),
                json!(other.fixed_cost()),
            ),
            (
                "cost_per_distance",
                json!(first.cost_per_distance()),
                json!(other.cost_per_distance()),
            ),
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
                Code::MixedFleet,
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
                        Code::ParameterOutOfRange,
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
            // JSON has no infinity or NaN: only the range is left to check.
            if v.fixed_cost < 0.0 {
                let mut fields = json!({ "parameter": "fixed_cost", "index": i, "got": v.fixed_cost });
                Bounds::of((Bound::Included(0.0), Bound::Unbounded)).insert_into(&mut fields);
                return Err(ServiceError::new(
                    Code::ParameterOutOfRange,
                    format!("vehicle {i}: fixed_cost must be >= 0, got {}", v.fixed_cost),
                    fields,
                ));
            }
            if v.cost_per_distance <= 0.0 {
                let mut fields =
                    json!({ "parameter": "cost_per_distance", "index": i, "got": v.cost_per_distance });
                Bounds::of((Bound::Excluded(0.0), Bound::Unbounded)).insert_into(&mut fields);
                return Err(ServiceError::new(
                    Code::ParameterOutOfRange,
                    format!(
                        "vehicle {i}: cost_per_distance must be > 0, got {}",
                        v.cost_per_distance
                    ),
                    fields,
                ));
            }
            Ok(vehicle
                .with_fixed_cost(v.fixed_cost)
                .with_cost_per_distance(v.cost_per_distance))
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
        // Nearest neighbour drives each route with its own vehicle.
        vehicles: solution.routes().iter().map(|r| r.vehicle_id()).collect(),
        method: Method::NearestNeighbor,
        stop_reason: StopReason::Completed,
    }
}

fn solve_savings(customers: &[Customer], dm: &DistanceMatrix, vehicles: &[Vehicle]) -> Plan {
    let vehicle_template = &vehicles[0];
    let solution = clarke_wright_savings(customers, dm, vehicle_template);
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    Plan {
        vehicles: vec![0; routes.len()],
        routes,
        method: Method::Savings,
        stop_reason: StopReason::Completed,
    }
}

/// GA with every route planned by `vehicle` (one vehicle type): its
/// capacity and limits bound each route, its fixed cost is paid per route.
fn solve_ga(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicle: &Vehicle,
    cfg: &InputConfig,
) -> Result<Plan, ServiceError> {
    let (capacity, limits) = (vehicle.capacity(), RouteLimits::of(vehicle));
    let route_cost = vehicle.fixed_cost_in_distance();
    let problem = RoutingGaProblem::new(customers.to_vec(), dm.clone(), capacity)
        .with_limits(limits)
        .with_route_cost(route_cost);

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
                Code::ParameterOutOfRange,
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
    if let Some(ms) = time_limit(method, cfg)? {
        ga_config = ga_config.with_time_limit_ms(ms);
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
        split_tw(tour, customers, dm, capacity, &limits, route_cost).routes
    } else {
        split(tour, customers, dm, capacity, route_cost).routes
    };
    let route_limits = vec![limits; routes.len()];
    let (routes, _) = apply_local_search(&routes, &route_limits, dm, customers);
    Ok(Plan {
        vehicles: vec![0; routes.len()],
        routes,
        method: Method::Genetic,
        stop_reason: ga_result.stop_reason,
    })
}

/// ALNS with every route planned by `vehicle` (one vehicle type), as
/// [`solve_ga`].
fn solve_alns(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicle: &Vehicle,
    cfg: &InputConfig,
) -> Result<Plan, ServiceError> {
    let (capacity, limits) = (vehicle.capacity(), RouteLimits::of(vehicle));
    let problem = RoutingAlnsProblem::new(customers.to_vec(), dm.clone(), capacity)
        .with_limits(limits)
        .with_route_cost(vehicle.fixed_cost_in_distance());

    // The operator set of Ropke & Pisinger (2006): the search adapts among
    // random, worst and related (Shaw) removal and greedy and regret-2
    // insertion. Every insertion keeps time windows and the route limits.
    let destroy_ops = vec![
        Destroy::Random(RandomRemoval),
        Destroy::Worst(WorstRemoval::new(dm.clone())),
        Destroy::Shaw(ShawRemoval::new(dm.clone(), customers.to_vec())),
        Destroy::Route(RouteRemoval),
    ];
    let route_cost = vehicle.fixed_cost_in_distance();
    let repair_ops = vec![
        Repair::Greedy(
            GreedyInsertion::new(dm.clone(), customers.to_vec(), capacity)
                .with_limits(limits)
                .with_route_cost(route_cost),
        ),
        Repair::Regret(
            RegretInsertion::new(dm.clone(), customers.to_vec(), capacity)
                .with_limits(limits)
                .with_route_cost(route_cost)
                .with_k(2),
        ),
    ];

    let method = Method::Alns;
    let max_iterations = match cfg.max_iterations {
        Some(n) => runner_count(method, "config.max_iterations", n, |n| {
            AlnsConfig::default().with_max_iterations(n).validate()
        })?,
        None => 500,
    };
    // Temperatures are multiples of the typical worsening of a candidate, so the
    // distance unit of the instance does not matter; the run starts where that
    // is accepted one time in three and cools geometrically to a hundredth of the
    // start exactly when the iteration budget ends -- a schedule that never
    // cools leaves the search a random walk to the last iteration.
    let mut alns_config = AlnsConfig::default()
        .with_max_iterations(max_iterations)
        .with_relative_temperature(1.0, alns_cooling_rate(max_iterations, 0.01), 0.01, 20);
    if let Some(seed) = seed(method, cfg)? {
        alns_config = alns_config.with_seed(seed);
    }
    if let Some(ms) = time_limit(method, cfg)? {
        alns_config = alns_config.with_time_limit_ms(ms);
    }

    alns_config
        .validate()
        .map_err(|e| settings_refused(Method::Alns, e))?;

    let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &alns_config)
        .map_err(|e| settings_refused(Method::Alns, e))?;

    // Apply local search to improve ALNS result
    let alns_routes: Vec<Vec<usize>> = result.best.routes().to_vec();
    let route_limits = vec![limits; alns_routes.len()];
    let (routes, _) = apply_local_search(&alns_routes, &route_limits, dm, customers);
    Ok(Plan {
        vehicles: vec![0; routes.len()],
        routes,
        method: Method::Alns,
        stop_reason: result.stop_reason,
    })
}

// ============================================================================
// Tests (native — exercise the solver functions directly)
// ============================================================================

/// The geometric cooling rate that takes a temperature to `floor` times its
/// start over `iterations` iterations.
fn alns_cooling_rate(iterations: usize, floor: f64) -> f64 {
    floor.powf(1.0 / iterations.max(1) as f64)
}

#[cfg(test)]
mod tests {
    /// The cooling reaches the floor exactly when the budget ends, and is a
    /// rate the settings check accepts for any budget.
    #[test]
    fn alns_cooling_reaches_its_floor_at_the_end_of_the_budget() {
        for iterations in [1usize, 2, 500, 10_000, 1_000_000_000] {
            let rate = alns_cooling_rate(iterations, 0.01);
            assert!(rate > 0.0 && rate < 1.0, "{iterations}: {rate}");
            let reached = rate.powf(iterations as f64);
            assert!((reached - 0.01).abs() < 1e-6, "{iterations}: {reached}");
            assert!(AlnsConfig::default()
                .with_relative_temperature(1.0, rate, 0.01, 20)
                .validate()
                .is_ok());
        }
    }

    use super::*;

    /// The published vocabulary is what the solver answers with: every
    /// violation has an example (a new variant fails to compile here until it
    /// has one), and the fields listed are the ones it serializes.
    #[test]
    fn the_code_catalog_covers_every_violation() {
        let mut seen = [false; 13];
        for v in PlanViolation::examples() {
            let i = match v {
                PlanViolation::CapacityExceeded { .. } => 0,
                PlanViolation::TimeWindowViolated { .. } => 1,
                PlanViolation::MaxDistanceExceeded { .. } => 2,
                PlanViolation::MaxDurationExceeded { .. } => 3,
                PlanViolation::UnknownCustomer { .. } => 4,
                PlanViolation::CustomerVisitedTwice { .. } => 5,
                PlanViolation::CustomerMissing { .. } => 6,
                PlanViolation::WronglyUnassigned { .. } => 7,
                PlanViolation::UnknownVehicle { .. } => 8,
                PlanViolation::VehicleReused { .. } => 9,
                PlanViolation::TooManyRoutes { .. } => 10,
                PlanViolation::DistanceMismatch { .. } => 11,
                PlanViolation::CostMismatch { .. } => 12,
            };
            assert!(!seen[i], "two examples of variant {i}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|s| *s), "a violation without an example");
        let catalog = codes();
        let capacity = &catalog.violations[0];
        // serde_json orders an object's keys by name.
        assert_eq!(capacity.fields, ["capacity", "load", "route"]);
        assert!(catalog
            .violations
            .iter()
            .all(|v| v.checkers == ["check_vrp_solution"]));
        assert_eq!(catalog.refusals.len(), Code::ALL.len());
    }

    /// The README's code tables list exactly the published vocabulary, so the
    /// tables and `codes()` cannot drift apart.
    #[test]
    fn the_readme_documents_every_code_and_no_other() {
        let readme = include_str!("../README.md");
        let mut documented = std::collections::BTreeSet::new();
        let mut in_table = false;
        for line in readme.lines() {
            if line.starts_with("| `code` |") {
                in_table = true;
                continue;
            }
            if !line.starts_with('|') {
                in_table = false;
                continue;
            }
            if in_table && !line.starts_with("|---") {
                let first = line.split('|').nth(1).expect("a cell");
                for code in first.split('·') {
                    documented.insert(code.trim().trim_matches('`').to_string());
                }
            }
        }
        let catalog = codes();
        let published: std::collections::BTreeSet<String> = catalog
            .refusals
            .iter()
            .map(|r| r.code.clone())
            .chain(catalog.violations.iter().map(|v| v.code.clone()))
            .collect();
        assert_eq!(documented, published);
    }
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

    // ---- vehicle costs ----

    /// Three customers whose windows make the single route a detour: one
    /// route of 60.07 or two of 41.05 in all.
    fn detour_customers() -> Vec<InputCustomer> {
        vec![
            customer(
                serde_json::json!({ "id": 1, "x": 10.0, "y": 0.0, "time_window": [0.0, 10.0] }),
            ),
            customer(
                serde_json::json!({ "id": 2, "x": 10.0, "y": 1.0, "time_window": [100.0, 110.0] }),
            ),
            customer(
                serde_json::json!({ "id": 3, "x": -10.0, "y": 0.0, "time_window": [40.0, 50.0] }),
            ),
        ]
    }

    fn solve_with_fixed_cost(method: &str, fixed_cost: f64) -> VrpOutput {
        let vehicles = [vehicle(serde_json::json!({ "fixed_cost": fixed_cost }))];
        let config: InputConfig = serde_json::from_value(match method {
            "ga" => serde_json::json!({ "seed": 1, "max_generations": 60 }),
            "alns" => serde_json::json!({ "seed": 1, "max_iterations": 300 }),
            _ => serde_json::json!({}),
        })
        .expect("valid config");
        solve((0.0, 0.0), &detour_customers(), &vehicles, method, &config).expect("solves")
    }

    #[test]
    fn a_fixed_cost_trades_distance_for_fewer_routes() {
        for method in ["ga", "alns"] {
            let free = solve_with_fixed_cost(method, 0.0);
            assert_eq!(
                free.routes.len(),
                2,
                "{method} without a fixed cost: {:?}",
                free.routes
            );
            assert!((free.total_cost - free.total_distance).abs() < 1e-9);

            let costly = solve_with_fixed_cost(method, 30.0);
            assert_eq!(
                costly.routes,
                vec![vec![1, 3, 2]],
                "{method} with fixed cost 30"
            );
            // 60.07 + 30 < 41.05 + 60
            assert!((costly.total_cost - (costly.total_distance + 30.0)).abs() < 1e-9);
            assert!(costly.total_cost < free.total_distance + 60.0);
        }
    }

    #[test]
    fn total_cost_prices_each_route_with_its_vehicle() {
        let vehicles = [
            vehicle(serde_json::json!({ "capacity": 1, "fixed_cost": 5, "cost_per_distance": 2 })),
            vehicle(serde_json::json!({ "capacity": 1, "fixed_cost": 1, "cost_per_distance": 3 })),
        ];
        let customers = [
            customer(serde_json::json!({ "id": 1, "x": 3.0, "y": 4.0, "demand": 1 })),
            customer(serde_json::json!({ "id": 2, "x": -3.0, "y": 4.0, "demand": 1 })),
        ];
        let out = solve(
            (0.0, 0.0),
            &customers,
            &vehicles,
            "nn",
            &InputConfig::default(),
        )
        .expect("solves");
        assert_eq!(out.routes.len(), 2);
        // Each route is a 10-long round trip: (5 + 2·10) + (1 + 3·10).
        assert!((out.total_cost - 56.0).abs() < 1e-9, "{}", out.total_cost);

        let plan: InputSolution = serde_json::from_value(serde_json::json!({
            "routes": out.routes,
            "route_vehicles": out.route_vehicles,
            "total_cost": 55.0,
        }))
        .expect("valid plan");
        let checked = check(
            (0.0, 0.0),
            &customers,
            &vehicles,
            "nn",
            &InputConfig::default(),
            &plan,
        )
        .expect("checks");
        assert_eq!(
            checked.violations,
            vec![PlanViolation::CostMismatch {
                reported: 55.0,
                actual: 56.0
            }]
        );
    }

    #[test]
    fn vehicle_costs_are_checked() {
        for (json, parameter) in [
            (serde_json::json!({ "fixed_cost": -1 }), "fixed_cost"),
            (
                serde_json::json!({ "cost_per_distance": 0 }),
                "cost_per_distance",
            ),
        ] {
            let err = build_vehicles(&[vehicle(json)]).expect_err("out of range");
            assert_eq!(err.code(), "parameter_out_of_range");
            assert_eq!(err.fields["parameter"], parameter);
        }
        let mixed = [
            vehicle(serde_json::json!({ "fixed_cost": 10 })),
            vehicle(serde_json::json!({ "fixed_cost": 20 })),
        ];
        let err = solve(
            (0.0, 0.0),
            &detour_customers(),
            &mixed,
            "savings",
            &InputConfig::default(),
        )
        .expect_err("one vehicle type");
        assert_eq!(err.code(), "mixed_fleet");
        assert_eq!(err.fields["parameter"], "fixed_cost");
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

    /// A short run of `method`: the settings it reads, at small values -- a
    /// setting a method does not read is refused.
    fn quick(method: &str) -> InputConfig {
        match method {
            "ga" => InputConfig {
                population_size: Some(10),
                max_generations: Some(5),
                seed: Some(1),
                ..InputConfig::default()
            },
            "alns" => InputConfig {
                max_iterations: Some(20),
                seed: Some(1),
                ..InputConfig::default()
            },
            _ => InputConfig::default(),
        }
    }

    /// The plan is a function of the request: `"nn"` and `"savings"` draw no
    /// random numbers, and `"ga"` and `"alns"` given a seed and no time limit
    /// make the same run again. (`computation_time_ms` measures the call.)
    #[test]
    fn the_same_request_gives_the_same_plan() {
        let customers = ring(12);
        let vehicles = [vehicle(serde_json::json!({ "capacity": 20.0 }))];
        for method in ["nn", "savings", "ga", "alns"] {
            let run = || {
                let out = solve((0.0, 0.0), &customers, &vehicles, method, &quick(method))
                    .unwrap_or_else(|e| panic!("{method}: {}", e.message));
                (
                    out.routes,
                    out.route_vehicles,
                    out.total_distance.to_bits(),
                    out.unassigned,
                    out.stop_reason,
                )
            };
            assert_eq!(run(), run(), "{method}");
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
        let internal = build_customers((0.0, 0.0), &customers, i32::MAX)
            .expect("valid")
            .customers;
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
                let config = |method: &str| InputConfig {
                    max_vehicles,
                    ..quick(method)
                };
                for method in ["nn", "savings", "ga", "alns"] {
                    let out = solve((0.0, 0.0), &customers, &fleet, method, &config(method))
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
        let e = solve((0.0, 0.0), &customers, &fleet, "nn", &quick("nn")).expect_err("0 distance");
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
            let e =
                solve((0.0, 0.0), &customers, &fleet, method, &quick(method)).expect_err(method);
            assert_eq!(e.fields["code"], "mixed_fleet", "{method}");
            assert_eq!(e.fields["parameter"], "max_duration", "{method}");
            assert_eq!(e.fields["index"], 1, "{method}");
        }
        assert!(solve((0.0, 0.0), &customers, &fleet, "nn", &quick("nn")).is_ok());
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
            let out = solve(
                (0.0, 0.0),
                &customers,
                &one_capacity,
                method,
                &quick(method),
            )
            .expect(method);
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
        let fixed = |method: &str| InputConfig {
            max_vehicles: Some(2),
            ..quick(method)
        };
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &six_of_ten(), &two, method, &fixed(method)).expect(method);
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

    /// Each route comes with its own figures, and the plan's totals are their
    /// sums -- for every method, with a fixed cost and a per-distance cost.
    #[test]
    fn route_summaries_are_what_the_totals_sum() {
        let customers = six_of_ten();
        let truck = vec![vehicle(serde_json::json!({
            "capacity": 20.0, "cost_per_distance": 2.0, "fixed_cost": 5.0
        }))];
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &truck, method, &quick(method)).expect(method);
            assert_eq!(out.route_summaries.len(), out.routes.len(), "{method}");
            let distance: f64 = out.route_summaries.iter().map(|r| r.distance).sum();
            let cost: f64 = out.route_summaries.iter().map(|r| r.cost).sum();
            assert!((distance - out.total_distance).abs() < 1e-9, "{method}");
            assert!((cost - out.total_cost).abs() < 1e-9, "{method}");
            for (route, summary) in out.routes.iter().zip(&out.route_summaries) {
                // Every customer asks for 10 units.
                assert_eq!(summary.load, 10 * route.len() as i64, "{method}");
                assert!(
                    (summary.cost - (5.0 + 2.0 * summary.distance)).abs() < 1e-9,
                    "{method}"
                );
                // No service times or windows: the route takes as long as it is.
                assert!(
                    (summary.duration - summary.distance).abs() < 1e-9,
                    "{method}"
                );
            }
        }
    }

    /// With service times and a window that opens late, a route is back at the
    /// depot later than its length.
    #[test]
    fn a_route_summary_counts_service_and_waiting_in_its_duration() {
        let customers = vec![customer(serde_json::json!({
            "id": 1, "x": 3.0, "y": 4.0, "demand": 1.0,
            "service_time": 2.0, "time_window": [10.0, 50.0]
        }))];
        let out = solve((0.0, 0.0), &customers, &fleet(&[10.0]), "nn", &quick("nn")).expect("nn");
        let summary = &out.route_summaries[0];
        assert!((summary.distance - 10.0).abs() < 1e-9);
        // Arrives at 5, waits until 10, serves for 2 (leaves at 12), 5 back.
        assert!((summary.duration - 17.0).abs() < 1e-9, "{summary:?}");
    }

    /// A customer whose demand exceeds every vehicle's capacity rides on no
    /// route -- it comes back in `unassigned` for every method, and the
    /// customers a vehicle can carry are still served (GA used to fold them all
    /// into one overloaded route, savings and ALNS gave the oversized customer
    /// a route of its own).
    #[test]
    fn a_customer_no_vehicle_can_carry_is_unassigned_by_every_method() {
        let customers = vec![
            customer(serde_json::json!({ "id": 1, "x": 1.0, "y": 1.0, "demand": 5.0 })),
            customer(serde_json::json!({ "id": 2, "x": 4.0, "y": 2.0, "demand": 3.0 })),
            customer(serde_json::json!({ "id": 3, "x": -2.0, "y": 3.0, "demand": 2.0 })),
        ];
        let vehicles = fleet(&[4.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let config = InputConfig {
                max_vehicles: Some(3),
                ..quick(method)
            };
            let out = solve((0.0, 0.0), &customers, &vehicles, method, &config).expect(method);
            let served: Vec<usize> = out.routes.iter().flatten().copied().collect();
            if method == "nn" {
                // One vehicle is one route for nearest neighbour, whatever
                // `max_vehicles` allows: it can leave a carryable customer out.
                assert!(out.unassigned.contains(&1), "{method}: {:?}", out.routes);
            } else {
                assert_eq!(out.unassigned, vec![1], "{method}: {:?}", out.routes);
                assert!(
                    served.contains(&2) && served.contains(&3),
                    "{method}: {served:?}"
                );
            }
            for route in &out.routes {
                let load: f64 = route.iter().map(|id| [5.0, 3.0, 2.0][id - 1]).sum();
                assert!(load <= 4.0, "{method}: load {load} on {route:?}");
            }
        }
        // Capacity 0: nothing with a demand fits; every customer is unassigned.
        let none = fleet(&[0.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &customers, &none, method, &quick(method)).expect(method);
            assert_eq!(out.unassigned, vec![1, 2, 3], "{method}");
            assert!(out.routes.is_empty(), "{method}: {:?}", out.routes);
        }
    }

    /// Unlimited, the same request still needs a third route -- so the test
    /// above measures the limit, not a problem that happened to fit.
    #[test]
    fn without_the_limit_the_same_request_opens_more_routes() {
        let two = fleet(&[20.0, 20.0]);
        for method in ["savings", "ga", "alns"] {
            let out = solve((0.0, 0.0), &six_of_ten(), &two, method, &quick(method)).expect(method);
            assert_eq!(out.num_vehicles, 3, "{method}");
            assert!(out.unassigned.is_empty(), "{method}");
        }
    }

    /// Emptying a route moves its customers into spare room elsewhere before
    /// it gives any up.
    #[test]
    fn spare_capacity_is_used_before_a_customer_is_left_out() {
        let roomy = fleet(&[30.0]);
        let fixed = |method: &str| InputConfig {
            max_vehicles: Some(2),
            ..quick(method)
        };
        for method in ["savings", "ga", "alns"] {
            let out =
                solve((0.0, 0.0), &six_of_ten(), &roomy, method, &fixed(method)).expect(method);
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
        let roomy = |method: &str| InputConfig {
            max_vehicles: Some(8),
            ..quick(method)
        };
        for method in ["savings", "ga", "alns"] {
            let out = solve(
                (0.0, 0.0),
                &customers,
                &one_capacity,
                method,
                &roomy(method),
            )
            .expect(method);
            let free = solve(
                (0.0, 0.0),
                &customers,
                &one_capacity,
                method,
                &quick(method),
            )
            .expect(method);
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
        let fixed = |method: &str| InputConfig {
            max_vehicles: Some(2),
            ..quick(method)
        };
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve(
                (0.0, 0.0),
                &customers,
                &fleet(&[100.0, 100.0]),
                method,
                &fixed(method),
            )
            .expect(method);
            assert!(out.num_vehicles <= 2, "{method}");
            let internal = build_customers((0.0, 0.0), &customers, i32::MAX)
                .expect("valid")
                .customers;
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
            let config = |method: &str| InputConfig {
                max_vehicles: Some(got),
                ..quick(method)
            };
            let err = solve(
                (0.0, 0.0),
                &ring(4),
                &fleet(&[10.0]),
                method,
                &config(method),
            )
            .expect_err("below 1");
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
            let err =
                solve((0.0, 0.0), &customers, &mixed, method, &quick(method)).expect_err(method);
            assert!(
                err.message.contains(&format!("\"{method}\"")) && err.message.contains("capacit"),
                "{method}: {err}"
            );
        }
        assert!(solve((0.0, 0.0), &customers, &mixed, "nn", &quick("nn")).is_ok());

        let uniform = fleet(&[20.0, 20.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            assert!(
                solve((0.0, 0.0), &customers, &uniform, method, &quick(method)).is_ok(),
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
            let out = solve((0.0, 0.0), &customers, &two, method, &quick(method)).expect(method);
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
            let out = solve((0.0, 0.0), &customers, &fleet, method, &quick(method)).expect(method);
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
            let out = solve((0.0, 0.0), &customers, &fleet, method, &quick(method)).expect(method);
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
            ..quick("nn")
        };
        let errors = [
            solve(
                (0.0, 0.0),
                &ring(2),
                &fleet(&[10.0, 100.0]),
                "ga",
                &quick("ga"),
            )
            .expect_err("mixed fleet"),
            solve((0.0, 0.0), &fraction, &[], "nn", &quick("nn")).expect_err("fractional demand"),
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
                solve((0.0, 0.0), &[], &[], "tabu", &quick("tabu")).expect_err("method"),
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
                    &quick("nn"),
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
                    &quick("nn"),
                )
                .expect_err("fraction"),
                "not_whole_units",
                serde_json::json!({ "parameter": "demand", "index": 0, "id": 9, "value": 0.5 }),
            ),
            (
                solve((0.0, 0.0), &[], &fleet(&[10.0, 2.5]), "nn", &quick("nn"))
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
                    &quick("nn"),
                )
                .expect_err("negative"),
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "demand", "index": 0, "id": 9,
                                    "min": 0.0, "max": f64::from(i32::MAX), "got": -3.0 }),
            ),
            (
                solve((0.0, 0.0), &[], &fleet(&[10.0, 3e9]), "nn", &quick("nn"))
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
                    &quick("nn"),
                )
                .expect_err("window"),
                "invalid_time_window",
                serde_json::json!({ "id": 3, "index": 0, "ready": 4.0, "due": 1.0 }),
            ),
            (
                solve((0.0, 0.0), &ring(2), &fleet(&[10.0, 100.0]), "ga", &quick("ga"))
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
                        ..quick("ga")
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
                        ..quick("ga")
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
                        ..quick("alns")
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

    /// A setting the method does not read is refused -- named, with the
    /// methods that do -- rather than accepted and ignored.
    #[test]
    fn a_setting_the_method_does_not_read_is_refused() {
        let cases = [
            (
                "nn",
                InputConfig {
                    population_size: Some(10),
                    ..InputConfig::default()
                },
                "population_size",
                "ga",
            ),
            (
                "savings",
                InputConfig {
                    seed: Some(1),
                    ..InputConfig::default()
                },
                "seed",
                "ga, alns",
            ),
            (
                "ga",
                InputConfig {
                    max_iterations: Some(20),
                    ..InputConfig::default()
                },
                "max_iterations",
                "alns",
            ),
            (
                "alns",
                InputConfig {
                    elite_ratio: Some(0.2),
                    ..InputConfig::default()
                },
                "elite_ratio",
                "ga",
            ),
        ];
        for (method, config, setting, readers) in cases {
            let err =
                solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), method, &config).expect_err(method);
            assert_eq!(
                err.fields,
                serde_json::json!({ "code": "invalid_option",
                    "parameter": format!("config.{setting}"), "method": method }),
                "{method}"
            );
            assert!(
                err.message.contains(&format!("read by {readers}")),
                "{}",
                err.message
            );
        }
        for method in ["nn", "savings", "ga", "alns"] {
            assert!(solve(
                (0.0, 0.0),
                &ring(4),
                &fleet(&[10.0]),
                method,
                &quick(method)
            )
            .is_ok());
        }
    }

    // ---- check ----

    /// The plan `solve` returned, as `check` reads one.
    fn as_solution(out: &VrpOutput) -> InputSolution {
        serde_json::from_value(json!({
            "routes": out.routes,
            "route_vehicles": out.route_vehicles,
            "unassigned": out.unassigned,
            "total_distance": out.total_distance,
        }))
        .expect("a solution")
    }

    /// Every method's own plan passes the checker -- under time windows,
    /// distance and duration limits, a fixed fleet and (for `"nn"`) a mixed
    /// one. The checker and the solvers share no code past the distance
    /// matrix and the route evaluator, so this is a property of both.
    #[test]
    fn every_method_s_plan_passes_the_check() {
        let windows: Vec<InputCustomer> = (1..=10)
            .map(|i| {
                let a = i as f64;
                customer(json!({
                    "id": 100 + i, "x": a.cos() * 10.0, "y": a.sin() * 10.0, "demand": 4.0,
                    "service_time": 1.0, "time_window": [0.0, 30.0 + 5.0 * a]
                }))
            })
            .collect();
        let problems: Vec<(&str, Vec<InputCustomer>, Vec<InputVehicle>)> = vec![
            ("plain", ring(9), fleet(&[15.0])),
            ("windows", windows, fleet(&[20.0])),
            (
                "limits",
                ring(9),
                vec![vehicle(
                    json!({ "capacity": 15.0, "max_distance": 40.0, "max_duration": 45.0 }),
                )],
            ),
            ("mixed (nn)", ring(9), fleet(&[20.0, 10.0, 15.0])),
        ];
        for (name, customers, vehicles) in &problems {
            for max_vehicles in [None, Some(2)] {
                for method in ["nn", "savings", "ga", "alns"] {
                    if name.starts_with("mixed") && method != "nn" {
                        continue;
                    }
                    let config = InputConfig {
                        max_vehicles,
                        ..quick(method)
                    };
                    let out = solve((0.0, 0.0), customers, vehicles, method, &config)
                        .unwrap_or_else(|e| panic!("{name} {method}: {e:?}"));
                    let checked = check(
                        (0.0, 0.0),
                        customers,
                        vehicles,
                        method,
                        &config,
                        &as_solution(&out),
                    )
                    .unwrap_or_else(|e| panic!("{name} {method}: {e:?}"));
                    assert!(
                        checked.feasible && checked.violations.is_empty(),
                        "{name} {method} {max_vehicles:?}: {:?} on {out:?}",
                        checked.violations
                    );
                }
            }
        }
    }

    /// A plan that breaks the problem is a result, each rule by name, in the
    /// caller's ids.
    #[test]
    fn a_broken_plan_reports_each_rule_it_breaks() {
        let customers: Vec<InputCustomer> = (1..=4)
            .map(|i| customer(json!({ "id": 10 * i, "x": i as f64, "y": 0.0, "demand": 5.0 })))
            .collect();
        let vehicles = fleet(&[10.0, 5.0]);
        let config = InputConfig {
            max_vehicles: Some(2),
            ..InputConfig::default()
        };
        // 10 twice, 40 missing, 77 unknown, vehicle 1 over capacity and used
        // twice, three routes for two, and a total nobody drove.
        let solution: InputSolution = serde_json::from_value(json!({
            "routes": [[10, 20], [10, 30], [77]],
            "route_vehicles": [1, 0, 1],
            "unassigned": [],
            "total_distance": 1.0
        }))
        .expect("a solution");
        let out =
            check((0.0, 0.0), &customers, &vehicles, "nn", &config, &solution).expect("checked");
        assert!(!out.feasible);
        assert_eq!(
            out.violations,
            [
                PlanViolation::TooManyRoutes {
                    routes: 3,
                    max_routes: 2
                },
                PlanViolation::CapacityExceeded {
                    route: 0,
                    load: 10,
                    capacity: 5
                },
                PlanViolation::CustomerVisitedTwice {
                    route: 1,
                    customer: 10
                },
                PlanViolation::UnknownCustomer {
                    route: 2,
                    customer: 77
                },
                PlanViolation::VehicleReused {
                    route: 2,
                    vehicle: 1
                },
                PlanViolation::CustomerMissing { customer: 40 },
            ]
        );
        // The total is not compared once a route cannot be driven (77).
        let wire = serde_json::to_value(&out.violations[1]).expect("json");
        assert_eq!(
            wire,
            json!({ "code": "capacity_exceeded", "route": 0, "load": 10, "capacity": 5 })
        );
    }

    #[test]
    fn a_reported_distance_is_compared_and_route_vehicles_must_match_the_routes() {
        let customers = ring(2);
        let vehicles = fleet(&[100.0]);
        let config = InputConfig::default();
        let out = solve((0.0, 0.0), &customers, &vehicles, "nn", &config).expect("solved");
        let mut solution = as_solution(&out);
        solution.total_distance = Some(out.total_distance + 1.0);
        let checked =
            check((0.0, 0.0), &customers, &vehicles, "nn", &config, &solution).expect("checked");
        assert!(matches!(
            checked.violations.as_slice(),
            [PlanViolation::DistanceMismatch { reported, .. }] if *reported == out.total_distance + 1.0
        ));
        solution.route_vehicles = Some(vec![]);
        let e = check((0.0, 0.0), &customers, &vehicles, "nn", &config, &solution)
            .expect_err("no vehicle per route");
        assert_eq!(e.code(), "dimension_mismatch");
        assert_eq!(e.fields["parameter"], "solution.route_vehicles");
    }

    /// The catalog and the refusal are one table: with each method, a setting
    /// is accepted exactly when the catalog says the method reads it -- and
    /// every setting is read by some method.
    #[test]
    fn the_method_catalog_is_what_solve_accepts() {
        use serde_json::json;
        let value = |setting: &str| match setting {
            "population_size" => json!(10),
            "max_generations" => json!(5),
            "mutation_rate" => json!(0.2),
            "elite_ratio" => json!(0.2),
            "max_iterations" => json!(20),
            "seed" => json!(1),
            "time_limit_ms" => json!(60_000),
            "max_vehicles" => json!(4),
            other => panic!("no sample value for {other}"),
        };
        // Small budgets the method reads, so each run is quick.
        let base = |method: &str| match method {
            "ga" => json!({ "population_size": 10, "max_generations": 5 }),
            "alns" => json!({ "max_iterations": 20 }),
            _ => json!({}),
        };
        let catalog = methods();
        assert_eq!(
            catalog.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
            ["nn", "savings", "ga", "alns"]
        );
        for info in &catalog {
            for setting in SETTINGS {
                let mut request = base(&info.name);
                request[setting] = value(setting);
                let config: InputConfig = serde_json::from_value(request).expect("a config");
                let run = solve((0.0, 0.0), &ring(4), &fleet(&[10.0]), &info.name, &config);
                if info.reads.iter().any(|r| r == setting) {
                    run.unwrap_or_else(|e| panic!("{} refused {setting} it reads: {e}", info.name));
                } else {
                    let err = run.expect_err(setting);
                    assert_eq!(err.code(), "invalid_option", "{} {setting}", info.name);
                }
            }
        }
        for setting in SETTINGS {
            assert!(
                catalog.iter().any(|m| m.reads.iter().any(|r| r == setting)),
                "{setting}"
            );
        }
    }

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
                ..quick("ga")
            };
            assert_eq!(
                refused("ga", config),
                expected("config.population_size", "ga", 2.0, got as f64)
            );
        }
        for got in [-1, 0] {
            let config = InputConfig {
                max_generations: Some(got),
                ..quick("ga")
            };
            assert_eq!(
                refused("ga", config),
                expected("config.max_generations", "ga", 1.0, got as f64)
            );
            let config = InputConfig {
                max_iterations: Some(got),
                ..quick("alns")
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
                ..quick("ga")
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
            let config = |method: &str| InputConfig {
                seed: Some(-1),
                ..quick(method)
            };
            let err = solve(
                (0.0, 0.0),
                &ring(4),
                &fleet(&[10.0]),
                method,
                &config(method),
            )
            .expect_err("negative seed");
            assert_eq!(
                err.fields,
                serde_json::json!({ "code": "parameter_out_of_range", "parameter": "config.seed",
                    "method": method, "min": 0.0, "max": null, "got": -1.0 })
            );
            let config = |method: &str| InputConfig {
                seed: Some(u64::MAX.into()),
                ..quick(method)
            };
            assert!(
                solve(
                    (0.0, 0.0),
                    &ring(4),
                    &fleet(&[10.0]),
                    method,
                    &config(method)
                )
                .is_ok(),
                "{method}: the largest seed"
            );
        }
        let err = whole_in::<u64>("config.seed", i128::from(u64::MAX) + 1, 0, u64::MAX.into())
            .expect_err("past u64");
        assert_eq!(err.fields["max"], serde_json::json!(u64::MAX as f64));
        assert_eq!(err.fields["min"], serde_json::json!(0.0));
    }

    /// `time_limit_ms` ends a GA or ALNS run that would otherwise go far
    /// longer; it is a whole number of at least 1, and the one-pass methods
    /// refuse it as a setting they do not read.
    #[test]
    fn a_time_limit_ends_the_search_methods() {
        for method in ["ga", "alns"] {
            let config = InputConfig {
                max_generations: (method == "ga").then_some(1_000_000),
                max_iterations: (method == "alns").then_some(100_000_000),
                seed: Some(1),
                time_limit_ms: Some(1),
                ..InputConfig::default()
            };
            let out = solve((0.0, 0.0), &ring(30), &fleet(&[20.0]), method, &config)
                .expect("a time-limited run");
            assert_eq!(out.stop_reason, StopReason::TimeLimit, "{method}");
            assert!(out.unassigned.is_empty(), "{method}");

            let zero = InputConfig {
                time_limit_ms: Some(0),
                ..InputConfig::default()
            };
            let err =
                solve((0.0, 0.0), &ring(3), &fleet(&[20.0]), method, &zero).expect_err("0 ms");
            assert_eq!(err.fields["code"], "parameter_out_of_range");
            assert_eq!(err.fields["parameter"], "config.time_limit_ms");
            assert_eq!(err.fields["min"], serde_json::json!(1.0));
        }
        let unread = InputConfig {
            time_limit_ms: Some(100),
            ..InputConfig::default()
        };
        let err = solve((0.0, 0.0), &ring(3), &fleet(&[20.0]), "savings", &unread)
            .expect_err("savings reads no time limit");
        assert_eq!(err.fields["code"], "invalid_option");
        assert_eq!(err.fields["parameter"], "config.time_limit_ms");
    }

    /// An id is a label from 0 up; a negative one is out of that range, at
    /// the customer's position.
    #[test]
    fn a_negative_id_is_out_of_range_at_its_customer() {
        let customers = [
            customer(serde_json::json!({ "id": 1, "x": 1.0, "y": 0.0 })),
            customer(serde_json::json!({ "id": -1, "x": 0.0, "y": 1.0 })),
        ];
        let err = solve((0.0, 0.0), &customers, &[], "nn", &quick("nn")).expect_err("negative id");
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
        assert!(result.is_ok(), "GA with valid config should succeed");
        let output = result.unwrap();
        assert_eq!(output.method, Method::Genetic);
        assert!(!output.routes.is_empty());
        assert!(
            output
                .routes
                .iter()
                .map(|r| route_distance(r, &dm))
                .sum::<f64>()
                > 0.0
        );
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
        let err = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg)
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
                solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg).expect_err("out of (0, 1]");
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
        let result = solve_alns(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
        let result = solve_alns(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
        assert!(result.is_ok(), "GA with default config should succeed");
    }

    #[test]
    fn alns_default_config() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_alns(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 1000), &cfg);
        assert!(
            result.is_ok(),
            "GA with 20 customers should succeed: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert!(!output.routes.is_empty());
        assert!(
            output
                .routes
                .iter()
                .map(|r| route_distance(r, &dm))
                .sum::<f64>()
                > 0.0
        );
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 5), &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), &cfg);
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

/// The closed string vocabularies of the wire types, as JSON Schema built from
/// the tables the binding parses and prints them from (the engine card's
/// schemas; tests only).
#[cfg(all(test, feature = "wasm"))]
pub(crate) mod schema {
    use schemars::{Schema, SchemaGenerator};

    use crate::engine_card::string_enum;

    pub(crate) fn stop_reason(_: &mut SchemaGenerator) -> Schema {
        string_enum(super::StopReason::ALL.iter().map(|r| r.name()))
    }

    pub(crate) fn method(_: &mut SchemaGenerator) -> Schema {
        string_enum(super::Method::ALL.iter().map(|m| m.name()))
    }

    pub(crate) fn settings(_: &mut SchemaGenerator) -> Schema {
        let item = string_enum(super::SETTINGS).to_value();
        Schema::try_from(serde_json::json!({ "type": "array", "items": item }))
            .expect("an object is a schema")
    }

    pub(crate) fn checkers(_: &mut SchemaGenerator) -> Schema {
        let item = string_enum(["check_vrp_solution"]).to_value();
        Schema::try_from(serde_json::json!({ "type": "array", "items": item }))
            .expect("an object is a schema")
    }
}

/// `solve_vrp` against an oracle built from the definitions: the plan is
/// complete, within capacity, and its reported distances are the distances of
/// its routes; on a map small enough to enumerate, no method beats the best
/// tour and the searches find it.
#[cfg(test)]
mod oracle_tests {
    use super::*;
    use serde_json::json;

    fn customers(count: usize, seed: u64) -> Vec<(f64, f64, f64)> {
        let mut state = seed;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % m) as f64
        };
        (0..count)
            .map(|_| (next(100) / 2.0, next(100) / 2.0, 1.0 + next(4)))
            .collect()
    }

    fn input(list: &[(f64, f64, f64)]) -> Vec<InputCustomer> {
        list.iter()
            .enumerate()
            .map(|(i, &(x, y, demand))| {
                serde_json::from_value(json!({ "id": i + 1, "x": x, "y": y, "demand": demand }))
                    .expect("a customer")
            })
            .collect()
    }

    fn vehicles(count: usize, capacity: f64) -> Vec<InputVehicle> {
        (0..count)
            .map(|_| serde_json::from_value(json!({ "capacity": capacity })).expect("a vehicle"))
            .collect()
    }

    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    }

    fn route_length(depot: (f64, f64), list: &[(f64, f64, f64)], route: &[usize]) -> f64 {
        let mut at = depot;
        let mut total = 0.0;
        for &id in route {
            let p = (list[id - 1].0, list[id - 1].1);
            total += dist(at, p);
            at = p;
        }
        total + dist(at, depot)
    }

    fn config() -> InputConfig {
        serde_json::from_value(json!({})).expect("defaults")
    }

    #[test]
    fn every_method_returns_a_complete_feasible_plan_with_true_distances() {
        let depot = (25.0, 25.0);
        for seed in 1..=4u64 {
            let list = customers(9, seed);
            let capacity = 7.0;
            for method in methods() {
                let out = solve(
                    depot,
                    &input(&list),
                    &vehicles(6, capacity),
                    &method.name,
                    &config(),
                )
                .unwrap_or_else(|e| panic!("{} seed {seed}: {e:?}", method.name));
                let mut seen = vec![0u32; list.len() + 1];
                let mut total = 0.0;
                for (route, summary) in out.routes.iter().zip(&out.route_summaries) {
                    let load: f64 = route.iter().map(|&id| list[id - 1].2).sum();
                    assert!(load <= capacity + 1e-9, "{} load {load}", method.name);
                    assert_eq!(summary.load as f64, load, "{} route load", method.name);
                    let length = route_length(depot, &list, route);
                    assert!((summary.distance - length).abs() < 1e-9, "{}", method.name);
                    total += length;
                    for &id in route {
                        seen[id] += 1;
                    }
                }
                for &id in &out.unassigned {
                    seen[id] += 1;
                }
                assert!(
                    seen[1..].iter().all(|&c| c == 1),
                    "{} seed {seed}: every customer exactly once, got {seen:?}",
                    method.name
                );
                assert!((out.total_distance - total).abs() < 1e-9, "{}", method.name);
            }
        }
    }

    fn best_tour(depot: (f64, f64), list: &[(f64, f64, f64)]) -> f64 {
        fn permute(
            k: usize,
            order: &mut Vec<usize>,
            best: &mut f64,
            depot: (f64, f64),
            list: &[(f64, f64, f64)],
        ) {
            if k == order.len() {
                *best = best.min(route_length(depot, list, order));
                return;
            }
            for i in k..order.len() {
                order.swap(k, i);
                permute(k + 1, order, best, depot, list);
                order.swap(k, i);
            }
        }
        let mut order: Vec<usize> = (1..=list.len()).collect();
        let mut best = f64::INFINITY;
        permute(0, &mut order, &mut best, depot, list);
        best
    }

    #[test]
    fn on_a_map_small_enough_to_enumerate_no_method_beats_the_best_tour_and_the_searches_find_it() {
        let depot = (25.0, 25.0);
        for seed in 11..=13u64 {
            let list = customers(7, seed);
            let optimum = best_tour(depot, &list);
            for method in methods() {
                let out = solve(
                    depot,
                    &input(&list),
                    &vehicles(1, 1e9),
                    &method.name,
                    &config(),
                )
                .unwrap_or_else(|e| panic!("{}: {e:?}", method.name));
                assert!(
                    out.total_distance >= optimum - 1e-9,
                    "{} seed {seed}: {} beats the optimum {optimum}",
                    method.name,
                    out.total_distance
                );
                if method.name == "ga" || method.name == "alns" {
                    assert!(
                        out.total_distance <= optimum * 1.001,
                        "{} seed {seed}: {} against the optimum {optimum}",
                        method.name,
                        out.total_distance
                    );
                }
            }
        }
    }
    fn plan(routes: serde_json::Value, extra: serde_json::Value) -> InputSolution {
        let mut v = json!({ "routes": routes });
        if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
            for (k, x) in e {
                o.insert(k.clone(), x.clone());
            }
        }
        serde_json::from_value(v).expect("a plan")
    }

    /// `check_vrp_solution` accepts what `solve_vrp` returns and names each
    /// way a plan can be wrong -- a customer served twice or not at all, a
    /// route over capacity, a reported distance that is not the routes'.
    #[test]
    fn the_checker_accepts_the_solvers_plans_and_names_each_defect() {
        let depot = (25.0, 25.0);
        let list = customers(8, 5);
        let capacity = 7.0;
        let cfg = config();
        let out = solve(depot, &input(&list), &vehicles(6, capacity), "ga", &cfg).expect("solved");
        let routes = json!(out.routes);
        let good = plan(
            routes.clone(),
            json!({ "unassigned": out.unassigned, "total_distance": out.total_distance }),
        );
        let check_of = |p: &InputSolution| {
            check(depot, &input(&list), &vehicles(6, capacity), "ga", &cfg, p)
                .expect("a checkable plan")
        };
        let ok = check_of(&good);
        assert!(ok.feasible, "{:?}", ok.violations);

        // a customer dropped from its route, and not listed as unassigned
        let mut dropped = out.routes.clone();
        let removed = dropped[0].remove(0);
        let r = check_of(&plan(json!(dropped), json!({})));
        assert!(
            r.violations
                .iter()
                .any(|v| matches!(v, PlanViolation::CustomerMissing { customer } if *customer == removed as i64)),
            "{:?}",
            r.violations
        );

        // a customer served by two routes
        let mut twice = out.routes.clone();
        let extra = twice[0][0];
        twice.push(vec![extra]);
        let r = check_of(&plan(json!(twice), json!({})));
        assert!(
            r.violations
                .iter()
                .any(|v| matches!(v, PlanViolation::CustomerVisitedTwice { customer, .. } if *customer == extra as i64)),
            "{:?}",
            r.violations
        );

        // every customer on one route: over the capacity
        let all: Vec<usize> = (1..=list.len()).collect();
        let r = check_of(&plan(json!([all]), json!({})));
        assert!(
            r.violations
                .iter()
                .any(|v| matches!(v, PlanViolation::CapacityExceeded { .. })),
            "{:?}",
            r.violations
        );

        // a reported distance that is not the routes' distance
        let r = check_of(&plan(
            routes,
            json!({ "unassigned": out.unassigned, "total_distance": out.total_distance + 5.0 }),
        ));
        assert!(
            r.violations
                .iter()
                .any(|v| matches!(v, PlanViolation::DistanceMismatch { .. })),
            "{:?}",
            r.violations
        );
    }
    /// `available_methods` tells the truth about settings: a method accepts
    /// each setting it lists as read and refuses every other.
    #[test]
    fn the_method_catalogue_lists_exactly_the_settings_each_method_reads() {
        let names: Vec<String> = methods().into_iter().map(|m| m.name).collect();
        assert_eq!(names, ["nn", "savings", "ga", "alns"]);
        let value = |setting: &str| match setting {
            "population_size" | "max_generations" | "max_iterations" => json!(20),
            "mutation_rate" | "elite_ratio" => json!(0.1),
            "seed" | "time_limit_ms" => json!(1000),
            "max_vehicles" => json!(5),
            other => panic!("unknown setting {other}"),
        };
        let depot = (25.0, 25.0);
        let list = customers(6, 3);
        for method in methods() {
            for setting in SETTINGS {
                let cfg: InputConfig =
                    serde_json::from_value(json!({ setting: value(setting) })).expect("a config");
                let result = solve(depot, &input(&list), &vehicles(4, 8.0), &method.name, &cfg);
                let reads = method.reads.iter().any(|r| r == setting);
                assert_eq!(
                    result.is_ok(),
                    reads,
                    "{} with {setting}: listed as read = {reads}, got {:?}",
                    method.name,
                    result.err().map(|e| e.code().to_string())
                );
            }
        }
    }

    /// A vehicle given no capacity has none: any demand a customer may have
    /// fits it (up to the 2,147,483,647 units the model counts), as the vehicle
    /// of a request with no `vehicles` does.
    #[test]
    fn a_vehicle_without_a_capacity_takes_any_demand() {
        let list = [(10.0, 0.0, 1.5e9), (20.0, 5.0, 4.0e8), (5.0, 15.0, 2.0e8)];
        let fleet: Vec<InputVehicle> = vec![serde_json::from_value(json!({})).expect("a vehicle")];
        for method in methods() {
            let out = solve((0.0, 0.0), &input(&list), &fleet, &method.name, &config())
                .unwrap_or_else(|e| panic!("{}: {e:?}", method.name));
            assert!(
                out.unassigned.is_empty(),
                "{}: left {:?} unserved by a vehicle without a limit",
                method.name,
                out.unassigned
            );
        }
    }

    /// Loads are added in 32 bits, so demands that add up past what 32 bits
    /// hold are refused -- not summed into a wrapped, negative load that every
    /// capacity check then accepts (or an overflow panic in a debug build).
    #[test]
    fn demands_that_add_up_past_the_model_are_refused() {
        let list = [(10.0, 0.0, 1.5e9), (20.0, 5.0, 1.5e9)];
        for capacity in [None, Some(2.0e9)] {
            let fleet: Vec<InputVehicle> = vec![serde_json::from_value(match capacity {
                Some(c) => json!({ "capacity": c }),
                None => json!({}),
            })
            .expect("a vehicle")];
            for method in methods() {
                let err = solve((0.0, 0.0), &input(&list), &fleet, &method.name, &config())
                    .expect_err(&method.name);
                assert_eq!(
                    err.fields["code"], "parameter_out_of_range",
                    "{}: {err:?}",
                    method.name
                );
                assert_eq!(err.fields["parameter"], "total_demand", "{}", method.name);
                assert_eq!(err.fields["got"], 3.0e9, "{}", method.name);
                assert_eq!(err.fields["max"], f64::from(i32::MAX), "{}", method.name);
            }
        }
        // A customer no vehicle carries is not summed: it is unassigned.
        let fleet = vehicles(1, 1000.0);
        let list = [(10.0, 0.0, 5.0), (20.0, 5.0, 2.0e9), (5.0, 15.0, 2.0e9)];
        let out = solve((0.0, 0.0), &input(&list), &fleet, "nn", &config()).expect("solves");
        assert_eq!(out.unassigned.len(), 2);
    }
}
