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
use crate::assignment::{assign_types, RouteDemand, VehicleType};
use crate::balance::{self, Balance, Quantity, RouteFigures};
use crate::constructive::{
    by_depot, clarke_wright_savings, nearest_neighbor, nearest_neighbor_tw, Share,
};
use crate::distance::DistanceMatrix;
use crate::evaluation::{
    has_pairs, has_soft_windows, has_time_windows, route_checks_needed, route_eval, route_feasible,
    route_peak_load, RouteEvaluator, RouteLimits,
};
use crate::fleet::{cheapest_insertion, limit_routes};
use crate::ga::RoutingGaProblem;
use crate::ga::{split, split_tw};
use crate::local_search::{
    cross_depot_improve, or_opt_improve, pair_relocate_improve, two_opt_improve,
};
use crate::models::{num_depots, Customer, TimeWindow, Vehicle};
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
    Internal,
}

impl Code {
    pub(crate) const ALL: [Code; 10] = [
        Code::MalformedInput,
        Code::ValueNotFinite,
        Code::UnknownOption,
        Code::InvalidOption,
        Code::ParameterOutOfRange,
        Code::DimensionMismatch,
        Code::DuplicateId,
        Code::InvalidTimeWindow,
        Code::NotWholeUnits,
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
    /// What each unit of time a vehicle is late costs, in units of distance, making the end of the
    /// `time_window` soft: a route may arrive after `due`, and is priced at its distance plus
    /// this times the lateness (`route_summaries[].lateness_cost`). Absent or `null`: the window
    /// is hard -- a late arrival is not allowed. At least 0; has no effect on a customer
    /// without a `time_window`.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    lateness_penalty: Option<f64>,
    /// Makes this customer the pickup of a pair: the `id` of the customer that unloads what is
    /// loaded here. The pair goes on one route, this customer first; `demand` is what the pair
    /// carries between the two stops, and the delivery's own `demand` is 0 (or absent). A
    /// delivery is named by one pickup only and is not a pickup itself. Absent or `null`: an
    /// ordinary customer. Pairs are planned by the `"nn"`, `"savings"` and `"alns"` methods, with
    /// one kind of vehicle per depot.
    #[serde(default, deserialize_with = "optional_whole")]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    delivery: Option<i128>,
}

impl InputCustomer {
    /// No service time and no time window: the customer adds no time but the
    /// travel to it.
    fn is_untimed(&self) -> bool {
        self.service_time == 0.0 && self.time_window.is_none()
    }
}

/// A depot: a point routes start from and return to.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputDepot {
    x: f64,
    y: f64,
}

/// The depots as the points the solver reads.
pub(crate) fn depot_points(depots: &[InputDepot]) -> Vec<(f64, f64)> {
    depots.iter().map(|d| (d.x, d.y)).collect()
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
    /// How many vehicles of this kind there are (a whole number `>= 1`). Absent
    /// or `null`: as many as the plan needs. A fleet may list kinds that differ in
    /// capacity, costs and limits; every method plans with all of them.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    count: Option<i128>,
    /// The depot this vehicle starts from and returns to: its index in `depots`
    /// (a whole number from 0). Absent or `null`: the first.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    depot: Option<i128>,
    /// A rest the driver takes during every route of this vehicle: `duration` long, starting at
    /// some time in `[earliest, latest]` (on the clock that starts at 0) at the stop the vehicle
    /// has just reached, before that stop's service -- so it can overlap the wait for a time window
    /// to open. A route has to contain it unless the vehicle is back at the depot by `earliest`.
    /// It is placed where the route finishes earliest and costs nothing itself: it makes the route
    /// longer, which `max_duration` and the time windows after it feel. A route that cannot place
    /// it is not allowed (`break_not_placeable` in a plan check). Absent or `null`: no break.
    #[serde(default, rename = "break")]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "InputBreak | null"))]
    rest: Option<InputBreak>,
}

/// The rest of a vehicle's driver ([`InputVehicle::rest`]).
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputBreak {
    /// The first time the break may start (`>= 0`).
    earliest: f64,
    /// The last time the break may start (`>= earliest`).
    latest: f64,
    /// How long the break takes (`> 0`).
    duration: f64,
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
#[derive(Deserialize, Default, Clone)]
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
    /// Weigh the busiest route against cost: the plan minimises `total_cost`
    /// plus `weight` × the largest workload of any route (`equity`), where a
    /// workload is what `quantity` counts. Absent or `null`: cost alone.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "InputBalance | null"))]
    balance: Option<InputBalance>,
}

/// The balance term of the objective (`config.balance`).
#[derive(Deserialize, Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct InputBalance {
    /// What a route's workload counts: `"load"` -- its load as a share of
    /// its vehicle's capacity; `"distance"` -- its length; `"duration"` --
    /// when it is back at the depot (needs travel times, service times or
    /// time windows; without them it is the distance).
    #[cfg_attr(
        feature = "wasm",
        tsify(type = "\"load\" | \"distance\" | \"duration\"")
    )]
    quantity: String,
    /// Units of cost one unit of the largest workload is worth (`>= 0`;
    /// `0` is cost alone).
    weight: f64,
}

impl InputConfig {
    /// This config for one of `parts` runs that share the time limit: its limit
    /// divided by `parts` (at least 1 ms). `None` when there is no limit to share.
    fn with_time_share(&self, parts: usize) -> Option<InputConfig> {
        let limit = self.time_limit_ms.filter(|&ms| ms > 0)?;
        let mut shared = self.clone();
        shared.time_limit_ms = Some((limit / parts as i128).max(1));
        Some(shared)
    }
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
    /// The route's vehicle's `fixed_cost` plus `cost_per_distance` × (`distance` + `lateness_cost`):
    /// a route's lateness is priced like distance.
    pub(crate) cost: f64,
    /// When the route is back at the depot, leaving it at time 0: travel
    /// (one unit of distance is one unit of time), waiting for time windows to
    /// open and service times. Equal to `distance` for a route with neither.
    pub(crate) duration: f64,
    /// How late the route is, summed over its customers with a soft window (`lateness_penalty`):
    /// each one's arrival after its `due`. `0` when every window is met or hard.
    pub(crate) lateness: f64,
    /// What that lateness costs, in units of distance: each customer's lateness times its
    /// `lateness_penalty`, summed. `0` without a soft window.
    pub(crate) lateness_cost: f64,
    /// What `config.balance.quantity` counts for this route; `null` without
    /// `config.balance`.
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) workload: Option<f64>,
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
    /// `fixed_cost` plus `cost_per_distance` × (the route's distance plus its `lateness_cost`).
    /// Equal to `total_distance` with the default costs and no soft window.
    pub(crate) total_cost: f64,
    /// The largest `workload` of any route (`0` for a plan without routes);
    /// `null` without `config.balance`.
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) equity: Option<f64>,
    /// What the plan minimises: `total_cost` + `config.balance.weight` ×
    /// `equity`; `null` without `config.balance`.
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) objective: Option<f64>,
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
pub(crate) const SETTINGS: [&str; 9] = [
    "population_size",
    "max_generations",
    "mutation_rate",
    "elite_ratio",
    "max_iterations",
    "seed",
    "time_limit_ms",
    "max_vehicles",
    "balance",
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
    /// - `balance`: every method -- the objective all of them minimise.
    fn reads_setting(self, setting: &str) -> bool {
        use Method::*;
        match setting {
            "population_size" | "max_generations" | "mutation_rate" | "elite_ratio" => {
                self == Genetic
            }
            "max_iterations" => self == Alns,
            "seed" | "time_limit_ms" => matches!(self, Genetic | Alns),
            "max_vehicles" | "balance" => true,
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

/// The problem a request states, as [`solve`] and [`check`] read it: the
/// depot, the customers and the fleet, and the distances and travel times
/// between the points when the request gives them (else from the
/// coordinates).
#[derive(Clone, Copy)]
pub(crate) struct ProblemInput<'a> {
    /// The depots, at least one: a vehicle starts from the one its `depot` names.
    pub(crate) depots: &'a [(f64, f64)],
    pub(crate) customers: &'a [InputCustomer],
    pub(crate) vehicles: &'a [InputVehicle],
    pub(crate) distances: Option<&'a [Vec<f64>]>,
    pub(crate) durations: Option<&'a [Vec<f64>]>,
}

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
    problem: &ProblemInput,
    method: &str,
    config: &InputConfig,
) -> Result<VrpOutput, ServiceError> {
    let ProblemInput {
        depots,
        customers: input_customers,
        vehicles: input_vehicles,
        distances,
        durations,
    } = *problem;
    let method = Method::parse(method)?;
    refuse_unread_settings(method, config)?;
    require_depots(depots)?;
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
    let fleet = build_vehicles(input_vehicles, depots.len())?;
    let vehicles: Vec<Vehicle> = fleet.iter().map(|t| t.vehicle().clone()).collect();
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
        positions,
    } = build_customers(depots, input_customers, largest_load)?;
    refuse_unsupported_pairs(method, &customers, &fleet)?;
    let dm = distance_matrix(
        &customers,
        &positions,
        distances,
        durations,
        input_customers.len(),
    )?;
    let balance = balance_of(config, input_customers, durations.is_some())?;
    if customers.len() <= depots.len() {
        return Ok(VrpOutput {
            routes: vec![],
            route_vehicles: vec![],
            route_summaries: vec![],
            total_distance: 0.0,
            total_cost: 0.0,
            equity: balance.map(|_| 0.0),
            objective: balance.map(|_| 0.0),
            num_vehicles: 0,
            method_used: method.name().to_string(),
            computation_time_ms: 0.0,
            unassigned: all_ids,
            stop_reason: StopReason::Completed,
        });
    }

    let planning = Planning {
        method,
        config,
        max_vehicles,
        balance,
    };
    let plan = if depots.len() == 1 {
        plan_one_depot(&planning, &customers, &dm, &fleet, &vehicles)?
    } else {
        plan_by_depots(&planning, &customers, &dm, &fleet, &vehicles)?
    };
    let route_summaries: Vec<RouteSummary> = plan
        .routes
        .iter()
        .zip(&plan.vehicles)
        .map(|(route, &v)| {
            let (built, _) = RouteEvaluator::new(&customers, &dm, &vehicles[v]).build_route(route);
            let depot = vehicles[v].depot_id();
            let distance = route_distance_from(route, depot, &dm);
            RouteSummary {
                distance,
                load: route_peak_load(route, &customers).into(),
                cost: vehicles[v].route_cost(priced_distance_from(route, depot, &dm, &customers)),
                duration: built.total_duration(),
                lateness: built.total_lateness(),
                lateness_cost: built.lateness_cost(),
                workload: balance.map(|b| {
                    RouteFigures::of(route, &dm, &customers).workload(b.quantity, &vehicles[v])
                }),
            }
        })
        .collect();
    let total_cost: f64 = route_summaries.iter().map(|r| r.cost).sum();
    let equity = balance.map(|_| {
        route_summaries
            .iter()
            .filter_map(|r| r.workload)
            .fold(0.0, f64::max)
    });
    let objective = balance.zip(equity).map(|(b, e)| total_cost + b.weight * e);
    let total_distance = route_summaries.iter().map(|r| r.distance).sum();
    let routes = map_routes(&plan.routes, id_map.as_slice(), depots.len());
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
        equity,
        objective,
        routes,
        route_vehicles: plan.vehicles,
        route_summaries,
        method_used: plan.method.name().to_string(),
        computation_time_ms: 0.0,
        unassigned,
        stop_reason: plan.stop_reason,
    })
}

/// What a plan is made with, besides the problem: the method and its settings,
/// the limit on routes and the balance term.
#[derive(Clone, Copy)]
struct Planning<'a> {
    method: Method,
    config: &'a InputConfig,
    max_vehicles: Option<usize>,
    balance: Option<Balance>,
}

/// The plan for a problem with one depot (entry 0 of `customers`): every
/// attempt the method makes -- each planning kind of the fleet, hard windows
/// where the method is greedy -- brought to the fleet, the fixed fleet and the
/// balance term, and the best one kept.
fn plan_one_depot(
    planning: &Planning,
    customers: &[Customer],
    dm: &DistanceMatrix,
    fleet: &[VehicleType],
    vehicles: &[Vehicle],
) -> Result<Plan, ServiceError> {
    let Planning {
        method,
        config,
        max_vehicles,
        balance,
    } = *planning;
    // One kind of vehicle with a count is a fleet of that many: the routes
    // are bounded like `max_vehicles` bounds them.
    let max_vehicles = match (fleet, method) {
        ([only], m) if m != Method::NearestNeighbor => match (max_vehicles, only.count()) {
            (Some(max), Some(count)) => Some(max.min(count)),
            (max, count) => max.or(count),
        },
        _ => max_vehicles,
    };
    // The methods that plan every route with one vehicle plan with one kind of
    // the fleet, and a fleet of several kinds is then given its vehicles route
    // by route (see `assign_fleet`). Which kind to plan with decides how good the
    // plan can be, so each kind that can carry the largest customer gets a turn
    // and the best plan stands -- never worse than the best one kind alone.
    let templates: Vec<usize> = if fleet.len() > 1 && method != Method::NearestNeighbor {
        planning_kinds(fleet, customers)
    } else {
        vec![planning_vehicle(vehicles)]
    };
    // A time limit is the run's whole, shared among the plans it makes.
    let shared = (templates.len() > 1)
        .then(|| config.with_time_share(templates.len()))
        .flatten();
    let config = shared.as_ref().unwrap_or(config);
    let mut chosen: Option<(Plan, usize, f64)> = None;
    // The one-pass methods are greedy, and what greed does with a soft window is no better than
    // what it does with a hard one by rule: a soft problem has every plan of the hard one and more,
    // so its plan must not cost more. They plan the problem as given and with every window
    // hard, and the cheaper plan (priced with the soft windows) stands, as with the planning kinds.
    let hardened: Option<Vec<Customer>> = (has_soft_windows(customers)
        && matches!(method, Method::NearestNeighbor | Method::Savings))
    .then(|| {
        customers
            .iter()
            .cloned()
            .map(Customer::without_lateness_penalty)
            .collect()
    });
    let mut sets: Vec<&[Customer]> = vec![customers];
    sets.extend(hardened.as_deref());
    let attempts: Vec<(usize, &[Customer])> = templates
        .iter()
        .flat_map(|&planner| sets.iter().map(move |&set| (planner, set)))
        .collect();
    for (planner, set) in attempts {
        let plan = match method {
            Method::NearestNeighbor => solve_nn(set, dm, fleet),
            Method::Savings => solve_savings(set, dm, &vehicles[planner], planner),
            Method::Genetic => solve_ga(set, dm, &vehicles[planner], planner, config)?,
            Method::Alns => solve_alns(set, dm, &vehicles[planner], planner, config)?,
        };
        // A fixed fleet is kept by every method alike: `"nn"` cannot exceed its
        // vehicle list, but it can exceed a smaller `max_vehicles`.
        let plan = match max_vehicles {
            Some(max) if plan.routes.len() > max => {
                let everyone: Vec<usize> = plan.routes.iter().flatten().copied().collect();
                // The plan of one kind of vehicle: every route drives the same one.
                let one_kind = plan
                    .vehicles
                    .first()
                    .copied()
                    .filter(|k| plan.vehicles.iter().all(|v| v == k));
                let limited =
                    limit_routes(plan.routes, plan.vehicles, vehicles, max, dm, customers);
                // Emptying whole routes leaves a customer out when the routes that
                // remain have the room only in pieces. Loading the vehicles first is
                // tried as well, and stands when it serves more customers.
                let packed = one_kind
                    .filter(|_| !limited.unserved.is_empty())
                    .map(|kind| {
                        pack_vehicles(&everyone, vec![kind; max], |k| &vehicles[k], customers, dm)
                    })
                    .filter(|(routes, _)| {
                        routes.iter().map(Vec::len).sum::<usize>()
                            > limited.routes.iter().map(Vec::len).sum::<usize>()
                    });
                let (routes, kinds) = packed.unwrap_or((limited.routes, limited.vehicles));
                let limits: Vec<RouteLimits> = kinds
                    .iter()
                    .map(|&v| RouteLimits::of(&vehicles[v]))
                    .collect();
                let (routes, _) = apply_local_search(&routes, &limits, dm, customers);
                Plan {
                    routes,
                    vehicles: kinds,
                    method,
                    stop_reason: plan.stop_reason,
                }
            }
            _ => plan,
        };
        let plan = if fleet.len() > 1 && method != Method::NearestNeighbor {
            assign_fleet(plan, fleet, customers, dm)
        } else {
            plan
        };
        // Every method ends on the same objective: moves between its routes
        // while they lower cost + weight × the busiest route.
        let plan = match balance {
            Some(b) => {
                let (routes, kinds) =
                    balance::improve(plan.routes, plan.vehicles, vehicles, customers, dm, b);
                Plan {
                    routes,
                    vehicles: kinds,
                    ..plan
                }
            }
            None => plan,
        };
        // The plan that leaves fewer customers out wins, then the one with the
        // lower objective (the cost, without a balance).
        let served: usize = plan.routes.iter().map(Vec::len).sum();
        let cost: f64 = plan
            .routes
            .iter()
            .zip(&plan.vehicles)
            .map(|(r, &v)| vehicles[v].route_cost(priced_distance(r, dm, customers)))
            .sum();
        let cost = match balance {
            Some(b) => {
                let equity = plan
                    .routes
                    .iter()
                    .zip(&plan.vehicles)
                    .map(|(r, &v)| {
                        RouteFigures::of(r, dm, customers).workload(b.quantity, &vehicles[v])
                    })
                    .fold(0.0, f64::max);
                cost + b.weight * equity
            }
            None => cost,
        };
        let better = chosen
            .as_ref()
            .is_none_or(|(_, s, c)| served > *s || (served == *s && cost < *c));
        if better {
            chosen = Some((plan, served, cost));
        }
    }
    let (plan, _, _) = chosen.expect("at least one plan is made");
    Ok(plan)
}

/// The plan for a problem with several depots: the customers are given to
/// depots, each depot is planned alone as a problem with one depot (every
/// method, fleet and setting as for one depot), and the plans are then
/// improved by moving customers between the routes of different depots.
fn plan_by_depots(
    planning: &Planning,
    customers: &[Customer],
    dm: &DistanceMatrix,
    fleet: &[VehicleType],
    vehicles: &[Vehicle],
) -> Result<Plan, ServiceError> {
    // Both span every route of the plan; a plan made depot by depot cannot keep them.
    for (parameter, given) in [
        ("config.max_vehicles", planning.max_vehicles.is_some()),
        ("config.balance", planning.balance.is_some()),
    ] {
        if given {
            return Err(ServiceError::new(
                Code::InvalidOption,
                format!(
                    "{parameter} is not supported with several depots yet -- it spans every \
                     route of the plan, and each depot is planned on its own; leave it out"
                ),
                json!({ "parameter": parameter }),
            ));
        }
    }
    let depots = num_depots(customers);
    // A time limit is the run's whole, shared among the depots' plans.
    let shared = planning.config.with_time_share(depots);
    let each = Planning {
        config: shared.as_ref().unwrap_or(planning.config),
        ..*planning
    };
    let fleet_depots: Vec<usize> = vehicles.iter().map(Vehicle::depot_id).collect();
    let mut stops: Vec<StopReason> = Vec::new();
    let (built, _) = by_depot(
        customers,
        dm,
        &fleet_depots,
        |_, table, sub_dm, positions| {
            // The depot's vehicles as its own table sees them: its depot is entry 0.
            let sub_fleet: Vec<VehicleType> = positions
                .iter()
                .enumerate()
                .map(|(k, &i)| VehicleType::new(vehicles[i].placed(k, 0), fleet[i].count()))
                .collect();
            let sub_vehicles: Vec<Vehicle> =
                sub_fleet.iter().map(|t| t.vehicle().clone()).collect();
            let plan = plan_one_depot(&each, table, sub_dm, &sub_fleet, &sub_vehicles)?;
            stops.push(plan.stop_reason);
            let mut served = vec![false; table.len()];
            for &c in plan.routes.iter().flatten() {
                served[c] = true;
            }
            Ok(Share {
                unassigned: (1..table.len()).filter(|&c| !served[c]).collect(),
                routes: plan.vehicles.into_iter().zip(plan.routes).collect(),
            })
        },
    )?;
    let mut routes: Vec<Vec<usize>> = Vec::new();
    let mut kinds: Vec<usize> = Vec::new();
    for b in built {
        for (kind, ids) in b.routes {
            kinds.push(kind);
            routes.push(ids);
        }
    }

    // Customers change depots while that lowers the cost, and each route
    // changed is ordered again from its own depot.
    let cost_of = |solution: &crate::models::Solution| -> f64 {
        let mut priced = solution.clone();
        priced.price(|r| &vehicles[r.vehicle_id()]);
        priced.total_cost()
    };
    for _ in 0..MAX_DEPOT_ROUNDS {
        let mut solution = crate::models::Solution::new();
        for (route, &kind) in routes.iter().zip(&kinds) {
            let (built, _) = RouteEvaluator::new(customers, dm, &vehicles[kind]).build_route(route);
            solution.add_route(built);
        }
        let before = cost_of(&solution);
        let moved = cross_depot_improve(&solution, customers, dm, vehicles);
        if cost_of(&moved) >= before - 1e-9 * (1.0 + before) {
            break;
        }
        kinds = moved.routes().iter().map(|r| r.vehicle_id()).collect();
        let limits: Vec<RouteLimits> = kinds
            .iter()
            .map(|&k| RouteLimits::of(&vehicles[k]))
            .collect();
        let depots_of: Vec<usize> = kinds.iter().map(|&k| vehicles[k].depot_id()).collect();
        let moved_routes: Vec<Vec<usize>> =
            moved.routes().iter().map(|r| r.customer_ids()).collect();
        routes = apply_local_search_from(&moved_routes, &depots_of, &limits, dm, customers).0;
    }
    let stop_reason = if stops.contains(&StopReason::Cancelled) {
        StopReason::Cancelled
    } else if stops.contains(&StopReason::TimeLimit) {
        StopReason::TimeLimit
    } else {
        stops
            .iter()
            .copied()
            .find(|s| *s != StopReason::Completed)
            .unwrap_or(StopReason::Completed)
    };
    Ok(Plan {
        routes,
        vehicles: kinds,
        method: planning.method,
        stop_reason,
    })
}

/// Most times the plan of several depots is improved and ordered again: each
/// round lowers the cost, so it ends sooner when no move gains.
const MAX_DEPOT_ROUNDS: usize = 8;

/// Refuses a request with pickup-delivery pairs that `method` or the fleet cannot plan yet, so that
/// what is not planned is said, not planned wrongly or quietly differently.
///
/// - The genetic algorithm builds a plan from one giant tour cut into routes, and a cut can part a
///   pair; the methods that place a pair as one unit (`nn`, `savings`, `alns`) are the ones that
///   plan pairs.
/// - A fleet of several kinds at one depot is brought to the plan route by route afterwards, by
///   cutting and refilling routes a customer at a time, which parts pairs. One kind of vehicle per
///   depot (any count, any number of depots) plans pairs as they are.
fn refuse_unsupported_pairs(
    method: Method,
    customers: &[Customer],
    fleet: &[VehicleType],
) -> Result<(), ServiceError> {
    if !has_pairs(customers) {
        return Ok(());
    }
    if method == Method::Genetic {
        return Err(ServiceError::new(
            Code::InvalidOption,
            "method \"ga\" does not place pickup-delivery pairs (customers with a delivery) -- \
             use \"alns\" for the best plans, or \"savings\" or \"nn\""
                .to_string(),
            json!({ "parameter": "method" }),
        ));
    }
    let mut kinds: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for t in fleet {
        *kinds.entry(t.vehicle().depot_id()).or_default() += 1;
    }
    if kinds.values().any(|&n| n > 1) {
        return Err(ServiceError::new(
            Code::InvalidOption,
            "pickup-delivery pairs are planned with one kind of vehicle per depot -- a depot \
             lists several kinds in vehicles; give it one (with a count if it has several)"
                .to_string(),
            json!({ "parameter": "vehicles" }),
        ));
    }
    Ok(())
}

/// Refuses a request with no depot.
fn require_depots(depots: &[(f64, f64)]) -> Result<(), ServiceError> {
    if depots.is_empty() {
        let mut fields = json!({ "parameter": "depots", "got": 0 });
        Bounds::of(1.0..).insert_into(&mut fields);
        return Err(ServiceError::new(
            Code::ParameterOutOfRange,
            "depots is empty -- a problem has at least one depot".to_string(),
            fields,
        ));
    }
    Ok(())
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
    /// The plan's largest route workload as reported (see `solve`'s
    /// `equity`); needs `config.balance`. Absent: not checked.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    equity: Option<f64>,
    /// The plan's objective as reported (see `solve`'s `objective`); needs
    /// `config.balance`. Absent: not checked.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    objective: Option<f64>,
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
    /// The vehicle's `break` cannot be placed on the route: the vehicle is still out when the break
    /// is due and no stop lets it start by its `latest` without closing a hard time window behind it.
    BreakNotPlaceable { route: usize },
    /// The route visits `delivery` before its `pickup`.
    PairOrder {
        route: usize,
        pickup: i64,
        delivery: i64,
    },
    /// `pickup` and its `delivery` are not on one route: on two routes, or one is on a route and
    /// the other is unassigned or on none.
    PairSplit { pickup: i64, delivery: i64 },
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
    /// The request has fewer vehicles of this kind than the plan has routes
    /// for it: `count` is its `vehicles[].count`; `route` is the first route
    /// past it.
    FleetCountExceeded {
        route: usize,
        vehicle: i64,
        count: usize,
    },
    /// More routes than `config.max_vehicles`.
    TooManyRoutes { routes: usize, max_routes: usize },
    /// The reported `total_distance` differs from the routes' own.
    DistanceMismatch { reported: f64, actual: f64 },
    /// The reported `total_cost` differs from what the routes cost with
    /// their vehicles.
    CostMismatch { reported: f64, actual: f64 },
    /// The reported `equity` differs from the routes' largest workload.
    EquityMismatch { reported: f64, actual: f64 },
    /// The reported `objective` differs from `total_cost` + weight × `equity`
    /// of the routes.
    ObjectiveMismatch { reported: f64, actual: f64 },
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
            Self::BreakNotPlaceable { route: 0 },
            Self::PairOrder {
                route: 0,
                pickup: 0,
                delivery: 0,
            },
            Self::PairSplit {
                pickup: 0,
                delivery: 0,
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
            Self::FleetCountExceeded {
                route: 0,
                vehicle: 0,
                count: 0,
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
            Self::EquityMismatch {
                reported: 0.0,
                actual: 0.0,
            },
            Self::ObjectiveMismatch {
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
    problem: &ProblemInput,
    method: &str,
    config: &InputConfig,
    solution: &InputSolution,
) -> Result<CheckOutput, ServiceError> {
    let ProblemInput {
        depots,
        customers: input_customers,
        vehicles: input_vehicles,
        distances,
        durations,
    } = *problem;
    use crate::evaluation::{check_solution, Fleet};
    use crate::models::{Route, Solution, ViolationType, Visit};

    let method = Method::parse(method)?;
    refuse_unread_settings(method, config)?;
    require_depots(depots)?;
    let max_routes = config
        .max_vehicles
        .map(|n| whole_in::<usize>("config.max_vehicles", n, 1, usize::MAX as i128))
        .transpose()?;
    let BuiltCustomers {
        customers,
        id_map,
        positions,
        ..
    } = build_customers(depots, input_customers, i32::MAX)?;
    let dm = distance_matrix(
        &customers,
        &positions,
        distances,
        durations,
        input_customers.len(),
    )?;
    let fleet_types = build_vehicles(input_vehicles, depots.len())?;
    refuse_unsupported_pairs(method, &customers, &fleet_types)?;
    let vehicles: Vec<Vehicle> = fleet_types.iter().map(|t| t.vehicle().clone()).collect();
    let counts: Vec<Option<usize>> = fleet_types.iter().map(VehicleType::count).collect();
    let balance = balance_of(config, input_customers, durations.is_some())?;
    if balance.is_some() && depots.len() > 1 {
        return Err(ServiceError::new(
            Code::InvalidOption,
            "config.balance is not supported with several depots yet -- it spans every route of the plan; leave it out".to_string(),
            json!({ "parameter": "config.balance" }),
        ));
    }
    if balance.is_none() {
        for (field, given) in [
            ("solution.equity", solution.equity.is_some()),
            ("solution.objective", solution.objective.is_some()),
        ] {
            if given {
                return Err(ServiceError::new(
                    Code::InvalidOption,
                    format!(
                        "{field} is measured by config.balance -- give it, or leave {field} out"
                    ),
                    json!({ "parameter": field }),
                ));
            }
        }
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
        .map(|(i, &id)| (id as i128, i + depots.len()))
        .collect();
    let mut caller_of: Vec<i128> = std::iter::repeat_n(-1, depots.len())
        .chain(id_map.iter().map(|&id| id as i128))
        .collect();
    let mut internal = |id: i128| -> usize {
        internal_of.get(&id).copied().unwrap_or_else(|| {
            caller_of.push(id);
            caller_of.len() - 1
        })
    };

    let mut plan = Solution::new();
    let mut actual_total = Some(0.0);
    let mut actual_cost = Some(0.0);
    let mut actual_equity = Some(0.0_f64);
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
            // The depot the route's vehicle starts from (the first, for a vehicle not in the fleet).
            let depot = vehicles.get(vehicle).map_or(0, Vehicle::depot_id);
            let d = route_distance_from(&internal_ids, depot, &dm);
            route.set_total_distance(d);
            // The lateness at soft windows is the route's own, like the distance: the plan reports
            // only a total cost, compared below.
            let lateness = route_eval(&internal_ids, depot, &dm, &customers, &RouteLimits::NONE)
                .map_or((0.0, 0.0), |e| (e.lateness, e.lateness_cost));
            route.set_lateness(lateness.0, lateness.1);
            actual_total = actual_total.map(|t| t + d);
            actual_cost = match (actual_cost, vehicles.get(vehicle)) {
                (Some(c), Some(v)) => Some(c + v.route_cost(d + lateness.1)),
                _ => None,
            };
            actual_equity = match (actual_equity, vehicles.get(vehicle), balance) {
                (Some(e), Some(v), Some(b)) => Some(
                    e.max(RouteFigures::of(&internal_ids, &dm, &customers).workload(b.quantity, v)),
                ),
                _ => None,
            };
        } else {
            actual_total = None;
            actual_cost = None;
            actual_equity = None;
        }
        plan.add_route(route);
    }
    for &id in &solution.unassigned {
        let i = internal(id);
        plan.add_unassigned(i);
    }

    let fleet = Fleet {
        vehicles: &vehicles,
        counts: &counts,
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
                ViolationType::BreakNotPlaceable { route_index } => {
                    PlanViolation::BreakNotPlaceable { route: route_index }
                }
                ViolationType::PairOrder {
                    route_index,
                    pickup_id,
                    delivery_id,
                } => PlanViolation::PairOrder {
                    route: route_index,
                    pickup: caller(pickup_id),
                    delivery: caller(delivery_id),
                },
                ViolationType::PairSplit {
                    pickup_id,
                    delivery_id,
                } => PlanViolation::PairSplit {
                    pickup: caller(pickup_id),
                    delivery: caller(delivery_id),
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
                ViolationType::FleetCountExceeded {
                    route_index, count, ..
                } => PlanViolation::FleetCountExceeded {
                    route: route_index,
                    vehicle: vehicle_given(route_index),
                    count,
                },
                ViolationType::TooManyRoutes { routes, max_routes } => {
                    PlanViolation::TooManyRoutes { routes, max_routes }
                }
                // Each route's distance was set to its own above.
                ViolationType::DistanceMismatch { .. } => return None,
                // Each route's lateness cost was set to its own above, like its distance.
                ViolationType::LatenessMismatch { .. } => return None,
            })
        })
        .collect();
    if let (Some(reported), Some(actual)) = (solution.total_distance, actual_total) {
        if (reported - actual).abs() > 1e-6 * actual.abs() {
            violations.push(PlanViolation::DistanceMismatch { reported, actual });
        }
    }
    if let (Some(reported), Some(actual)) = (solution.total_cost, actual_cost) {
        if (reported - actual).abs() > 1e-6 * actual.abs() {
            violations.push(PlanViolation::CostMismatch { reported, actual });
        }
    }
    if let (Some(reported), Some(actual)) = (solution.equity, actual_equity) {
        if (reported - actual).abs() > 1e-6 * actual.abs() {
            violations.push(PlanViolation::EquityMismatch { reported, actual });
        }
    }
    let actual_objective = match (balance, actual_cost, actual_equity) {
        (Some(b), Some(c), Some(e)) => Some(c + b.weight * e),
        _ => None,
    };
    if let (Some(reported), Some(actual)) = (solution.objective, actual_objective) {
        if (reported - actual).abs() > 1e-6 * actual.abs() {
            violations.push(PlanViolation::ObjectiveMismatch { reported, actual });
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
    /// The depots first, then every customer a vehicle can carry.
    customers: Vec<Customer>,
    /// The caller's id of the customer at `customers[depots + i]`.
    id_map: Vec<usize>,
    /// Every customer's id, in the order given -- what `unassigned` is read from.
    all_ids: Vec<usize>,
    /// The position in the request of the customer at `customers[depots + i]` --
    /// its row of `distances` is `depots` more.
    positions: Vec<usize>,
}

/// One customer of the request as `build_customers` has read it, before the table it is placed
/// in is known.
struct ReadCustomer {
    id: usize,
    demand: i32,
    customer: Customer,
    /// The position in the request of the customer this one is the pickup for, if it is one.
    delivery: Option<usize>,
}

/// For each customer (by position in the request), the position of the customer it is the pickup
/// for -- `None` for everyone else. A `delivery` that names no customer, the customer itself, a
/// customer with a demand of its own (a delivery unloads what its pickup loaded, so it asks for
/// nothing), a customer that is a pickup too, or one that another pickup names already is refused:
/// each of them would give a pair that cannot be one stop followed by one stop.
fn pair_deliveries(
    read: &[ReadCustomer],
    first_at: &std::collections::HashMap<usize, usize>,
) -> Result<Vec<Option<usize>>, ServiceError> {
    let mut deliveries: Vec<Option<usize>> = vec![None; read.len()];
    let mut named_by: Vec<Option<usize>> = vec![None; read.len()];
    for (position, r) in read.iter().enumerate() {
        let Some(delivery_id) = r.delivery else {
            continue;
        };
        let parameter = format!("customers[{position}].delivery");
        let refuse = |message: String, parameter: String| {
            ServiceError::new(
                Code::InvalidOption,
                message,
                json!({ "parameter": parameter }),
            )
        };
        let Some(&at) = first_at.get(&delivery_id) else {
            return Err(refuse(
                format!(
                    "customer {}: delivery {delivery_id} is not the id of any customer",
                    r.id
                ),
                parameter,
            ));
        };
        if at == position {
            return Err(refuse(
                format!("customer {}: a customer cannot be its own delivery", r.id),
                parameter,
            ));
        }
        let delivery = &read[at];
        if delivery.demand != 0 {
            return Err(refuse(
                format!(
                    "customer {delivery_id} is the delivery of customer {}: it unloads what the \
                     pickup loaded, so its own demand is 0 (the pickup's demand is what the pair \
                     carries)",
                    r.id
                ),
                format!("customers[{at}].demand"),
            ));
        }
        if delivery.delivery.is_some() {
            return Err(refuse(
                format!(
                    "customer {delivery_id} is the delivery of customer {} and cannot be the \
                     pickup of another pair",
                    r.id
                ),
                format!("customers[{at}].delivery"),
            ));
        }
        if let Some(first) = named_by[at] {
            return Err(ServiceError::new(
                Code::DuplicateId,
                format!(
                    "customer {delivery_id} is named as the delivery of the customers at \
                     positions {first} and {position} of customers (counting from 0); a delivery \
                     unloads one pickup"
                ),
                json!({ "id": delivery_id, "first": first, "second": position }),
            ));
        }
        named_by[at] = Some(position);
        deliveries[position] = Some(at);
    }
    Ok(deliveries)
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
    depots: &[(f64, f64)],
    input_customers: &[InputCustomer],
    largest_load: i32,
) -> Result<BuiltCustomers, ServiceError> {
    let mut customers: Vec<Customer> = Vec::with_capacity(input_customers.len() + depots.len());
    for (k, &(x, y)) in depots.iter().enumerate() {
        customers.push(Customer::depot_at(k, x, y));
    }

    // `id_map` names the customers a solver sees (every one a vehicle can
    // carry); `all_ids` names all of them, in the order given.
    let mut id_map: Vec<usize> = Vec::with_capacity(input_customers.len());
    let mut all_ids: Vec<usize> = Vec::with_capacity(input_customers.len());
    let mut positions: Vec<usize> = Vec::with_capacity(input_customers.len());
    let mut first_at: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(input_customers.len());
    // What each customer asks for once read, by position in the request: whether a solver sees
    // it is decided after the pairs are known, since a pair is left out whole.
    let mut read: Vec<ReadCustomer> = Vec::with_capacity(input_customers.len());

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
        // Numbered when it is placed in the table, below.
        let mut c = Customer::new(0, ic.x, ic.y, demand, ic.service_time);
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
        if let Some(penalty) = ic.lateness_penalty {
            // A penalty is a number of at least 0; the model refuses what is not (a NaN from a
            // caller that did not come through the bindings' finiteness check included).
            c = c.with_lateness_penalty(penalty).ok_or_else(|| {
                let mut fields = json!({
                    "parameter": "lateness_penalty", "id": id, "index": position, "got": penalty,
                });
                Bounds::of(0.0..).insert_into(&mut fields);
                ServiceError::new(
                    Code::ParameterOutOfRange,
                    format!("customer {id}: lateness_penalty {penalty} must be at least 0"),
                    fields,
                )
            })?;
        }
        let delivery = ic
            .delivery
            .map(|d| {
                let parameter = format!("customers[{position}].delivery");
                whole_in::<usize>(&parameter, d, 0, usize::MAX as i128)
            })
            .transpose()?;
        read.push(ReadCustomer {
            id,
            demand,
            customer: c,
            delivery,
        });
    }

    let deliveries = pair_deliveries(&read, &first_at)?;

    // A customer asking for more than the largest vehicle takes is left out, and so is the
    // customer it is paired with: a pickup too big for every vehicle takes its delivery along (a
    // delivery asks for nothing, so it is never left out on its own account).
    let mut seen: Vec<bool> = read.iter().map(|r| r.demand <= largest_load).collect();
    for (pickup, delivery) in deliveries.iter().enumerate() {
        if let Some(delivery) = *delivery {
            seen[delivery] = seen[pickup];
        }
    }
    // Where each customer a solver sees is in the table: the depots, then those in the order given.
    let mut row_of: Vec<Option<usize>> = vec![None; read.len()];
    let mut next_row = customers.len();
    for (position, &kept) in seen.iter().enumerate() {
        if kept {
            row_of[position] = Some(next_row);
            next_row += 1;
        }
    }
    let mut pickup_of: Vec<Option<usize>> = vec![None; read.len()];
    for (pickup, delivery) in deliveries.iter().enumerate() {
        if let Some(delivery) = *delivery {
            pickup_of[delivery] = Some(pickup);
        }
    }
    for (position, r) in read.into_iter().enumerate() {
        let Some(row) = row_of[position] else {
            continue;
        };
        let partner = |other: Option<usize>| other.and_then(|p| row_of[p]);
        id_map.push(r.id);
        positions.push(position);
        customers.push(
            r.customer
                .placed(row, false)
                .with_partner(partner(deliveries[position]), partner(pickup_of[position])),
        );
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
                "the demands of the customers a vehicle can carry add up to {total}; the model counts a route's load as a whole number of units up to {}, so they must add up to no more -- use a larger unit (kilograms for grams, say)",
                i32::MAX
            ),
            fields,
        ));
    }

    Ok(BuiltCustomers {
        customers,
        id_map,
        all_ids,
        positions,
    })
}

/// The lengths and travel times every method plans with. `distances` when the
/// request gives it, else the Euclidean distances between the coordinates;
/// `durations` when it gives that, else travel time equals distance. Both are
/// read as rows and columns `0..D` for the `D` depots and `D + p` for the
/// customer at position `p` of `customers`.
///
/// Each is refused unless it is square with one row per point (the depots and
/// every customer, `inputs` of them) and its entries are finite and not
/// negative. Neither need be symmetric: a one-way street is a shorter way
/// there than back, and the moves that reverse part of a route price the
/// reversed edges when the matrix is not (and check the windows on the travel
/// times again).
fn distance_matrix(
    customers: &[Customer],
    positions: &[usize],
    distances: Option<&[Vec<f64>]>,
    durations: Option<&[Vec<f64>]>,
    inputs: usize,
) -> Result<DistanceMatrix, ServiceError> {
    // Internal point k is a depot (k below the number of depots) or customers[k],
    // the one at request position positions[k - depots].
    let depots = num_depots(customers);
    let row_of = |k: usize| {
        if k < depots {
            k
        } else {
            positions[k - depots] + depots
        }
    };
    let n = customers.len();
    let internal = |rows: &[Vec<f64>]| -> Vec<f64> {
        let mut data = Vec::with_capacity(n * n);
        for a in 0..n {
            for b in 0..n {
                data.push(rows[row_of(a)][row_of(b)]);
            }
        }
        data
    };
    let matrix = match distances {
        Some(rows) => {
            check_grid("distances", "distance", rows, inputs, depots)?;
            DistanceMatrix::from_data(n, internal(rows)).expect("n × n entries")
        }
        None => DistanceMatrix::from_customers(customers),
    };
    match durations {
        Some(rows) => {
            check_grid("durations", "travel time", rows, inputs, depots)?;
            Ok(matrix
                .with_travel_times(internal(rows))
                .expect("n × n entries"))
        }
        None => Ok(matrix),
    }
}

/// Refuses `rows` (the request's `parameter`, entries of `what`) unless it has
/// one row per point -- the depot and `inputs` customers -- each as long, of
/// finite entries not below 0.
fn check_grid(
    parameter: &str,
    what: &str,
    rows: &[Vec<f64>],
    inputs: usize,
    depots: usize,
) -> Result<(), ServiceError> {
    let size = inputs + depots;
    if rows.len() != size {
        return Err(ServiceError::new(
            Code::DimensionMismatch,
            if depots == 1 {
                format!(
                    "{parameter} has {} rows for the depot and {inputs} customers -- one row per point, the depot first",
                    rows.len()
                )
            } else {
                format!(
                    "{parameter} has {} rows for the {depots} depots and {inputs} customers -- one row per point, the depots first",
                    rows.len()
                )
            },
            json!({ "parameter": parameter, "expected": size, "got": rows.len() }),
        ));
    }
    for (i, row) in rows.iter().enumerate() {
        let at = format!("{parameter}[{i}]");
        if row.len() != size {
            return Err(ServiceError::new(
                Code::DimensionMismatch,
                format!(
                    "{at} has {} entries -- one per point, {size} in all",
                    row.len()
                ),
                json!({ "parameter": at, "expected": size, "got": row.len() }),
            ));
        }
        for (j, &d) in row.iter().enumerate() {
            if !d.is_finite() {
                return Err(ServiceError::new(
                    Code::ValueNotFinite,
                    format!("{at}[{j}] is {d} -- a {what} is a finite number"),
                    json!({ "parameter": at, "index": j }),
                ));
            }
            if d < 0.0 {
                let mut fields = json!({ "parameter": at, "index": j, "got": d });
                Bounds::of(0.0..).insert_into(&mut fields);
                return Err(ServiceError::new(
                    Code::ParameterOutOfRange,
                    format!("{at}[{j}] is {d} -- a {what} is at least 0"),
                    fields,
                ));
            }
        }
    }
    Ok(())
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
/// `config.balance` as the solver reads it, refused when its quantity is not
/// one it knows, its weight is not a number `>= 0`, or it balances
/// `"duration"` on a problem with no time in it (no travel times, service
/// times or time windows), where the duration is only the distance again.
fn balance_of(
    cfg: &InputConfig,
    customers: &[InputCustomer],
    travel_times: bool,
) -> Result<Option<Balance>, ServiceError> {
    let Some(input) = &cfg.balance else {
        return Ok(None);
    };
    let quantity = Quantity::parse(&input.quantity).ok_or_else(|| {
        ServiceError::new(
            Code::UnknownOption,
            format!(
                "unknown config.balance.quantity '{}'. Supported: \"load\", \"distance\", \"duration\"",
                input.quantity
            ),
            json!({ "parameter": "config.balance.quantity", "got": input.quantity,
                    "expected": Quantity::NAMES }),
        )
    })?;
    let weight = input.weight;
    if !weight.is_finite() {
        // Only JavaScript can send one (the binding refuses it first).
        return Err(ServiceError::new(
            Code::ValueNotFinite,
            format!("config.balance.weight must be finite, got {weight}"),
            json!({ "parameter": "config.balance.weight" }),
        ));
    }
    if weight < 0.0 {
        let mut fields = json!({ "parameter": "config.balance.weight", "got": weight });
        Bounds::of((Bound::Included(0.0), Bound::Unbounded)).insert_into(&mut fields);
        return Err(ServiceError::new(
            Code::ParameterOutOfRange,
            format!("config.balance.weight must be >= 0, got {weight}"),
            fields,
        ));
    }
    if quantity == Quantity::Duration
        && !travel_times
        && customers.iter().all(InputCustomer::is_untimed)
    {
        return Err(ServiceError::new(
            Code::InvalidOption,
            "config.balance.quantity \"duration\" needs time in the problem -- travel times \
             (durations), service times or time windows; without them a route's duration is its \
             distance: balance \"distance\""
                .to_string(),
            json!({ "parameter": "config.balance.quantity" }),
        ));
    }
    Ok(Some(Balance { quantity, weight }))
}

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
        cfg.balance.is_some(),
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

/// Converts internal route indices back to original customer IDs; `depots` is
/// the number of depots, the entries before the first customer.
fn map_routes(routes: &[Vec<usize>], id_map: &[usize], depots: usize) -> Vec<Vec<usize>> {
    routes
        .iter()
        .map(|route| {
            route
                .iter()
                .map(|&internal_idx| id_map[internal_idx - depots])
                .collect()
        })
        .collect()
}

/// What a route is priced at: its distance plus the cost of its lateness at soft windows (the
/// distance alone when every window is hard).
fn priced_distance(route: &[usize], dm: &DistanceMatrix, customers: &[Customer]) -> f64 {
    priced_distance_from(route, 0, dm, customers)
}

/// [`priced_distance`] of a route that starts and ends at `depot`.
fn priced_distance_from(
    route: &[usize],
    depot: usize,
    dm: &DistanceMatrix,
    customers: &[Customer],
) -> f64 {
    if has_soft_windows(customers) {
        if let Some(eval) = route_eval(route, depot, dm, customers, &RouteLimits::NONE) {
            return eval.priced_distance();
        }
    }
    route_distance_from(route, depot, dm)
}

/// Depot, `route`'s customers in order, depot: the length of the round trip.
fn route_distance(route: &[usize], dm: &DistanceMatrix) -> f64 {
    route_distance_from(route, 0, dm)
}

/// [`route_distance`] of a route that starts and ends at `depot`.
fn route_distance_from(route: &[usize], depot: usize, dm: &DistanceMatrix) -> f64 {
    let mut prev = depot;
    let mut d = 0.0;
    for &c in route {
        d += dm.get(prev, c);
        prev = c;
    }
    d + dm.get(prev, depot)
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
    apply_local_search_from(routes, &vec![0; routes.len()], limits, dm, customers)
}

/// [`apply_local_search`] with the depot each route starts from.
fn apply_local_search_from(
    routes: &[Vec<usize>],
    depots: &[usize],
    limits: &[RouteLimits],
    dm: &DistanceMatrix,
    customers: &[Customer],
) -> (Vec<Vec<usize>>, f64) {
    let mut improved_routes = Vec::with_capacity(routes.len());
    let mut total = 0.0;
    for ((route, limits), &depot) in routes.iter().zip(limits).zip(depots) {
        let (r1, _) = two_opt_improve(route, depot, dm, customers, limits);
        let (r2, dist) = or_opt_improve(&r1, depot, dm, customers, limits);
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

/// Builds the fleet from input: its kinds of vehicle in the order given, each
/// with its count and its depot. No `vehicles` is one kind of unlimited
/// capacity at every depot.
fn build_vehicles(
    input_vehicles: &[InputVehicle],
    depots: usize,
) -> Result<Vec<VehicleType>, ServiceError> {
    if input_vehicles.is_empty() {
        // One kind of unlimited capacity and number at every depot.
        return Ok((0..depots)
            .map(|d| VehicleType::new(Vehicle::new(d, i32::MAX).with_depot(d), None))
            .collect());
    }
    let types: Vec<VehicleType> = input_vehicles
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
            if let Some(b) = &v.rest {
                let refuse = |parameter: &str, got: f64, bounds: Bounds, text: String| {
                    let mut fields = json!({ "parameter": parameter, "index": i, "got": got });
                    bounds.insert_into(&mut fields);
                    ServiceError::new(Code::ParameterOutOfRange, text, fields)
                };
                if b.earliest < 0.0 {
                    return Err(refuse(
                        "break.earliest",
                        b.earliest,
                        Bounds::of((Bound::Included(0.0), Bound::Unbounded)),
                        format!("vehicle {i}: break.earliest must be >= 0, got {}", b.earliest),
                    ));
                }
                if b.latest < b.earliest {
                    return Err(refuse(
                        "break.latest",
                        b.latest,
                        Bounds::of((Bound::Included(b.earliest), Bound::Unbounded)),
                        format!(
                            "vehicle {i}: break.latest must be >= break.earliest ({}), got {}",
                            b.earliest, b.latest
                        ),
                    ));
                }
                if b.duration <= 0.0 {
                    return Err(refuse(
                        "break.duration",
                        b.duration,
                        Bounds::of((Bound::Excluded(0.0), Bound::Unbounded)),
                        format!("vehicle {i}: break.duration must be > 0, got {}", b.duration),
                    ));
                }
                vehicle = vehicle.with_break(crate::models::VehicleBreak::new(
                    b.earliest, b.latest, b.duration,
                ));
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
            let count = v
                .count
                .map(|n| {
                    whole_in::<usize>("count", n, 1, usize::MAX as i128).map_err(|mut e| {
                        e.fields["index"] = json!(i);
                        e
                    })
                })
                .transpose()?;
            let depot = match v.depot {
                None => 0,
                Some(n) => {
                    let depot = whole_in::<usize>("depot", n, 0, usize::MAX as i128).map_err(
                        |mut e| {
                            e.fields["index"] = json!(i);
                            e
                        },
                    )?;
                    if depot >= depots {
                        let mut fields =
                            json!({ "parameter": "depot", "index": i, "got": depot as f64 });
                        Bounds::of(0.0..=(depots - 1) as f64).insert_into(&mut fields);
                        return Err(ServiceError::new(
                            Code::ParameterOutOfRange,
                            format!(
                                "vehicle {i}: depot must be an index into depots, from 0 to {}, got {depot}",
                                depots - 1
                            ),
                            fields,
                        ));
                    }
                    depot
                }
            };
            Ok(VehicleType::new(
                vehicle
                    .with_depot(depot)
                    .with_fixed_cost(v.fixed_cost)
                    .with_cost_per_distance(v.cost_per_distance),
                count,
            ))
        })
        .collect::<Result<_, ServiceError>>()?;
    // A route is timed once and then given to the kind that carries it cheapest
    // (`assign_fleet`): that is sound only when every kind keeps the same clock.
    if types
        .windows(2)
        .any(|w| w[0].vehicle().rest() != w[1].vehicle().rest())
    {
        return Err(ServiceError::new(
            Code::InvalidOption,
            "every vehicle kind must have the same break (or none): a route is timed once and then given to the kind that carries it cheapest".to_string(),
            json!({ "parameter": "vehicles[].break" }),
        ));
    }
    Ok(types)
}

/// The kind of vehicle the methods that plan every route alike plan with: the
/// one that carries the most (then the one that costs least per unit of
/// distance travelled, then the first listed). Every route they make fits it.
fn planning_vehicle(vehicles: &[Vehicle]) -> usize {
    let mut best = 0;
    for (i, v) in vehicles.iter().enumerate().skip(1) {
        let b = &vehicles[best];
        let better = v.capacity() > b.capacity()
            || (v.capacity() == b.capacity()
                && (v.fixed_cost_in_distance(), v.cost_per_distance())
                    < (b.fixed_cost_in_distance(), b.cost_per_distance()));
        if better {
            best = i;
        }
    }
    best
}

/// The kinds a fleet of several is planned with: each distinct kind that can
/// carry the largest customer, the one that carries the most first.
fn planning_kinds(fleet: &[VehicleType], customers: &[Customer]) -> Vec<usize> {
    let largest = customers.iter().map(Customer::demand).max().unwrap_or(0);
    let mut kinds: Vec<usize> = Vec::new();
    for (k, t) in fleet.iter().enumerate() {
        let duplicate = kinds.iter().any(|&j| t.same_kind_as(&fleet[j]));
        if !duplicate && t.vehicle().capacity() >= largest {
            kinds.push(k);
        }
    }
    if kinds.is_empty() {
        // Only possible when a customer asks for more than every kind carries,
        // which `build_customers` has already set aside.
        kinds.push(planning_vehicle(
            &fleet
                .iter()
                .map(|t| t.vehicle().clone())
                .collect::<Vec<_>>(),
        ));
    }
    kinds
}

/// Cuts `route`, in its order, into pieces that `vehicle` carries: each piece
/// takes customers until the next would overload it or break the vehicle's
/// limits and time windows. A piece of one customer may still be more than the
/// vehicle can do; the assignment that follows is what decides.
fn split_route(
    route: &[usize],
    vehicle: &Vehicle,
    customers: &[Customer],
    dm: &DistanceMatrix,
) -> Vec<Vec<usize>> {
    let limits = RouteLimits::of(vehicle);
    let constrained = route_checks_needed(customers, &limits);
    let mut pieces: Vec<Vec<usize>> = Vec::new();
    let mut load = 0;
    for &c in route {
        let demand = customers[c].demand();
        if let Some(current) = pieces.last_mut() {
            let mut fits = load + demand <= vehicle.capacity();
            if fits && constrained {
                current.push(c);
                fits = route_feasible(current, 0, dm, customers, &limits);
                current.pop();
            }
            if fits {
                current.push(c);
                load += demand;
                continue;
            }
        }
        pieces.push(vec![c]);
        load = demand;
    }
    pieces
}

/// Gives every route of `plan` the kind of vehicle that carries it cheapest,
/// within the counts of the fleet. A route that no vehicle left can carry is cut
/// into pieces that the kinds still available carry; what the fleet still cannot
/// take is emptied: those customers go into the routes that remain wherever
/// they fit, and the ones that fit nowhere are left unassigned, as with a
/// fixed fleet.
fn assign_fleet(
    plan: Plan,
    fleet: &[VehicleType],
    customers: &[Customer],
    dm: &DistanceMatrix,
) -> Plan {
    let template = fleet[0].vehicle();
    let demand_of = |route: &[usize]| {
        let (built, _) = RouteEvaluator::new(customers, dm, template).build_route(route);
        RouteDemand {
            load: route.iter().map(|&c| customers[c].demand()).sum(),
            distance: route_distance(route, dm),
            duration: built.total_duration(),
            lateness_cost: built.lateness_cost(),
        }
    };
    let assign = |routes: &[Vec<usize>]| -> Vec<Option<usize>> {
        let demands: Vec<RouteDemand> = routes.iter().map(|r| demand_of(r)).collect();
        let sizes: Vec<usize> = routes.iter().map(Vec::len).collect();
        assign_types(&demands, &sizes, fleet)
    };
    let everyone: Vec<usize> = plan.routes.iter().flatten().copied().collect();
    let mut routes = plan.routes;
    let mut given = assign(&routes);
    // A route no vehicle left carries is cut for the kinds still available.
    for _ in 0..4 {
        if given.iter().all(Option::is_some) {
            break;
        }
        let mut used = vec![0usize; fleet.len()];
        for kind in given.iter().flatten() {
            used[*kind] += 1;
        }
        let mut cutter: Option<usize> = None;
        for (k, t) in fleet.iter().enumerate() {
            let free = t.count().is_none_or(|c| used[k] < c);
            if free && cutter.is_none_or(|b| t.vehicle().capacity() > fleet[b].vehicle().capacity())
            {
                cutter = Some(k);
            }
        }
        let Some(cutter) = cutter else { break };
        let mut cut = false;
        let mut next: Vec<Vec<usize>> = Vec::with_capacity(routes.len());
        for (route, slot) in routes.into_iter().zip(&given) {
            if slot.is_some() {
                next.push(route);
                continue;
            }
            let pieces = split_route(&route, fleet[cutter].vehicle(), customers, dm);
            cut |= pieces.len() > 1;
            next.extend(pieces);
        }
        routes = next;
        given = assign(&routes);
        if !cut {
            break;
        }
    }
    if given.iter().any(Option::is_none) {
        let mut kept: Vec<Vec<usize>> = Vec::new();
        let mut kept_types: Vec<usize> = Vec::new();
        let mut pool: Vec<usize> = Vec::new();
        for (route, slot) in routes.into_iter().zip(&given) {
            match slot {
                Some(t) => {
                    kept.push(route);
                    kept_types.push(*t);
                }
                None => pool.extend(route),
            }
        }
        // The largest demands first: they have the fewest places to go.
        pool.sort_by_key(|&c| (std::cmp::Reverse(customers[c].demand()), c));
        for c in pool {
            let placed = cheapest_insertion(
                &kept,
                |ri| fleet[kept_types[ri]].vehicle().capacity(),
                |ri| RouteLimits::of(fleet[kept_types[ri]].vehicle()),
                c,
                dm,
                customers,
            );
            if let Some((ri, pos, _)) = placed {
                kept[ri].insert(pos, c);
            }
        }
        routes = kept;
        // Customers were left out. The routes were shaped for one kind of
        // vehicle, so what the real fleet could not take says little about what
        // it can carry: loading the vehicles first is tried as well, and stands
        // when it serves more customers.
        let served: usize = routes.iter().map(Vec::len).sum();
        if let Some((packed, kinds)) = pack_fleet(&everyone, fleet, customers, dm) {
            if packed.iter().map(Vec::len).sum::<usize>() > served {
                let limits: Vec<RouteLimits> = kinds
                    .iter()
                    .map(|&k| RouteLimits::of(fleet[k].vehicle()))
                    .collect();
                let (routes, _) = apply_local_search(&packed, &limits, dm, customers);
                return Plan {
                    routes,
                    vehicles: kinds,
                    method: plan.method,
                    stop_reason: plan.stop_reason,
                };
            }
        }
        // The vehicles each kept route already has still carry it, so this
        // assigns every route -- at least as cheaply as before.
        given = assign(&routes);
    }
    let types: Vec<usize> = given
        .into_iter()
        .map(|t| t.expect("every kept route has a vehicle that carries it"))
        .collect();
    Plan {
        routes,
        vehicles: types,
        method: plan.method,
        stop_reason: plan.stop_reason,
    }
}

/// Loads a fleet of counted vehicles before routing it; `None` when a kind
/// has no `count` (such a fleet is never short of vehicles). See
/// [`pack_vehicles`].
fn pack_fleet(
    everyone: &[usize],
    fleet: &[VehicleType],
    customers: &[Customer],
    dm: &DistanceMatrix,
) -> Option<(Vec<Vec<usize>>, Vec<usize>)> {
    let mut kinds: Vec<usize> = Vec::new();
    for (k, t) in fleet.iter().enumerate() {
        kinds.extend(std::iter::repeat_n(k, t.count()?));
    }
    Some(pack_vehicles(
        everyone,
        kinds,
        |k| fleet[k].vehicle(),
        customers,
        dm,
    ))
}

/// Loads the vehicles before routing them: the customers, the largest demand
/// first, each go to the vehicle with the least room left that still takes
/// them (best-fit decreasing; Johnson 1973), at the cheapest place on its
/// route that keeps the vehicle's limits and every time window. `kinds` names
/// the kind of each vehicle there is; the routes that carry someone come back
/// with the kind of each.
///
/// A plan made route-first leaves a customer out when its routes do not divide
/// among the vehicles there are -- shaped for one kind of a mixed fleet, or
/// more of them than the fleet has -- although the fleet has the capacity.
/// Loading first does not have that blind spot; it pays for it in distance, so
/// it is only a second candidate.
///
/// # Complexity
/// O(n · V · n) for n customers and V vehicles (each placement tries a route's
/// positions), plus the window checks when customers carry windows.
fn pack_vehicles<'a>(
    everyone: &[usize],
    kinds: Vec<usize>,
    vehicle_of: impl Fn(usize) -> &'a Vehicle,
    customers: &[Customer],
    dm: &DistanceMatrix,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut routes: Vec<Vec<usize>> = vec![Vec::new(); kinds.len()];
    let mut order: Vec<usize> = everyone.to_vec();
    order.sort_by_key(|&c| (std::cmp::Reverse(customers[c].demand()), c));
    for c in order {
        let demand = customers[c].demand();
        // The vehicles that have room, the fullest first; equal room resolves
        // to the earlier vehicle, so the same request gives the same plan.
        let mut room: Vec<(i32, usize)> = routes
            .iter()
            .enumerate()
            .filter_map(|(v, route)| {
                let load: i32 = route.iter().map(|&x| customers[x].demand()).sum();
                let left = vehicle_of(kinds[v]).capacity() - load - demand;
                (left >= 0).then_some((left, v))
            })
            .collect();
        room.sort_unstable();
        for (_, v) in room {
            let placed = cheapest_insertion(
                std::slice::from_ref(&routes[v]),
                |_| vehicle_of(kinds[v]).capacity(),
                |_| RouteLimits::of(vehicle_of(kinds[v])),
                c,
                dm,
                customers,
            );
            if let Some((_, pos, _)) = placed {
                routes[v].insert(pos, c);
                break;
            }
        }
    }
    routes
        .into_iter()
        .zip(kinds)
        .filter(|(route, _)| !route.is_empty())
        .unzip()
}

// ============================================================================
// Solver methods
// ============================================================================

fn solve_nn(customers: &[Customer], dm: &DistanceMatrix, fleet: &[VehicleType]) -> Plan {
    // Nearest neighbour drives each route with a vehicle of its own, taken in
    // the order of the fleet: a kind of `count` vehicles is that many in a row
    // (no more than there are customers to drive).
    // Every copy keeps its kind's id -- its index in the fleet (`build_vehicles`)
    // -- so a route's `vehicle_id` is the kind that drove it.
    let mut vehicles: Vec<Vehicle> = Vec::new();
    for t in fleet {
        let copies = t.count().unwrap_or(customers.len()).min(customers.len());
        vehicles.extend(std::iter::repeat_n(t.vehicle().clone(), copies));
    }
    let solution = if has_time_windows(customers) {
        nearest_neighbor_tw(customers, dm, &vehicles)
    } else {
        nearest_neighbor(customers, dm, &vehicles)
    };
    Plan {
        routes: solution.routes().iter().map(|r| r.customer_ids()).collect(),
        vehicles: solution.routes().iter().map(|r| r.vehicle_id()).collect(),
        method: Method::NearestNeighbor,
        stop_reason: StopReason::Completed,
    }
}

fn solve_savings(
    customers: &[Customer],
    dm: &DistanceMatrix,
    vehicle: &Vehicle,
    kind: usize,
) -> Plan {
    let mut solution = clarke_wright_savings(customers, dm, vehicle);
    // Savings builds a pair as a route of its own and merges routes whole, so what is left over
    // is pairs that belong in another route's gaps: moving each pair to its cheapest place is
    // part of the method here, not an option (on pickup-delivery benchmarks it takes the
    // distance over the best known from 119 % to 22 %).
    if has_pairs(customers) {
        solution = pair_relocate_improve(&solution, customers, dm, vehicle);
    }
    let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    Plan {
        vehicles: vec![kind; routes.len()],
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
    kind: usize,
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
    // Offspring are kept apart in cost (Prins 2004's spacing rule): clones
    // otherwise fill the population and end the search at its stagnation limit.
    let mut ga_config = GaConfig::default()
        .with_population_size(population_size)
        .with_max_generations(max_generations)
        .with_distinct_fitness(true)
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
        vehicles: vec![kind; routes.len()],
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
    kind: usize,
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
        vehicles: vec![kind; routes.len()],
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
        let mut seen = [false; 18];
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
                PlanViolation::FleetCountExceeded { .. } => 9,
                PlanViolation::TooManyRoutes { .. } => 10,
                PlanViolation::DistanceMismatch { .. } => 11,
                PlanViolation::CostMismatch { .. } => 12,
                PlanViolation::EquityMismatch { .. } => 13,
                PlanViolation::ObjectiveMismatch { .. } => 14,
                PlanViolation::BreakNotPlaceable { .. } => 15,
                PlanViolation::PairOrder { .. } => 16,
                PlanViolation::PairSplit { .. } => 17,
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
        solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &detour_customers(),
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            method,
            &config,
        )
        .expect("solves")
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

    /// Three customers on one spot; the matrix says 2 is far from 1 and 3.
    fn matrix_problem() -> (Vec<InputCustomer>, Vec<Vec<f64>>) {
        let customers = (1..=3)
            .map(|id| customer(serde_json::json!({ "id": id, "x": 0.0, "y": 0.0, "demand": 1 })))
            .collect();
        let distances = vec![
            vec![0.0, 1.0, 5.0, 1.0],
            vec![1.0, 0.0, 9.0, 1.0],
            vec![5.0, 9.0, 0.0, 9.0],
            vec![1.0, 1.0, 9.0, 0.0],
        ];
        (customers, distances)
    }

    #[test]
    fn given_distances_replace_the_coordinates() {
        let (customers, distances) = matrix_problem();
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: Some(&distances),
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
            // The coordinates put every point on the depot (distance 0); the
            // matrix makes any tour at least 1 + 1 + 5 + ... -- the best one
            // depot-1-3-depot plus depot-2-depot, or one route through all.
            assert!(
                out.total_distance >= 13.0 - 1e-9,
                "{method}: {}",
                out.total_distance
            );
            let mut served: Vec<usize> = out.routes.concat();
            served.sort_unstable();
            assert_eq!(served, vec![1, 2, 3], "{method}");

            let plan: InputSolution = serde_json::from_value(serde_json::json!({
                "routes": out.routes,
                "route_vehicles": out.route_vehicles,
                "total_distance": out.total_distance,
            }))
            .expect("valid plan");
            let checked = check(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: Some(&distances),
                    durations: None,
                },
                method,
                &quick(method),
                &plan,
            )
            .expect("checks");
            assert!(checked.feasible, "{method}: {:?}", checked.violations);
        }
    }

    #[test]
    fn a_one_way_matrix_is_planned_in_the_direction_it_is_short() {
        // Around the ring depot -> 1 -> 2 -> 3 -> depot every leg is 1; every
        // leg the other way is 10. The best plan is that ring, length 4.
        let customers: Vec<InputCustomer> = (1..=3)
            .map(|id| customer(serde_json::json!({ "id": id, "x": 0.0, "y": 0.0, "demand": 1 })))
            .collect();
        let mut distances = vec![vec![10.0; 4]; 4];
        for (a, row) in distances.iter_mut().enumerate() {
            row[a] = 0.0;
            row[(a + 1) % 4] = 1.0;
        }
        for method in ["nn", "savings", "ga", "alns"] {
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: Some(&distances),
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
            let walked: f64 = out
                .routes
                .iter()
                .map(|r| {
                    let mut prev = 0;
                    let mut d = 0.0;
                    for &id in r {
                        d += distances[prev][id];
                        prev = id;
                    }
                    d + distances[prev][0]
                })
                .sum();
            assert!((out.total_distance - walked).abs() < 1e-9, "{method}");
            if method != "nn" {
                assert!(
                    (out.total_distance - 4.0).abs() < 1e-9,
                    "{method}: {:?}",
                    out.routes
                );
            }
        }
    }

    #[test]
    fn durations_keep_the_clock_and_distances_the_length() {
        // Every leg takes ten times its length. Customer 1 is one unit away
        // but due at 5: by distance it is reachable, by time it is not.
        let (mut customers, distances) = matrix_problem();
        let durations: Vec<Vec<f64>> = distances
            .iter()
            .map(|row| row.iter().map(|d| d * 10.0).collect())
            .collect();
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: Some(&distances),
                durations: Some(&durations),
            },
            "nn",
            &InputConfig::default(),
        )
        .expect("solves");
        for (route, summary) in out.routes.iter().zip(&out.route_summaries) {
            let walk = |m: &[Vec<f64>]| {
                let mut prev = 0;
                let mut d = 0.0;
                for &id in route {
                    d += m[prev][id];
                    prev = id;
                }
                d + m[prev][0]
            };
            assert!((summary.distance - walk(&distances)).abs() < 1e-9);
            assert!((summary.duration - walk(&durations)).abs() < 1e-9);
        }

        customers[0] = customer(serde_json::json!({
            "id": 1, "x": 0.0, "y": 0.0, "demand": 1, "time_window": [0.0, 5.0]
        }));
        let by_distance = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: Some(&distances),
                durations: None,
            },
            "nn",
            &InputConfig::default(),
        )
        .expect("solves");
        assert!(by_distance.unassigned.is_empty());
        for method in ["nn", "savings", "ga", "alns"] {
            let by_time = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: Some(&distances),
                    durations: Some(&durations),
                },
                method,
                &quick(method),
            )
            .expect(method);
            assert_eq!(by_time.unassigned, vec![1], "{method}");
        }

        let mut short = durations.clone();
        short[3].pop();
        let e = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: None,
                durations: Some(&short),
            },
            "nn",
            &InputConfig::default(),
        )
        .expect_err("refused");
        assert_eq!(e.fields["code"], "dimension_mismatch");
        assert_eq!(e.fields["parameter"], "durations[3]");
    }

    #[test]
    fn a_customer_left_out_keeps_the_others_on_their_rows() {
        // Customer 2 asks for more than the vehicle carries and is reported
        // unassigned; 1 and 3 must still read rows 1 and 3, not 1 and 2.
        let (mut customers, distances) = matrix_problem();
        customers[1] = customer(serde_json::json!({ "id": 2, "x": 0.0, "y": 0.0, "demand": 5 }));
        let truck = [vehicle(serde_json::json!({ "capacity": 2 }))];
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &truck,
                distances: Some(&distances),
                durations: None,
            },
            "nn",
            &InputConfig::default(),
        )
        .expect("solves");
        assert_eq!(out.unassigned, vec![2]);
        // depot-1-3-depot = 1 + 1 + 1 (row 2 would make it 1 + 9 + 5).
        assert!(
            (out.total_distance - 3.0).abs() < 1e-9,
            "{}",
            out.total_distance
        );
    }

    #[test]
    fn a_distance_matrix_of_the_wrong_shape_or_values_is_refused() {
        let (customers, distances) = matrix_problem();
        let refuse = |d: Vec<Vec<f64>>| {
            solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: Some(&d),
                    durations: None,
                },
                "nn",
                &InputConfig::default(),
            )
            .expect_err("refused")
        };

        let e = refuse(distances[..3].to_vec());
        assert_eq!(e.fields["code"], "dimension_mismatch");
        assert_eq!(e.fields["parameter"], "distances");
        assert_eq!(
            (e.fields["expected"].clone(), e.fields["got"].clone()),
            (4.into(), 3.into())
        );

        let mut short = distances.clone();
        short[2].pop();
        let e = refuse(short);
        assert_eq!(e.fields["code"], "dimension_mismatch");
        assert_eq!(e.fields["parameter"], "distances[2]");

        let mut negative = distances.clone();
        negative[1][3] = -1.0;
        negative[3][1] = -1.0;
        let e = refuse(negative);
        assert_eq!(e.fields["code"], "parameter_out_of_range");
        assert_eq!(e.fields["parameter"], "distances[1]");
        assert_eq!(e.fields["index"], 3);
        assert_eq!(e.fields["min"], 0.0);

        let mut nan = distances.clone();
        nan[0][2] = f64::NAN;
        let e = refuse(nan);
        assert_eq!(e.fields["code"], "value_not_finite");
        assert_eq!(
            (e.fields["parameter"].clone(), e.fields["index"].clone()),
            ("distances[0]".into(), 2.into())
        );
    }

    #[test]
    fn total_cost_prices_each_route_with_its_vehicle() {
        let vehicles = [
            vehicle(
                serde_json::json!({ "capacity": 1, "fixed_cost": 5, "cost_per_distance": 2, "count": 1 }),
            ),
            vehicle(
                serde_json::json!({ "capacity": 1, "fixed_cost": 1, "cost_per_distance": 3, "count": 1 }),
            ),
        ];
        let customers = [
            customer(serde_json::json!({ "id": 1, "x": 3.0, "y": 4.0, "demand": 1 })),
            customer(serde_json::json!({ "id": 2, "x": -3.0, "y": 4.0, "demand": 1 })),
        ];
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
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
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
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
            let err = build_vehicles(&[vehicle(json)], 1).expect_err("out of range");
            assert_eq!(err.code(), "parameter_out_of_range");
            assert_eq!(err.fields["parameter"], parameter);
        }
    }

    // ---- entry point ----

    #[test]
    fn an_unknown_method_is_rejected_even_with_no_customers() {
        let err = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &[],
                vehicles: &[],
                distances: None,
                durations: None,
            },
            "tabu",
            &InputConfig::default(),
        )
        .expect_err("unknown method");
        assert!(
            err.message.contains("tabu") && err.message.contains("\"alns\""),
            "{err}"
        );
    }

    #[test]
    fn no_customers_is_an_empty_solution() {
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &[],
                vehicles: &[],
                distances: None,
                durations: None,
            },
            "ga",
            &InputConfig::default(),
        )
        .expect("empty");
        assert!(out.routes.is_empty());
        assert_eq!(out.method_used, "ga");
    }

    #[test]
    fn an_inverted_time_window_names_its_customer() {
        let customers = [customer(serde_json::json!({
            "id": 42, "x": 1.0, "y": 1.0, "time_window": [9.0, 3.0]
        }))];
        let err = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: None,
                durations: None,
            },
            "nn",
            &InputConfig::default(),
        )
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
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: None,
                    durations: None,
                },
                method,
                &InputConfig::default(),
            )
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
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: None,
                durations: None,
            },
            "nn",
            &InputConfig::default(),
        )
        .expect("solvable");
        assert_eq!(out.routes, vec![vec![1]]);
    }

    /// 0.3.3 rounded a demand of 2.4 to 2 and solved that problem instead.
    #[test]
    fn a_fractional_demand_is_refused_not_rounded() {
        let customers = [customer(serde_json::json!({
            "id": 7, "x": 1.0, "y": 1.0, "demand": 2.4
        }))];
        let err = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: None,
                durations: None,
            },
            "nn",
            &InputConfig::default(),
        )
        .expect_err("fractional demand");
        assert!(err.message.contains("customer 7: demand is 2.4"), "{err}");
    }

    #[test]
    fn demand_and_capacity_must_be_whole_units_in_range() {
        for bad in [-1.0, 0.5, 3e9] {
            let customers = [customer(serde_json::json!({
                "id": 1, "x": 1.0, "y": 1.0, "demand": bad
            }))];
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &[],
                    distances: None,
                    durations: None,
                },
                "nn",
                &InputConfig::default(),
            )
            .expect_err("demand out of whole units");
            assert!(err.message.contains("customer 1: demand"), "{bad}: {err}");

            // Checked before the empty-problem shortcut, so a fleet is
            // validated even with nobody to serve.
            let vehicles = [vehicle(serde_json::json!({ "capacity": bad }))];
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &[],
                    vehicles: &vehicles,
                    distances: None,
                    durations: None,
                },
                "nn",
                &InputConfig::default(),
            )
            .expect_err("capacity out of whole units");
            assert!(err.message.contains("vehicle 0: capacity"), "{bad}: {err}");
        }

        let customers = [customer(serde_json::json!({
            "id": 1, "x": 1.0, "y": 1.0, "demand": 0.0
        }))];
        let vehicles = [vehicle(serde_json::json!({ "capacity": 10.0 }))];
        assert!(solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None
            },
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
                let out = solve(
                    &ProblemInput {
                        depots: &[(0.0, 0.0)],
                        customers: &customers,
                        vehicles: &vehicles,
                        distances: None,
                        durations: None,
                    },
                    method,
                    &quick(method),
                )
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
        let internal = build_customers(&[(0.0, 0.0)], &customers, i32::MAX)
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
                    let out = solve(
                        &ProblemInput {
                            depots: &[(0.0, 0.0)],
                            customers: &customers,
                            vehicles: &fleet,
                            distances: None,
                            durations: None,
                        },
                        method,
                        &config(method),
                    )
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
        let e = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &fleet,
                distances: None,
                durations: None,
            },
            "nn",
            &quick("nn"),
        )
        .expect_err("0 distance");
        // A limit must be > 0: zero itself is refused, so the bound is open.
        assert_eq!(
            e.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "max_distance", "index": 0,
                    "min": null, "max": null, "exclusive_min": 0.0, "got": 0.0 })
        );
    }

    // ---- a fleet of several kinds ----

    /// Solves `customers` with `method` and the checker's verdict on the plan:
    /// the plan, and the violations it breaks (none when feasible).
    fn solved_and_checked(
        customers: &[InputCustomer],
        vehicles: &[InputVehicle],
        method: &str,
        config: &InputConfig,
    ) -> (VrpOutput, Vec<PlanViolation>) {
        let problem = ProblemInput {
            depots: &[(0.0, 0.0)],
            customers,
            vehicles,
            distances: None,
            durations: None,
        };
        let out = solve(&problem, method, config).expect(method);
        let plan: InputSolution = serde_json::from_value(serde_json::json!({
            "routes": out.routes,
            "route_vehicles": out.route_vehicles,
            "unassigned": out.unassigned,
            "total_distance": out.total_distance,
            "total_cost": out.total_cost,
        }))
        .expect("valid plan");
        let checked = check(&problem, method, config, &plan).expect("checks");
        (out, checked.violations)
    }

    /// Every method plans the same fleet: kinds that differ in capacity and
    /// costs are all used, each route is on a kind that carries it, and the
    /// checker finds nothing wrong with the plan.
    #[test]
    fn every_method_plans_a_fleet_of_several_kinds() {
        let customers = ring(8); // 8 customers of 5: 40 in all
        let kinds = [
            vehicle(serde_json::json!({ "capacity": 10, "fixed_cost": 1 })),
            vehicle(serde_json::json!({ "capacity": 40, "fixed_cost": 50 })),
        ];
        for method in ["nn", "savings", "ga", "alns"] {
            let (out, violations) = solved_and_checked(&customers, &kinds, method, &quick(method));
            assert!(violations.is_empty(), "{method}: {violations:?}");
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
            for (route, summary) in out.route_summaries.iter().enumerate() {
                let capacity = [10, 40][out.route_vehicles[route]];
                assert!(summary.load <= capacity, "{method}: route {route}");
            }
        }
    }

    /// A fleet of several kinds that is nearly full: 38 customers, 16 409 units,
    /// eight vehicles of 18 500 in all -- and the same day with one vehicle of
    /// 1 500 fewer (17 000). Both fit: best-fit decreasing by demand alone packs
    /// them. The routes a method shapes for one kind of vehicle do not divide
    /// among the kinds there are, and a customer used to be left out although
    /// the fleet has the room; a longer search left out more than a short one.
    #[test]
    fn a_nearly_full_fleet_of_several_kinds_leaves_no_customer_that_fits() {
        let stops: [(f64, f64, i32); 38] = [
            (-12.0, 27.0, 195),
            (12.0, 9.0, 131),
            (4.0, -7.0, 237),
            (28.0, 3.0, 273),
            (-12.0, 27.0, 430),
            (-5.0, -19.0, 325),
            (-17.0, 14.0, 959),
            (-5.0, -19.0, 605),
            (-5.0, -19.0, 151),
            (-24.0, 6.0, 225),
            (-5.0, -19.0, 231),
            (21.0, 20.0, 375),
            (21.0, 20.0, 842),
            (4.0, -7.0, 173),
            (-17.0, 14.0, 1130),
            (4.0, -7.0, 735),
            (26.0, -11.0, 163),
            (4.0, -7.0, 132),
            (21.0, 20.0, 122),
            (-5.0, -19.0, 210),
            (15.0, -26.0, 520),
            (-5.0, -19.0, 348),
            (26.0, -11.0, 1256),
            (12.0, 9.0, 745),
            (-8.0, -22.0, 134),
            (-24.0, 6.0, 220),
            (-17.0, 14.0, 98),
            (-5.0, -19.0, 189),
            (26.0, -11.0, 705),
            (-17.0, 14.0, 380),
            (-28.0, -15.0, 1064),
            (-24.0, 6.0, 1560),
            (-12.0, 27.0, 137),
            (28.0, 3.0, 131),
            (12.0, 9.0, 183),
            (-24.0, 6.0, 168),
            (-5.0, -19.0, 602),
            (12.0, 9.0, 325),
        ];
        let customers: Vec<InputCustomer> = stops
            .iter()
            .enumerate()
            .map(|(i, &(x, y, demand))| {
                customer(serde_json::json!({ "id": i + 1, "x": x, "y": y, "demand": demand }))
            })
            .collect();
        let kinds = |small: usize| {
            [
                vehicle(serde_json::json!({ "capacity": 1000, "count": 2 })),
                vehicle(serde_json::json!({ "capacity": 1500, "count": small })),
                vehicle(serde_json::json!({ "capacity": 2500, "count": 2 })),
                vehicle(serde_json::json!({ "capacity": 3500, "count": 1 })),
                vehicle(serde_json::json!({ "capacity": 5000, "count": 1 })),
            ]
        };
        let long = |method: &str| match method {
            "ga" => InputConfig {
                max_generations: Some(200),
                seed: Some(42),
                ..InputConfig::default()
            },
            "alns" => InputConfig {
                max_iterations: Some(3000),
                seed: Some(42),
                ..InputConfig::default()
            },
            _ => InputConfig::default(),
        };
        for small in [2, 1] {
            let fleet = kinds(small);
            for method in ["savings", "ga", "alns"] {
                for config in [quick(method), long(method)] {
                    let (out, violations) = solved_and_checked(&customers, &fleet, method, &config);
                    assert!(violations.is_empty(), "{method} {small}: {violations:?}");
                    assert!(
                        out.unassigned.is_empty(),
                        "{method} with {small} of 1500: left out {:?}",
                        out.unassigned
                    );
                }
            }
        }
    }

    /// One kind of vehicle, three of them, exactly full: demands 6, 6, 4, 4, 5, 5
    /// in vehicles of 10. It fits as (6, 4), (6, 4), (5, 5), but the customers
    /// that are near each other are the two 4s and the two 5s, so a plan made by
    /// distance pairs those, leaves each 6 alone on a fourth route, and bringing
    /// it down to three routes has nowhere to put a 6.
    #[test]
    fn a_full_fleet_of_one_kind_leaves_no_customer_that_fits() {
        let customers: Vec<InputCustomer> = [
            (-10.0, 0.0, 6),
            (-10.0, 1.0, 6),
            (10.0, 0.0, 4),
            (10.0, 1.0, 4),
            (0.0, 10.0, 5),
            (0.0, 11.0, 5),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, y, demand))| {
            customer(serde_json::json!({ "id": i + 1, "x": x, "y": y, "demand": demand }))
        })
        .collect();
        let three = [vehicle(serde_json::json!({ "capacity": 10, "count": 3 }))];
        for method in ["savings", "ga", "alns"] {
            let (out, violations) = solved_and_checked(&customers, &three, method, &quick(method));
            assert!(violations.is_empty(), "{method}: {violations:?}");
            assert!(
                out.unassigned.is_empty(),
                "{method}: left out {:?} of a fleet that takes them all",
                out.unassigned
            );
            assert!(out.routes.len() <= 3, "{method}");
        }
    }

    /// Loading the vehicles first keeps what the route-first plan keeps: the vehicle's duration
    /// limit with its break, and the time windows -- a late arrival at a soft window is priced,
    /// at a hard one it is refused. Demands 6, 6, 4, 4, 5, 5 in three vehicles of 10 fit only
    /// as (6, 4), (6, 4), (5, 5), so every method needs the loaded plan here.
    #[test]
    fn the_loaded_fleet_keeps_breaks_and_windows() {
        let stops = [
            (-10.0, 0.0, 6),
            (-10.0, 1.0, 6),
            (10.0, 0.0, 4),
            (10.0, 1.0, 4),
            (0.0, 10.0, 5),
            (0.0, 11.0, 5),
        ];
        let build = |window: Option<(f64, Option<f64>)>| -> Vec<InputCustomer> {
            stops
                .iter()
                .enumerate()
                .map(|(i, &(x, y, demand))| {
                    let mut c =
                        serde_json::json!({ "id": i + 1, "x": x, "y": y, "demand": demand });
                    if let Some((close, penalty)) = window {
                        c["time_window"] = serde_json::json!([0.0, close]);
                        if let Some(p) = penalty {
                            c["lateness_penalty"] = serde_json::json!(p);
                        }
                    }
                    customer(c)
                })
                .collect()
        };
        let fleet = [vehicle(serde_json::json!({
            "capacity": 10,
            "count": 3,
            "max_duration": 400.0,
            "break": { "earliest": 5.0, "latest": 300.0, "duration": 20.0 }
        }))];
        // No window; a hard window that every route can keep; a soft window some route will miss.
        for customers in [
            build(None),
            build(Some((200.0, None))),
            build(Some((30.0, Some(1.0)))),
        ] {
            for method in ["savings", "ga", "alns"] {
                let (out, violations) =
                    solved_and_checked(&customers, &fleet, method, &quick(method));
                assert!(violations.is_empty(), "{method}: {violations:?}");
                assert!(
                    out.unassigned.is_empty(),
                    "{method}: left out {:?}",
                    out.unassigned
                );
                assert!(out.routes.len() <= 3, "{method}");
            }
        }
    }

    /// A kind without `count` is as many vehicles as the plan needs -- the
    /// fleet a single vehicle of a request has always stood for -- by every
    /// method, `"nn"` included.
    #[test]
    fn a_kind_without_a_count_is_as_many_vehicles_as_the_plan_needs() {
        let customers = ring(8);
        let one_kind = fleet(&[10.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let (out, violations) =
                solved_and_checked(&customers, &one_kind, method, &quick(method));
            assert!(violations.is_empty(), "{method}: {violations:?}");
            assert!(out.unassigned.is_empty(), "{method}");
            assert_eq!(out.routes.len(), 4, "{method}: 40 units in vehicles of 10");
        }
    }

    /// `count` bounds the vehicles of a kind: no method plans more routes for
    /// it, and the customers its vehicles cannot carry are reported.
    #[test]
    fn a_count_bounds_the_vehicles_of_a_kind() {
        let customers = ring(8);
        let two_vans = [vehicle(serde_json::json!({ "capacity": 10, "count": 2 }))];
        for method in ["nn", "savings", "ga", "alns"] {
            let (out, violations) =
                solved_and_checked(&customers, &two_vans, method, &quick(method));
            assert!(violations.is_empty(), "{method}: {violations:?}");
            assert_eq!(out.routes.len(), 2, "{method}");
            assert_eq!(out.unassigned.len(), 4, "{method}: 40 units, 20 carried");
        }
        // Two kinds: the dear one has a single vehicle, so the rest of the work
        // goes to the cheap one -- and the checker agrees on the counts.
        let kinds = [
            vehicle(serde_json::json!({ "capacity": 20, "fixed_cost": 30, "count": 1 })),
            vehicle(serde_json::json!({ "capacity": 10, "fixed_cost": 1 })),
        ];
        for method in ["savings", "ga", "alns"] {
            let (out, violations) = solved_and_checked(&customers, &kinds, method, &quick(method));
            assert!(violations.is_empty(), "{method}: {violations:?}");
            assert!(out.unassigned.is_empty(), "{method}: {:?}", out.unassigned);
            let first = out.route_vehicles.iter().filter(|&&v| v == 0).count();
            assert!(
                first <= 1,
                "{method}: kind 0 has one vehicle, used {first} times"
            );
        }
    }

    /// Among the kinds that carry a route, the cheapest drives it.
    #[test]
    fn the_cheapest_kind_that_carries_a_route_drives_it() {
        let customers = ring(8);
        let kinds = [
            vehicle(serde_json::json!({ "capacity": 40, "fixed_cost": 100 })),
            vehicle(serde_json::json!({ "capacity": 40, "fixed_cost": 0 })),
        ];
        for method in ["savings", "ga", "alns"] {
            let (out, violations) = solved_and_checked(&customers, &kinds, method, &quick(method));
            assert!(violations.is_empty(), "{method}: {violations:?}");
            assert!(
                out.route_vehicles.iter().all(|&v| v == 1),
                "{method}: {:?}",
                out.route_vehicles
            );
        }
    }

    #[test]
    fn a_count_below_one_is_refused() {
        let customers = ring(3);
        let kinds = [
            vehicle(serde_json::json!({ "capacity": 10 })),
            vehicle(serde_json::json!({ "capacity": 10, "count": 0 })),
        ];
        let err = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &kinds,
                distances: None,
                durations: None,
            },
            "nn",
            &quick("nn"),
        )
        .expect_err("count 0");
        assert_eq!(err.code(), "parameter_out_of_range");
        assert_eq!(err.fields["parameter"], "count");
        assert_eq!(err.fields["index"], 1);
        assert_eq!(err.fields["min"], 1.0);
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &one_capacity,
                    distances: None,
                    durations: None,
                },
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &six_of_ten(),
                    vehicles: &two,
                    distances: None,
                    durations: None,
                },
                method,
                &fixed(method),
            )
            .expect(method);
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &truck,
                    distances: None,
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
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
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &fleet(&[10.0]),
                distances: None,
                durations: None,
            },
            "nn",
            &quick("nn"),
        )
        .expect("nn");
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &vehicles,
                    distances: None,
                    durations: None,
                },
                method,
                &config,
            )
            .expect(method);
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &none,
                    distances: None,
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &six_of_ten(),
                    vehicles: &two,
                    distances: None,
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &six_of_ten(),
                    vehicles: &roomy,
                    distances: None,
                    durations: None,
                },
                method,
                &fixed(method),
            )
            .expect(method);
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &one_capacity,
                    distances: None,
                    durations: None,
                },
                method,
                &roomy(method),
            )
            .expect(method);
            let free = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &one_capacity,
                    distances: None,
                    durations: None,
                },
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &fleet(&[100.0, 100.0]),
                    distances: None,
                    durations: None,
                },
                method,
                &fixed(method),
            )
            .expect(method);
            assert!(out.num_vehicles <= 2, "{method}");
            let internal = build_customers(&[(0.0, 0.0)], &customers, i32::MAX)
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(4),
                    vehicles: &fleet(&[10.0]),
                    distances: None,
                    durations: None,
                },
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &two,
                    distances: None,
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &fleet,
                    distances: None,
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &fleet,
                    distances: None,
                    durations: None,
                },
                method,
                &quick(method),
            )
            .expect(method);
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &fraction,
                    vehicles: &[],
                    distances: None,
                    durations: None,
                },
                "nn",
                &quick("nn"),
            )
            .expect_err("fractional demand"),
            solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(8),
                    vehicles: &fleet(&[10.0]),
                    distances: None,
                    durations: None,
                },
                "ga",
                &no_fleet,
            )
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
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &[], vehicles: &[], distances: None, durations: None }, "tabu", &quick("tabu")).expect_err("method"),
                "unknown_option",
                serde_json::json!({ "parameter": "method", "got": "tabu" }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &[
                        customer(serde_json::json!({ "id": 5, "x": 1.0, "y": 0.0 })),
                        customer(serde_json::json!({ "id": 5, "x": 2.0, "y": 0.0 })),
                    ], vehicles: &[], distances: None, durations: None },
                    "nn",
                    &quick("nn"),
                )
                .expect_err("duplicate"),
                "duplicate_id",
                serde_json::json!({ "id": 5, "first": 0, "second": 1 }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &one(serde_json::json!({ "id": 9, "x": 1.0, "y": 0.0, "demand": 0.5 })), vehicles: &[], distances: None, durations: None },
                    "nn",
                    &quick("nn"),
                )
                .expect_err("fraction"),
                "not_whole_units",
                serde_json::json!({ "parameter": "demand", "index": 0, "id": 9, "value": 0.5 }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &[], vehicles: &fleet(&[10.0, 2.5]), distances: None, durations: None }, "nn", &quick("nn"))
                    .expect_err("capacity"),
                "not_whole_units",
                serde_json::json!({ "parameter": "capacity", "index": 1, "id": null }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &one(serde_json::json!({ "id": 9, "x": 1.0, "y": 0.0, "demand": -3.0 })), vehicles: &[], distances: None, durations: None },
                    "nn",
                    &quick("nn"),
                )
                .expect_err("negative"),
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "demand", "index": 0, "id": 9,
                                    "min": 0.0, "max": f64::from(i32::MAX), "got": -3.0 }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &[], vehicles: &fleet(&[10.0, 3e9]), distances: None, durations: None }, "nn", &quick("nn"))
                    .expect_err("too large"),
                "parameter_out_of_range",
                serde_json::json!({ "parameter": "capacity", "index": 1, "id": null,
                                    "min": 0.0, "max": f64::from(i32::MAX), "got": 3e9 }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &one(serde_json::json!({ "id": 3, "x": 1.0, "y": 0.0, "time_window": [4.0, 1.0] })), vehicles: &[], distances: None, durations: None },
                    "nn",
                    &quick("nn"),
                )
                .expect_err("window"),
                "invalid_time_window",
                serde_json::json!({ "id": 3, "index": 0, "ready": 4.0, "due": 1.0 }),
            ),
            (
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &ring(8), vehicles: &fleet(&[10.0]), distances: None, durations: None },
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
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &ring(4), vehicles: &fleet(&[10.0]), distances: None, durations: None },
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
                solve(&ProblemInput { depots: &[(0.0, 0.0)], customers: &ring(4), vehicles: &fleet(&[10.0]), distances: None, durations: None },
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
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(4),
                    vehicles: &fleet(&[10.0]),
                    distances: None,
                    durations: None,
                },
                method,
                &config,
            )
            .expect_err(method);
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(4),
                    vehicles: &fleet(&[10.0]),
                    distances: None,
                    durations: None
                },
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
                    let out = solve(
                        &ProblemInput {
                            depots: &[(0.0, 0.0)],
                            customers,
                            vehicles,
                            distances: None,
                            durations: None,
                        },
                        method,
                        &config,
                    )
                    .unwrap_or_else(|e| panic!("{name} {method}: {e:?}"));
                    let checked = check(
                        &ProblemInput {
                            depots: &[(0.0, 0.0)],
                            customers,
                            vehicles,
                            distances: None,
                            durations: None,
                        },
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

    // ---- vehicle break ----

    /// Every route of every method keeps the driver's break: the route's clock with the break
    /// placed (the evaluator's, built from the numbers) raises nothing, and a route out past the
    /// break's `earliest` does carry one -- through construction, every search move, a fixed fleet's
    /// emptied routes, and the plan check.
    #[test]
    fn every_method_keeps_the_drivers_break() {
        // Hard windows closing at 80: the 30 the break takes is time the stops after it do not
        // have, so a method that plans without the break packs routes the break then overruns.
        let customers: Vec<InputCustomer> = (1..=9)
            .map(|i| {
                let a = i as f64;
                customer(serde_json::json!({
                    "id": i, "x": a.cos() * 10.0, "y": a.sin() * 10.0, "demand": 5.0,
                    "service_time": 3.0, "time_window": [0.0, 80.0]
                }))
            })
            .collect();
        let fleet = vec![vehicle(serde_json::json!({
            "capacity": 1000.0,
            "break": { "earliest": 5.0, "latest": 15.0, "duration": 30.0 }
        }))];
        let internal = build_customers(&[(0.0, 0.0)], &customers, i32::MAX)
            .expect("valid")
            .customers;
        let dm = DistanceMatrix::from_customers(&internal);
        // Built from the numbers, not from the code under test.
        let oracle =
            Vehicle::new(0, 1000).with_break(crate::models::VehicleBreak::new(5.0, 15.0, 30.0));
        let mut routes_with_break = 0;
        for max_vehicles in [None, Some(2)] {
            for method in ["nn", "savings", "ga", "alns"] {
                let config = InputConfig {
                    max_vehicles,
                    ..quick(method)
                };
                let problem = ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles: &fleet,
                    distances: None,
                    durations: None,
                };
                let out =
                    solve(&problem, method, &config).unwrap_or_else(|e| panic!("{method}: {e:?}"));
                let evaluator = crate::evaluation::RouteEvaluator::new(&internal, &dm, &oracle);
                for route in &out.routes {
                    let (timed, violations) = evaluator.build_route(route);
                    assert!(
                        violations.is_empty(),
                        "{method} {max_vehicles:?}: {violations:?} on {route:?}"
                    );
                    routes_with_break += usize::from(timed.rest().is_some());
                }
                let served: usize = out.routes.iter().map(Vec::len).sum();
                assert_eq!(served + out.unassigned.len(), 9, "{method} lost a customer");
            }
        }
        assert!(
            routes_with_break > 0,
            "no route was long enough to need the break"
        );
    }

    /// A plan the break cannot be placed on is a result: `break_not_placeable`, and one that
    /// can place it is clean.
    #[test]
    fn a_plan_whose_route_cannot_place_the_break_reports_it() {
        let customers = vec![customer(json!({ "id": 7, "x": 10.0, "y": 0.0 }))];
        let problem = |earliest: f64, latest: f64| {
            vec![vehicle(json!({
                "capacity": 10.0,
                "break": { "earliest": earliest, "latest": latest, "duration": 1.0 }
            }))]
        };
        let solution: InputSolution = serde_json::from_value(json!({
            "routes": [[7]], "route_vehicles": [0], "unassigned": [], "total_distance": 20.0
        }))
        .expect("a solution");
        let run = |vehicles: &[InputVehicle]| {
            check(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &customers,
                    vehicles,
                    distances: None,
                    durations: None,
                },
                "nn",
                &InputConfig::default(),
                &solution,
            )
            .expect("a check")
        };
        // Out at 10 .. 11 and back at 21: the break due from 2 must start by 5 -- it cannot (arrival 10).
        let out = run(&problem(2.0, 5.0));
        assert!(!out.feasible);
        assert!(
            out.violations
                .iter()
                .any(|v| matches!(v, PlanViolation::BreakNotPlaceable { route: 0 })),
            "{:?}",
            out.violations
        );
        // The window reaches the arrival: the break goes there.
        let out = run(&problem(2.0, 12.0));
        assert!(out.feasible, "{:?}", out.violations);
    }

    /// Kinds that differ in their break are refused: a route is timed once and then given to the
    /// cheapest kind that carries it, which only holds when they keep one clock.
    #[test]
    fn vehicle_kinds_with_different_breaks_are_refused() {
        let b = json!({ "earliest": 5.0, "latest": 40.0, "duration": 20.0 });
        let other = json!({ "earliest": 5.0, "latest": 40.0, "duration": 25.0 });
        for (a, c) in [
            (
                json!({ "capacity": 10.0, "break": b }),
                json!({ "capacity": 20.0 }),
            ),
            (
                json!({ "capacity": 10.0, "break": b }),
                json!({ "capacity": 20.0, "break": other }),
            ),
        ] {
            let vehicles = vec![vehicle(a), vehicle(c)];
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &[customer(json!({ "id": 1, "x": 1.0, "y": 0.0 }))],
                    vehicles: &vehicles,
                    distances: None,
                    durations: None,
                },
                "nn",
                &InputConfig::default(),
            )
            .expect_err("refused");
            assert_eq!(err.code(), "invalid_option", "{err:?}");
            assert_eq!(err.fields["parameter"], "vehicles[].break");
        }
        // The same break on every kind is a fleet.
        let vehicles = vec![
            vehicle(json!({ "capacity": 10.0, "break": b })),
            vehicle(json!({ "capacity": 20.0, "break": b })),
        ];
        solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &[customer(json!({ "id": 1, "x": 1.0, "y": 0.0 }))],
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            "nn",
            &InputConfig::default(),
        )
        .expect("a fleet that shares one break");
    }

    /// A break is refused where it cannot mean anything: before time 0, ending before it starts,
    /// taking no time.
    #[test]
    fn a_break_that_cannot_exist_is_refused_with_its_parameter() {
        for (spec, parameter) in [
            (
                json!({ "earliest": -1.0, "latest": 5.0, "duration": 1.0 }),
                "break.earliest",
            ),
            (
                json!({ "earliest": 5.0, "latest": 4.0, "duration": 1.0 }),
                "break.latest",
            ),
            (
                json!({ "earliest": 1.0, "latest": 4.0, "duration": 0.0 }),
                "break.duration",
            ),
        ] {
            let vehicles = vec![vehicle(json!({ "capacity": 10.0, "break": spec }))];
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &[customer(json!({ "id": 1, "x": 1.0, "y": 0.0 }))],
                    vehicles: &vehicles,
                    distances: None,
                    durations: None,
                },
                "nn",
                &InputConfig::default(),
            )
            .expect_err("refused");
            assert_eq!(err.fields["parameter"], parameter, "{err:?}");
            assert_eq!(err.fields["index"], 0);
        }
    }

    /// A plan that breaks the problem is a result, each rule by name, in the
    /// caller's ids.
    #[test]
    fn a_broken_plan_reports_each_rule_it_breaks() {
        let customers: Vec<InputCustomer> = (1..=4)
            .map(|i| customer(json!({ "id": 10 * i, "x": i as f64, "y": 0.0, "demand": 5.0 })))
            .collect();
        let vehicles = vec![
            vehicle(json!({ "capacity": 10.0, "count": 1 })),
            vehicle(json!({ "capacity": 5.0, "count": 1 })),
        ];
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
        let out = check(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            "nn",
            &config,
            &solution,
        )
        .expect("checked");
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
                PlanViolation::FleetCountExceeded {
                    route: 2,
                    vehicle: 1,
                    count: 1
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

    /// A reported total is judged by its share of the plan's own, so a wrong
    /// figure on a map in small units is as wrong as on a map of order 10.
    #[test]
    fn a_reported_total_is_compared_by_its_share_whatever_the_unit() {
        for scale in [1e-9, 1.0, 1e6] {
            let customers: Vec<InputCustomer> = ring(2)
                .into_iter()
                .enumerate()
                .map(|(i, _)| {
                    let a = (i + 1) as f64;
                    customer(serde_json::json!({
                        "id": i + 1,
                        "x": a.cos() * 10.0 * scale,
                        "y": a.sin() * 10.0 * scale,
                        "demand": 5.0
                    }))
                })
                .collect();
            let vehicles = fleet(&[100.0]);
            let config = InputConfig::default();
            let input = ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            };
            let out = solve(&input, "nn", &config).expect("solved");
            let mut solution = as_solution(&out);
            let right = check(&input, "nn", &config, &solution).expect("checked");
            assert!(right.violations.is_empty(), "a correct plan, scale {scale}");
            solution.total_distance = Some(out.total_distance * 1.001);
            let wrong = check(&input, "nn", &config, &solution).expect("checked");
            assert!(
                wrong
                    .violations
                    .iter()
                    .any(|v| matches!(v, PlanViolation::DistanceMismatch { .. })),
                "a total 0.1 % off is caught, scale {scale}"
            );
        }
    }

    #[test]
    fn a_reported_distance_is_compared_and_route_vehicles_must_match_the_routes() {
        let customers = ring(2);
        let vehicles = fleet(&[100.0]);
        let config = InputConfig::default();
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            "nn",
            &config,
        )
        .expect("solved");
        let mut solution = as_solution(&out);
        solution.total_distance = Some(out.total_distance + 1.0);
        let checked = check(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            "nn",
            &config,
            &solution,
        )
        .expect("checked");
        assert!(matches!(
            checked.violations.as_slice(),
            [PlanViolation::DistanceMismatch { reported, .. }] if *reported == out.total_distance + 1.0
        ));
        solution.route_vehicles = Some(vec![]);
        let e = check(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            "nn",
            &config,
            &solution,
        )
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
            "balance" => json!({ "quantity": "distance", "weight": 1.0 }),
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
                let run = solve(
                    &ProblemInput {
                        depots: &[(0.0, 0.0)],
                        customers: &ring(4),
                        vehicles: &fleet(&[10.0]),
                        distances: None,
                        durations: None,
                    },
                    &info.name,
                    &config,
                );
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
            solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(4),
                    vehicles: &fleet(&[10.0]),
                    distances: None,
                    durations: None,
                },
                method,
                &config,
            )
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
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &ring(4),
                vehicles: &fleet(&[10.0]),
                distances: None,
                durations: None,
            },
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
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(4),
                    vehicles: &fleet(&[10.0]),
                    distances: None,
                    durations: None,
                },
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
                    &ProblemInput {
                        depots: &[(0.0, 0.0)],
                        customers: &ring(4),
                        vehicles: &fleet(&[10.0]),
                        distances: None,
                        durations: None
                    },
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(30),
                    vehicles: &fleet(&[20.0]),
                    distances: None,
                    durations: None,
                },
                method,
                &config,
            )
            .expect("a time-limited run");
            assert_eq!(out.stop_reason, StopReason::TimeLimit, "{method}");
            assert!(out.unassigned.is_empty(), "{method}");

            let zero = InputConfig {
                time_limit_ms: Some(0),
                ..InputConfig::default()
            };
            let err = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &ring(3),
                    vehicles: &fleet(&[20.0]),
                    distances: None,
                    durations: None,
                },
                method,
                &zero,
            )
            .expect_err("0 ms");
            assert_eq!(err.fields["code"], "parameter_out_of_range");
            assert_eq!(err.fields["parameter"], "config.time_limit_ms");
            assert_eq!(err.fields["min"], serde_json::json!(1.0));
        }
        let unread = InputConfig {
            time_limit_ms: Some(100),
            ..InputConfig::default()
        };
        let err = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &ring(3),
                vehicles: &fleet(&[20.0]),
                distances: None,
                durations: None,
            },
            "savings",
            &unread,
        )
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
        let err = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &[],
                distances: None,
                durations: None,
            },
            "nn",
            &quick("nn"),
        )
        .expect_err("negative id");
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let err = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg)
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
            let err = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg)
                .expect_err("out of (0, 1]");
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
        let result = solve_alns(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let result = solve_alns(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
        assert!(result.is_ok(), "GA with default config should succeed");
    }

    #[test]
    fn alns_default_config() {
        let (customers, dm) = test_customers(3);
        let cfg = InputConfig::default();
        let result = solve_alns(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 1000), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 5), 0, &cfg);
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
        let result = solve_ga(&customers, &dm, &Vehicle::new(0, 100), 0, &cfg);
        assert!(result.is_ok(), "GA with 2 customers should succeed");
    }

    // ---- soft time windows ----

    fn far_customer(penalty: Option<f64>) -> Vec<InputCustomer> {
        // 20 away and due at 5: late by 15 whatever is done.
        vec![customer(serde_json::json!({
            "id": 1, "x": 20.0, "y": 0.0, "demand": 1.0, "time_window": [0.0, 5.0],
            "lateness_penalty": penalty,
        }))]
    }

    fn problem<'a>(
        customers: &'a [InputCustomer],
        vehicles: &'a [InputVehicle],
    ) -> ProblemInput<'a> {
        ProblemInput {
            depots: &[(0.0, 0.0)],
            customers,
            vehicles,
            distances: None,
            durations: None,
        }
    }

    #[test]
    fn a_soft_window_is_served_late_and_the_lateness_is_priced_like_distance() {
        let vehicles = fleet(&[10.0]);
        for method in ["nn", "savings", "ga", "alns"] {
            let customers = far_customer(Some(2.0));
            let out = solve(&problem(&customers, &vehicles), method, &quick(method)).expect(method);
            assert!(out.unassigned.is_empty(), "{method} left the customer out");
            let route = &out.route_summaries[0];
            assert_eq!(
                (route.distance, route.lateness, route.lateness_cost),
                (40.0, 15.0, 30.0)
            );
            // A vehicle at 1 per distance: 40 + 30.
            assert_eq!(
                (route.cost, out.total_cost, out.total_distance),
                (70.0, 70.0, 40.0),
                "{method}"
            );
        }
        // The same customer with a hard window cannot be reached in time.
        let customers = far_customer(None);
        let out = solve(
            &problem(&customers, &vehicles),
            "savings",
            &quick("savings"),
        )
        .expect("solves");
        assert_eq!(out.unassigned, vec![1]);
    }

    #[test]
    fn a_lateness_penalty_below_zero_is_refused_at_its_customer() {
        let customers = far_customer(Some(-1.0));
        let vehicles = fleet(&[10.0]);
        let err = solve(
            &problem(&customers, &vehicles),
            "nn",
            &InputConfig::default(),
        )
        .expect_err("a negative penalty is refused");
        assert_eq!(err.code(), "parameter_out_of_range");
        assert_eq!(err.fields["parameter"], "lateness_penalty");
        assert_eq!(
            (err.fields["id"].as_i64(), err.fields["index"].as_i64()),
            (Some(1), Some(0))
        );
        assert_eq!(err.fields["got"].as_f64(), Some(-1.0));
        assert_eq!(err.fields["min"].as_f64(), Some(0.0));
    }

    #[test]
    fn a_plan_is_checked_at_its_priced_cost() {
        let customers = far_customer(Some(2.0));
        let vehicles = fleet(&[10.0]);
        let check_with = |total_cost: f64| {
            let plan: InputSolution = serde_json::from_value(serde_json::json!({
                "routes": [[1]], "total_cost": total_cost,
            }))
            .expect("valid plan");
            check(
                &problem(&customers, &vehicles),
                "nn",
                &InputConfig::default(),
                &plan,
            )
            .expect("checks")
        };
        // 40 of distance and 15 late at 2 a unit.
        assert!(check_with(70.0).violations.is_empty());
        // The distance alone is not what the plan costs.
        assert_eq!(
            check_with(40.0).violations,
            vec![PlanViolation::CostMismatch {
                reported: 40.0,
                actual: 70.0
            }]
        );
    }

    #[test]
    fn a_soft_problem_costs_no_more_than_the_hard_one_for_the_one_pass_methods() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = move |bound: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * bound
        };
        let vehicles = fleet(&[60.0]);
        for _ in 0..12 {
            let raw: Vec<(f64, f64, f64, f64)> = (0..30)
                .map(|_| (next(100.0), next(100.0), next(200.0), 50.0 + next(80.0)))
                .collect();
            let build = |soft: bool| -> Vec<InputCustomer> {
                raw.iter()
                    .enumerate()
                    .map(|(i, &(x, y, ready, width))| {
                        customer(serde_json::json!({
                            "id": i + 1, "x": x, "y": y, "demand": 3.0, "service_time": 5.0,
                            "time_window": [ready, ready + width],
                            "lateness_penalty": soft.then_some(2.0),
                        }))
                    })
                    .collect()
            };
            for method in ["nn", "savings"] {
                let run = |soft: bool| {
                    let customers = build(soft);
                    solve(&problem(&customers, &vehicles), method, &quick(method)).expect(method)
                };
                let (hard, soft) = (run(false), run(true));
                assert!(soft.unassigned.len() <= hard.unassigned.len(), "{method}");
                if soft.unassigned.len() == hard.unassigned.len() {
                    assert!(
                        soft.total_cost <= hard.total_cost + 1e-9,
                        "{method}: soft {} against hard {}",
                        soft.total_cost,
                        hard.total_cost
                    );
                }
            }
        }
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
                    &ProblemInput {
                        depots: &[depot],
                        customers: &input(&list),
                        vehicles: &vehicles(6, capacity),
                        distances: None,
                        durations: None,
                    },
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
                    &ProblemInput {
                        depots: &[depot],
                        customers: &input(&list),
                        vehicles: &vehicles(1, 1e9),
                        distances: None,
                        durations: None,
                    },
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
        let out = solve(
            &ProblemInput {
                depots: &[depot],
                customers: &input(&list),
                vehicles: &vehicles(6, capacity),
                distances: None,
                durations: None,
            },
            "ga",
            &cfg,
        )
        .expect("solved");
        let routes = json!(out.routes);
        let good = plan(
            routes.clone(),
            json!({ "unassigned": out.unassigned, "total_distance": out.total_distance }),
        );
        let check_of = |p: &InputSolution| {
            check(
                &ProblemInput {
                    depots: &[depot],
                    customers: &input(&list),
                    vehicles: &vehicles(6, capacity),
                    distances: None,
                    durations: None,
                },
                "ga",
                &cfg,
                p,
            )
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
            "balance" => json!({ "quantity": "load", "weight": 0.5 }),
            other => panic!("unknown setting {other}"),
        };
        let depot = (25.0, 25.0);
        let list = customers(6, 3);
        for method in methods() {
            for setting in SETTINGS {
                let cfg: InputConfig =
                    serde_json::from_value(json!({ setting: value(setting) })).expect("a config");
                let result = solve(
                    &ProblemInput {
                        depots: &[depot],
                        customers: &input(&list),
                        vehicles: &vehicles(4, 8.0),
                        distances: None,
                        durations: None,
                    },
                    &method.name,
                    &cfg,
                );
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
            let out = solve(
                &ProblemInput {
                    depots: &[(0.0, 0.0)],
                    customers: &input(&list),
                    vehicles: &fleet,
                    distances: None,
                    durations: None,
                },
                &method.name,
                &config(),
            )
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
                let err = solve(
                    &ProblemInput {
                        depots: &[(0.0, 0.0)],
                        customers: &input(&list),
                        vehicles: &fleet,
                        distances: None,
                        durations: None,
                    },
                    &method.name,
                    &config(),
                )
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
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &input(&list),
                vehicles: &fleet,
                distances: None,
                durations: None,
            },
            "nn",
            &config(),
        )
        .expect("solves");
        assert_eq!(out.unassigned.len(), 2);
    }
}

/// `config.balance`: every method answers cost + weight × the busiest route,
/// the output and the checker report it, and without it nothing changes.
#[cfg(test)]
mod balance_tests {
    use super::*;
    use serde_json::json;

    fn customers(count: usize, seed: u64) -> Vec<InputCustomer> {
        let mut state = seed;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % m) as f64
        };
        (0..count)
            .map(|i| {
                serde_json::from_value(json!({
                    "id": i + 1, "x": next(100), "y": next(100), "demand": 1.0 + next(4)
                }))
                .expect("a customer")
            })
            .collect()
    }

    fn vehicles() -> Vec<InputVehicle> {
        vec![serde_json::from_value(json!({ "capacity": 15.0, "count": 12 })).expect("a vehicle")]
    }

    fn config(method: &str, balance: serde_json::Value) -> InputConfig {
        let mut c = match method {
            "ga" => json!({ "seed": 3, "max_generations": 40 }),
            "alns" => json!({ "seed": 3, "max_iterations": 200 }),
            _ => json!({}),
        };
        if !balance.is_null() {
            c["balance"] = balance;
        }
        serde_json::from_value(c).expect("a config")
    }

    fn problem<'a>(list: &'a [InputCustomer], fleet: &'a [InputVehicle]) -> ProblemInput<'a> {
        ProblemInput {
            depots: &[(50.0, 50.0)],
            customers: list,
            vehicles: fleet,
            distances: None,
            durations: None,
        }
    }

    fn plan_of(out: &VrpOutput) -> InputSolution {
        serde_json::from_value(json!({
            "routes": out.routes,
            "route_vehicles": out.route_vehicles,
            "unassigned": out.unassigned,
            "total_distance": out.total_distance,
            "total_cost": out.total_cost,
            "equity": out.equity,
            "objective": out.objective,
        }))
        .expect("a plan")
    }

    fn largest_distance(out: &VrpOutput) -> f64 {
        out.route_summaries
            .iter()
            .map(|r| r.distance)
            .fold(0.0, f64::max)
    }

    #[test]
    fn every_method_trades_cost_for_a_lighter_busiest_route() {
        let fleet = vehicles();
        for seed in 1..=3_u64 {
            let list = customers(30, seed);
            for method in ["nn", "savings", "ga", "alns"] {
                let plain = solve(
                    &problem(&list, &fleet),
                    method,
                    &config(method, json!(null)),
                )
                .expect("solves");
                let plain_e = largest_distance(&plain);
                let weight = 3.0;
                let cfg = config(method, json!({ "quantity": "distance", "weight": weight }));
                let out = solve(&problem(&list, &fleet), method, &cfg).expect("solves");
                let equity = out.equity.expect("balanced");
                let objective = out.objective.expect("balanced");
                assert!(
                    objective <= plain.total_cost + weight * plain_e + 1e-9,
                    "{method} seed {seed}: objective {objective} vs plain {}",
                    plain.total_cost + weight * plain_e
                );
                assert_eq!(out.unassigned, plain.unassigned, "{method} seed {seed}");
                let largest = out
                    .route_summaries
                    .iter()
                    .map(|r| r.workload.expect("balanced"))
                    .fold(0.0, f64::max);
                assert_eq!(largest, equity);
                assert!((objective - (out.total_cost + weight * equity)).abs() < 1e-9);
                for r in &out.route_summaries {
                    assert_eq!(r.workload, Some(r.distance));
                }
                // The checker recomputes what was reported, and the plan keeps
                // every rule.
                let checked =
                    check(&problem(&list, &fleet), method, &cfg, &plan_of(&out)).expect("checks");
                assert!(
                    checked.feasible,
                    "{method} seed {seed}: {:?}",
                    checked.violations
                );
            }
        }
    }

    #[test]
    fn a_strong_weight_lowers_the_busiest_route() {
        let fleet = vehicles();
        let list = customers(30, 7);
        let plain = solve(
            &problem(&list, &fleet),
            "savings",
            &config("savings", json!(null)),
        )
        .expect("solves");
        let out = solve(
            &problem(&list, &fleet),
            "savings",
            &config("savings", json!({ "quantity": "distance", "weight": 20.0 })),
        )
        .expect("solves");
        assert!(
            out.equity.expect("balanced") < largest_distance(&plain),
            "{:?} vs {}",
            out.equity,
            largest_distance(&plain)
        );
    }

    #[test]
    fn without_a_balance_or_with_weight_zero_the_plan_is_the_cost_plan() {
        let fleet = vehicles();
        let list = customers(25, 11);
        for method in ["nn", "savings", "ga", "alns"] {
            let plain = solve(
                &problem(&list, &fleet),
                method,
                &config(method, json!(null)),
            )
            .expect("solves");
            assert_eq!(plain.equity, None);
            assert_eq!(plain.objective, None);
            assert!(plain.route_summaries.iter().all(|r| r.workload.is_none()));
            let zero = solve(
                &problem(&list, &fleet),
                method,
                &config(method, json!({ "quantity": "load", "weight": 0.0 })),
            )
            .expect("solves");
            assert_eq!(zero.routes, plain.routes, "{method}");
            assert_eq!(
                zero.total_cost.to_bits(),
                plain.total_cost.to_bits(),
                "{method}"
            );
            assert_eq!(zero.objective, Some(plain.total_cost), "{method}");
        }
    }

    #[test]
    fn load_is_a_share_of_the_capacity_of_the_route_vehicle() {
        let fleet = vehicles();
        let list = customers(20, 5);
        let out = solve(
            &problem(&list, &fleet),
            "savings",
            &config("savings", json!({ "quantity": "load", "weight": 1.0 })),
        )
        .expect("solves");
        for r in &out.route_summaries {
            assert_eq!(r.workload, Some(r.load as f64 / 15.0));
        }
    }

    #[test]
    fn a_balance_the_solver_cannot_read_is_refused() {
        let fleet = vehicles();
        let list = customers(10, 1);
        let refused = |balance: serde_json::Value| {
            solve(&problem(&list, &fleet), "nn", &config("nn", balance)).expect_err("refused")
        };
        let e = refused(json!({ "quantity": "stops", "weight": 1.0 }));
        assert_eq!(e.code(), "unknown_option");
        assert_eq!(e.fields["parameter"], "config.balance.quantity");
        assert_eq!(
            e.fields["expected"],
            json!(["load", "distance", "duration"])
        );
        let e = refused(json!({ "quantity": "load", "weight": -1.0 }));
        assert_eq!(e.code(), "parameter_out_of_range");
        assert_eq!(e.fields["min"], json!(0.0));
        assert_eq!(e.fields["got"], json!(-1.0));
        // No time in the problem: a duration would be the distance.
        let e = refused(json!({ "quantity": "duration", "weight": 1.0 }));
        assert_eq!(e.code(), "invalid_option");
        // With a service time it is a balance of its own.
        let mut timed = customers(10, 1);
        timed[0] =
            serde_json::from_value(json!({ "id": 1, "x": 3.0, "y": 4.0, "service_time": 5.0 }))
                .expect("a customer");
        let out = solve(
            &problem(&timed, &fleet),
            "nn",
            &config("nn", json!({ "quantity": "duration", "weight": 1.0 })),
        )
        .expect("solves");
        for r in &out.route_summaries {
            assert_eq!(r.workload, Some(r.duration));
        }
        // A reported equity needs the balance it was measured by.
        let plan: InputSolution =
            serde_json::from_value(json!({ "routes": [[1]], "equity": 3.0 })).expect("a plan");
        let e = check(
            &problem(&list, &fleet),
            "nn",
            &config("nn", json!(null)),
            &plan,
        )
        .expect_err("refused");
        assert_eq!(e.code(), "invalid_option");
        assert_eq!(e.fields["parameter"], "solution.equity");
    }

    #[test]
    fn the_checker_reports_a_wrong_equity_and_objective() {
        let fleet = vehicles();
        let list = customers(15, 2);
        let cfg = config("savings", json!({ "quantity": "distance", "weight": 2.0 }));
        let out = solve(&problem(&list, &fleet), "savings", &cfg).expect("solves");
        let mut plan = plan_of(&out);
        plan.equity = out.equity.map(|e| e + 1.0);
        plan.objective = out.objective.map(|o| o - 5.0);
        let checked = check(&problem(&list, &fleet), "savings", &cfg, &plan).expect("checks");
        let codes: Vec<String> = checked
            .violations
            .iter()
            .map(|v| {
                serde_json::to_value(v).expect("serializes")["code"]
                    .as_str()
                    .expect("a code")
                    .to_string()
            })
            .collect();
        assert_eq!(codes, ["equity_mismatch", "objective_mismatch"]);
    }
}

/// `"nn"` drives each route with a vehicle of its own, `count` copies per kind:
/// every route must be reported with the kind of the vehicle that drove it.
#[cfg(test)]
mod nn_fleet_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nn_reports_the_kind_of_each_route_in_a_fleet_with_counts() {
        let customers: Vec<InputCustomer> = (0..40_i64)
            .map(|i| {
                serde_json::from_value(json!({
                    "id": i + 1,
                    "x": ((i * 7) % 60 - 30) as f64,
                    "y": ((i * 13) % 60 - 30) as f64,
                    "demand": (200 + (i * 37) % 401) as f64,
                }))
                .expect("a customer")
            })
            .collect();
        let vehicles: Vec<InputVehicle> = [(2500.0, 3), (3500.0, 2), (5000.0, 1)]
            .iter()
            .map(|&(capacity, count)| {
                serde_json::from_value(json!({ "capacity": capacity, "count": count }))
                    .expect("a vehicle")
            })
            .collect();
        let problem = ProblemInput {
            depots: &[(0.0, 0.0)],
            customers: &customers,
            vehicles: &vehicles,
            distances: None,
            durations: None,
        };
        let config: InputConfig = serde_json::from_value(json!({})).expect("a config");
        let out = solve(&problem, "nn", &config).expect("solves");
        // Each route's load fits the kind it is reported with.
        let capacity = [2500_i64, 3500, 5000];
        for (summary, &kind) in out.route_summaries.iter().zip(&out.route_vehicles) {
            assert!(
                summary.load <= capacity[kind],
                "load {} on kind {kind}",
                summary.load
            );
        }
        // The plan passes its own checker.
        let plan: InputSolution = serde_json::from_value(json!({
            "routes": out.routes,
            "route_vehicles": out.route_vehicles,
            "unassigned": out.unassigned,
            "total_distance": out.total_distance,
            "total_cost": out.total_cost,
        }))
        .expect("a plan");
        let checked = check(&problem, "nn", &config, &plan).expect("checks");
        assert!(
            checked.feasible,
            "{:?} (route_vehicles {:?})",
            checked.violations, out.route_vehicles
        );
    }
}

/// `solve` and `check` with several depots, against what the definitions say: every
/// customer is served once or reported, each route runs out of its vehicle's depot and
/// is measured from it, and a plan the solver makes is a plan the checker accepts.
#[cfg(test)]
mod multi_depot_tests {
    use super::*;
    use serde_json::json;

    /// A tiny seeded source of numbers.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    }

    fn customer(id: usize, x: f64, y: f64, demand: f64) -> InputCustomer {
        serde_json::from_value(json!({ "id": id, "x": x, "y": y, "demand": demand }))
            .expect("a customer")
    }

    fn vehicle(json: serde_json::Value) -> InputVehicle {
        serde_json::from_value(json).expect("a vehicle")
    }

    fn config(method: &str) -> InputConfig {
        serde_json::from_value(match method {
            "ga" => json!({ "seed": 3, "population_size": 12, "max_generations": 8 }),
            "alns" => json!({ "seed": 3, "max_iterations": 40 }),
            _ => json!({}),
        })
        .expect("settings")
    }

    /// Random customers around `depots`, and a fleet of a few kinds at each depot.
    fn problem(
        rng: &mut Lcg,
        depots: usize,
    ) -> (Vec<(f64, f64)>, Vec<InputCustomer>, Vec<InputVehicle>) {
        let places: Vec<(f64, f64)> = (0..depots)
            .map(|_| (rng.below(100) as f64, rng.below(100) as f64))
            .collect();
        let n = 6 + rng.below(10) as usize;
        let customers = (0..n)
            .map(|i| {
                customer(
                    i + 1,
                    rng.below(100) as f64,
                    rng.below(100) as f64,
                    1.0 + rng.below(5) as f64,
                )
            })
            .collect();
        let mut vehicles = Vec::new();
        for d in 0..depots {
            vehicles.push(vehicle(json!({
                "capacity": 8 + rng.below(8), "depot": d, "fixed_cost": rng.below(20),
            })));
        }
        (places, customers, vehicles)
    }

    fn solve_with(
        depots: &[(f64, f64)],
        customers: &[InputCustomer],
        vehicles: &[InputVehicle],
        method: &str,
        settings: &InputConfig,
    ) -> Result<VrpOutput, ServiceError> {
        solve(
            &ProblemInput {
                depots,
                customers,
                vehicles,
                distances: None,
                durations: None,
            },
            method,
            settings,
        )
    }

    /// What the plan says, from the definitions: each route measured from its vehicle's depot,
    /// within the vehicle's capacity, every customer once or unassigned.
    fn assert_plan_is_right(
        depots: &[(f64, f64)],
        customers: &[InputCustomer],
        vehicles: &[InputVehicle],
        out: &VrpOutput,
        what: &str,
    ) {
        let at = |id: usize| {
            let c = customers
                .iter()
                .find(|c| c.id as usize == id)
                .expect("a customer");
            (c.x, c.y)
        };
        let demand = |id: usize| {
            customers
                .iter()
                .find(|c| c.id as usize == id)
                .unwrap()
                .demand
        };
        let mut seen = std::collections::HashSet::new();
        let (mut total, mut cost) = (0.0, 0.0);
        for (r, route) in out.routes.iter().enumerate() {
            let kind = &vehicles[out.route_vehicles[r]];
            let depot = depots[kind.depot.unwrap_or(0) as usize];
            let mut from = depot;
            let mut length = 0.0;
            let mut load = 0.0;
            for &id in route {
                assert!(seen.insert(id), "{what}: customer {id} twice");
                length += dist(from, at(id));
                from = at(id);
                load += demand(id);
            }
            length += dist(from, depot);
            assert!(
                load <= kind.capacity,
                "{what}: route {r} carries {load} > {}",
                kind.capacity
            );
            assert!(
                (out.route_summaries[r].distance - length).abs() < 1e-6 * (1.0 + length),
                "{what}: route {r} reports {} but runs {length}",
                out.route_summaries[r].distance
            );
            total += length;
            cost += kind.fixed_cost + kind.cost_per_distance * length;
        }
        for c in customers {
            let id = c.id as usize;
            assert!(
                seen.contains(&id) != out.unassigned.contains(&id),
                "{what}: customer {id} is served {} and unassigned {}",
                seen.contains(&id),
                out.unassigned.contains(&id)
            );
        }
        assert!(
            (out.total_distance - total).abs() < 1e-6 * (1.0 + total),
            "{what}: total distance"
        );
        assert!(
            (out.total_cost - cost).abs() < 1e-6 * (1.0 + cost),
            "{what}: total cost"
        );
        assert_eq!(out.num_vehicles, out.routes.len());
    }

    #[test]
    fn every_method_makes_a_plan_the_definitions_and_the_checker_accept() {
        let mut rng = Lcg(7);
        let mut routes_from_other_depots = 0;
        for case in 0..24 {
            let depots = 2 + (case % 2);
            let (places, customers, vehicles) = problem(&mut rng, depots);
            for method in ["nn", "savings", "ga", "alns"] {
                let settings = config(method);
                let what = format!("case {case} {method}");
                let out = solve_with(&places, &customers, &vehicles, method, &settings)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_plan_is_right(&places, &customers, &vehicles, &out, &what);
                routes_from_other_depots += out
                    .route_vehicles
                    .iter()
                    .filter(|&&k| vehicles[k].depot.unwrap_or(0) != 0)
                    .count();
                // The checker takes the plan back as feasible.
                let solution: InputSolution = serde_json::from_value(json!({
                    "routes": out.routes, "route_vehicles": out.route_vehicles,
                    "unassigned": out.unassigned, "total_distance": out.total_distance,
                    "total_cost": out.total_cost,
                }))
                .expect("a solution");
                let checked = check(
                    &ProblemInput {
                        depots: &places,
                        customers: &customers,
                        vehicles: &vehicles,
                        distances: None,
                        durations: None,
                    },
                    method,
                    &settings,
                    &solution,
                )
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert!(checked.feasible, "{what}: {:?}", checked.violations);
            }
        }
        assert!(
            routes_from_other_depots > 20,
            "{routes_from_other_depots} routes from a later depot"
        );
    }

    #[test]
    fn each_cluster_is_served_from_its_own_depot() {
        // Two clusters, one by each depot: serving all from depot 0 costs far more.
        let depots = [(0.0, 0.0), (1000.0, 0.0)];
        let mut customers = Vec::new();
        for i in 0..4 {
            customers.push(customer(10 + i, 5.0 + i as f64, 3.0, 2.0));
            customers.push(customer(20 + i, 995.0 - i as f64, 3.0, 2.0));
        }
        let fleet = vec![
            vehicle(json!({ "capacity": 20 })),
            vehicle(json!({ "capacity": 20, "depot": 1 })),
        ];
        let only_first = vec![vehicle(json!({ "capacity": 20 }))];
        for method in ["nn", "savings", "ga", "alns"] {
            let settings = config(method);
            let both = solve_with(&depots, &customers, &fleet, method, &settings).expect(method);
            assert_plan_is_right(&depots, &customers, &fleet, &both, method);
            let alone =
                solve_with(&depots, &customers, &only_first, method, &settings).expect(method);
            assert_plan_is_right(&depots, &customers, &only_first, &alone, method);
            assert!(
                both.total_distance < 100.0,
                "{method}: {} {:?} {:?}",
                both.total_distance,
                both.routes,
                both.route_vehicles
            );
            assert!(
                alone.total_distance > 1900.0,
                "{method}: {}",
                alone.total_distance
            );
            // The customers near depot 1 ride out of it.
            for (route, &kind) in both.routes.iter().zip(&both.route_vehicles) {
                let from_second = fleet[kind].depot == Some(1);
                assert!(
                    route.iter().all(|&id| (id >= 20) == from_second),
                    "{method}: {route:?}"
                );
            }
        }
    }

    /// A fixed fleet at each of two depots, each exactly full: the demands 6, 6, 4, 4, 5, 5 near a
    /// depot fit its three vehicles of 10 only as (6, 4), (6, 4), (5, 5). Each depot is planned as
    /// a problem of its own, so each loads its vehicles first when its routes leave someone out.
    #[test]
    fn a_full_fleet_at_each_depot_leaves_no_customer_that_fits() {
        let depots = [(0.0, 0.0), (1000.0, 0.0)];
        let around = |base_x: f64, first_id: usize| -> Vec<InputCustomer> {
            [
                (-10.0, 0.0, 6.0),
                (-10.0, 1.0, 6.0),
                (10.0, 0.0, 4.0),
                (10.0, 1.0, 4.0),
                (0.0, 10.0, 5.0),
                (0.0, 11.0, 5.0),
            ]
            .iter()
            .enumerate()
            .map(|(i, &(dx, y, demand))| customer(first_id + i, base_x + dx, y, demand))
            .collect()
        };
        let mut customers = around(0.0, 10);
        customers.extend(around(1000.0, 20));
        let fleet = vec![
            vehicle(json!({ "capacity": 10, "count": 3 })),
            vehicle(json!({ "capacity": 10, "count": 3, "depot": 1 })),
        ];
        for method in ["savings", "ga", "alns"] {
            let out =
                solve_with(&depots, &customers, &fleet, method, &config(method)).expect(method);
            assert_plan_is_right(&depots, &customers, &fleet, &out, method);
            assert!(
                out.unassigned.is_empty(),
                "{method}: left out {:?}",
                out.unassigned
            );
            for (route, &kind) in out.routes.iter().zip(&out.route_vehicles) {
                let from_second = fleet[kind].depot == Some(1);
                assert!(
                    route.iter().all(|&id| (id >= 20) == from_second),
                    "{method}: {route:?}"
                );
            }
        }
    }

    #[test]
    fn customers_change_depots_when_that_saves_a_vehicle() {
        // Customer 1 (x = 40) is nearer to depot 0, customer 2 (x = 95) to depot 1; the vehicle
        // at depot 0 costs 50 to use. Alone, customer 1's route is 80 + 50, customer 2's is 10;
        // together from depot 1 they are 120 -- cheaper by 20, so customer 1 changes depots.
        let depots = [(0.0, 0.0), (100.0, 0.0)];
        let customers = vec![customer(1, 40.0, 0.0, 1.0), customer(2, 95.0, 0.0, 1.0)];
        let fleet = vec![
            vehicle(json!({ "capacity": 10, "fixed_cost": 50 })),
            vehicle(json!({ "capacity": 10, "depot": 1 })),
        ];
        for method in ["nn", "savings", "ga", "alns"] {
            let out =
                solve_with(&depots, &customers, &fleet, method, &config(method)).expect(method);
            assert_plan_is_right(&depots, &customers, &fleet, &out, method);
            assert_eq!(out.route_vehicles, [1], "{method}: {out:?}");
            assert!(
                (out.total_cost - 120.0).abs() < 1e-9,
                "{method}: {}",
                out.total_cost
            );
        }
    }

    #[test]
    fn a_route_starts_at_its_own_depot_whichever_it_is() {
        // Both customers are nearest to depot 1 (at 100): from it, 95 is the nearer and 70
        // the farther, so nearest neighbour drives 95 first. (From the wrong start -- the first
        // customer of the depot's own table -- it would begin at 70.)
        let depots = [(0.0, 0.0), (100.0, 0.0)];
        let customers = vec![customer(1, 70.0, 0.0, 1.0), customer(2, 95.0, 0.0, 1.0)];
        let fleet = vec![
            vehicle(json!({ "capacity": 20 })),
            vehicle(json!({ "capacity": 20, "depot": 1 })),
        ];
        let out = solve_with(&depots, &customers, &fleet, "nn", &config("nn")).expect("nn");
        assert_eq!(out.routes, [vec![2, 1]], "{out:?}");
        assert_eq!(out.route_vehicles, [1]);
    }

    #[test]
    fn a_matrix_names_its_rows_depots_first() {
        // Three points on a line: depots at 0 and 100, customers 1 and 2 at 10 and 90, the
        // matrix given row by row for [depot 0, depot 1, customer 1, customer 2].
        let at = [0.0_f64, 100.0, 10.0, 90.0];
        let matrix: Vec<Vec<f64>> = at
            .iter()
            .map(|a| at.iter().map(|b| (a - b).abs()).collect())
            .collect();
        let depots = [(0.0, 0.0), (100.0, 0.0)];
        let customers = vec![customer(1, 10.0, 0.0, 1.0), customer(2, 90.0, 0.0, 1.0)];
        let fleet = vec![
            vehicle(json!({ "capacity": 5 })),
            vehicle(json!({ "capacity": 5, "depot": 1 })),
        ];
        let out = solve(
            &ProblemInput {
                depots: &depots,
                customers: &customers,
                vehicles: &fleet,
                distances: Some(&matrix),
                durations: None,
            },
            "nn",
            &config("nn"),
        )
        .expect("a matrix with a row for each depot and customer");
        assert_eq!(out.routes, [vec![1], vec![2]], "{out:?}");
        assert!((out.total_distance - 40.0).abs() < 1e-9);
        // One row short is refused, naming the depots.
        let short = &matrix[..3];
        let err = solve(
            &ProblemInput {
                depots: &depots,
                customers: &customers,
                vehicles: &fleet,
                distances: Some(short),
                durations: None,
            },
            "nn",
            &config("nn"),
        )
        .expect_err("a row missing");
        assert_eq!(err.code(), Code::DimensionMismatch.name());
        assert!(
            err.message.contains("2 depots and 2 customers"),
            "{}",
            err.message
        );
    }

    #[test]
    fn what_cannot_be_said_with_several_depots_is_refused_by_name() {
        let depots = [(0.0, 0.0), (50.0, 0.0)];
        let customers = vec![customer(1, 5.0, 0.0, 1.0)];
        let fleet = vec![
            vehicle(json!({ "capacity": 5 })),
            vehicle(json!({ "capacity": 5, "depot": 1 })),
        ];
        for (settings, parameter) in [
            (json!({ "max_vehicles": 2 }), "config.max_vehicles"),
            (
                json!({ "balance": { "quantity": "load", "weight": 1.0 } }),
                "config.balance",
            ),
        ] {
            let config: InputConfig = serde_json::from_value(settings).expect("settings");
            let err =
                solve_with(&depots, &customers, &fleet, "savings", &config).expect_err(parameter);
            assert_eq!(err.code(), Code::InvalidOption.name(), "{parameter}");
            assert_eq!(err.fields["parameter"], parameter);
        }
        // A vehicle's depot is one of the depots.
        let off = vec![vehicle(json!({ "capacity": 5, "depot": 2 }))];
        let err = solve_with(&depots, &customers, &off, "nn", &config("nn")).expect_err("depot 2");
        assert_eq!(err.code(), Code::ParameterOutOfRange.name());
        assert_eq!(err.fields["parameter"], "depot");
        assert_eq!(err.fields["index"], 0);
        assert_eq!(err.fields["max"], 1.0);
        // No depot at all.
        let none: [(f64, f64); 0] = [];
        let err = solve_with(&none, &customers, &fleet, "nn", &config("nn")).expect_err("no depot");
        assert_eq!(err.code(), Code::ParameterOutOfRange.name());
        assert_eq!(err.fields["parameter"], "depots");
    }

    #[test]
    fn a_fleet_left_out_is_one_unlimited_kind_at_every_depot_and_counts_bind_each_depot() {
        let depots = [(0.0, 0.0), (1000.0, 0.0)];
        let mut customers = Vec::new();
        for i in 0..3 {
            customers.push(customer(1 + i, 5.0 + i as f64, 0.0, 1.0));
            customers.push(customer(11 + i, 995.0 - i as f64, 0.0, 1.0));
        }
        // No vehicles: every customer rides from the depot nearest it.
        let out =
            solve_with(&depots, &customers, &[], "savings", &config("savings")).expect("no fleet");
        assert!(out.unassigned.is_empty(), "{out:?}");
        assert!(out.total_distance < 40.0, "{}", out.total_distance);
        // One vehicle of capacity 2 at depot 1: its three customers need two routes, only one
        // vehicle is there, so one customer is left out; depot 0's side is served.
        let fleet = vec![
            vehicle(json!({ "capacity": 9 })),
            vehicle(json!({ "capacity": 2, "depot": 1, "count": 1 })),
        ];
        let out = solve_with(&depots, &customers, &fleet, "savings", &config("savings"))
            .expect("counted");
        let from_second = out.route_vehicles.iter().filter(|&&k| k == 1).count();
        assert_eq!(from_second, 1, "{out:?}");
        assert!(out.unassigned.iter().all(|&id| id >= 11), "{out:?}");
    }
}

#[cfg(test)]
mod pair_tests {
    use super::*;
    use serde_json::json;

    /// A tiny seeded source of numbers.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    fn customers(json: serde_json::Value) -> Vec<InputCustomer> {
        serde_json::from_value(json).expect("customers")
    }

    fn fleet(json: serde_json::Value) -> Vec<InputVehicle> {
        serde_json::from_value(json).expect("vehicles")
    }

    fn config(method: &str) -> InputConfig {
        serde_json::from_value(match method {
            "alns" => json!({ "seed": 3, "max_iterations": 30 }),
            "ga" => json!({ "seed": 3, "population_size": 12, "max_generations": 6 }),
            _ => json!({}),
        })
        .expect("settings")
    }

    fn solve_pairs(
        depots: &[(f64, f64)],
        customers: &[InputCustomer],
        vehicles: &[InputVehicle],
        method: &str,
    ) -> Result<VrpOutput, ServiceError> {
        solve(
            &ProblemInput {
                depots,
                customers,
                vehicles,
                distances: None,
                durations: None,
            },
            method,
            &config(method),
        )
    }

    /// Pickups with the id of the delivery they go to, as `(pickup, delivery)`.
    fn pairs_of(customers: &[InputCustomer]) -> Vec<(i128, i128)> {
        customers
            .iter()
            .filter_map(|c| c.delivery.map(|d| (c.id, d)))
            .collect()
    }

    /// What the plan has to be, from the definitions and nothing the solver computed: a pair is
    /// served by one route or by none, the pickup first; the load after each stop (a pickup adds
    /// what it carries, a delivery takes it off) stays within the capacity of the route's vehicle;
    /// the route's reported `load` is the highest of those; every customer is on a route or
    /// unassigned.
    fn assert_pairs_are_kept(
        customers: &[InputCustomer],
        vehicles: &[InputVehicle],
        out: &VrpOutput,
        what: &str,
    ) {
        let demand = |id: i128| {
            customers
                .iter()
                .find(|c| c.id == id)
                .expect("a customer")
                .demand
        };
        let pairs = pairs_of(customers);
        for &(pickup, delivery) in &pairs {
            let at = |id: i128| {
                out.routes.iter().enumerate().find_map(|(r, route)| {
                    route
                        .iter()
                        .position(|&c| c as i128 == id)
                        .map(|stop| (r, stop))
                })
            };
            match (at(pickup), at(delivery)) {
                (Some((rp, sp)), Some((rd, sd))) => {
                    assert_eq!(
                        rp, rd,
                        "{what}: pair {pickup}->{delivery} split over routes"
                    );
                    assert!(sp < sd, "{what}: {delivery} visited before {pickup}");
                }
                (None, None) => {
                    assert!(
                        out.unassigned.contains(&(pickup as usize))
                            && out.unassigned.contains(&(delivery as usize)),
                        "{what}: pair {pickup}->{delivery} dropped without being unassigned"
                    );
                }
                other => panic!("{what}: pair {pickup}->{delivery} half served: {other:?}"),
            }
        }
        for (r, route) in out.routes.iter().enumerate() {
            let capacity = vehicles
                .get(out.route_vehicles[r])
                .map_or(f64::from(i32::MAX), |v| v.capacity);
            let (mut load, mut peak) = (0.0, 0.0_f64);
            for &c in route {
                let id = c as i128;
                load += match pairs.iter().find(|p| p.1 == id) {
                    Some(&(pickup, _)) => -demand(pickup),
                    None => demand(id),
                };
                peak = peak.max(load);
            }
            assert!(
                peak <= capacity,
                "{what}: route {r} peaks at {peak} > {capacity}"
            );
            assert_eq!(
                out.route_summaries[r].load as f64, peak,
                "{what}: route {r}: the reported load is the peak"
            );
        }
        let served: std::collections::HashSet<usize> =
            out.routes.iter().flatten().copied().collect();
        for c in customers {
            assert!(
                served.contains(&(c.id as usize)) != out.unassigned.contains(&(c.id as usize)),
                "{what}: customer {} is on a route xor unassigned",
                c.id
            );
        }
    }

    /// Four pairs on a line: with a capacity of 2, two can be open at once but not three.
    fn four_pairs() -> Vec<InputCustomer> {
        customers(json!([
            { "id": 1, "x": 10.0, "y": 0.0, "demand": 1, "delivery": 11 },
            { "id": 2, "x": 20.0, "y": 0.0, "demand": 1, "delivery": 12 },
            { "id": 3, "x": 30.0, "y": 0.0, "demand": 1, "delivery": 13 },
            { "id": 4, "x": 40.0, "y": 0.0, "demand": 1, "delivery": 14 },
            { "id": 11, "x": 15.0, "y": 5.0 },
            { "id": 12, "x": 25.0, "y": 5.0 },
            { "id": 13, "x": 35.0, "y": 5.0 },
            { "id": 14, "x": 45.0, "y": 5.0 },
        ]))
    }

    #[test]
    fn every_method_but_ga_plans_pairs_whole_and_in_order() {
        let (depots, customers) = ([(0.0, 0.0)], four_pairs());
        let vehicles = fleet(json!([{ "capacity": 2 }]));
        for method in ["nn", "savings", "alns"] {
            let out = solve_pairs(&depots, &customers, &vehicles, method).expect(method);
            assert!(out.unassigned.is_empty(), "{method}: {out:?}");
            assert_pairs_are_kept(&customers, &vehicles, &out, method);
        }
    }

    #[test]
    fn pairs_planned_over_random_requests_stay_whole_ordered_and_within_capacity() {
        let mut rng = Lcg(11);
        for case in 0..30 {
            let pairs = 2 + rng.below(5) as usize;
            let mut list = Vec::new();
            for k in 0..pairs {
                let (px, py) = (rng.below(100), rng.below(100));
                let q = 1 + rng.below(4);
                list.push(
                    json!({ "id": 100 + k, "x": px, "y": py, "demand": q, "delivery": 200 + k }),
                );
                list.push(json!({ "id": 200 + k, "x": rng.below(100), "y": rng.below(100) }));
            }
            for k in 0..(rng.below(4) as usize) {
                list.push(json!({
                    "id": 300 + k, "x": rng.below(100), "y": rng.below(100),
                    "demand": 1 + rng.below(3),
                }));
            }
            let customers = customers(serde_json::Value::Array(list));
            let depots = [(rng.below(100) as f64, rng.below(100) as f64)];
            let vehicles = fleet(json!([{ "capacity": 4 + rng.below(6) }]));
            for method in ["nn", "savings", "alns"] {
                let out = solve_pairs(&depots, &customers, &vehicles, method).expect(method);
                assert_pairs_are_kept(
                    &customers,
                    &vehicles,
                    &out,
                    &format!("case {case} {method}"),
                );
            }
        }
    }

    /// Moving every pair to its cheapest place is part of the `"savings"` method: on requests
    /// where the merged routes leave pairs that fit better in other routes, the plan is shorter than
    /// the merging alone makes it, and never longer.
    #[test]
    fn savings_moves_pairs_to_their_cheapest_place() {
        use crate::constructive::clarke_wright_savings;
        let mut rng = Lcg(5);
        let mut shorter = 0;
        for case in 0..40 {
            let mut list = Vec::new();
            for k in 0..(6 + rng.below(5) as usize) {
                list.push(json!({
                    "id": 100 + k, "x": rng.below(100), "y": rng.below(100),
                    "demand": 1 + rng.below(3), "delivery": 200 + k,
                }));
                list.push(json!({ "id": 200 + k, "x": rng.below(100), "y": rng.below(100) }));
            }
            let customers = customers(serde_json::Value::Array(list));
            let (capacity, depots) = (6 + rng.below(4), [(50.0, 50.0)]);
            let vehicles = fleet(json!([{ "capacity": capacity }]));
            let out = solve_pairs(&depots, &customers, &vehicles, "savings").expect("savings");
            let built = build_customers(&depots, &customers, i32::MAX).expect("customers");
            let dm = DistanceMatrix::from_customers(&built.customers);
            let vehicle = Vehicle::new(0, capacity as i32);
            let merged = clarke_wright_savings(&built.customers, &dm, &vehicle).total_distance();
            assert!(
                out.total_distance <= merged + 1e-9,
                "case {case}: {} > {merged}",
                out.total_distance
            );
            shorter += usize::from(out.total_distance < merged - 1e-9);
        }
        assert!(
            shorter >= 10,
            "only {shorter} of 40 plans are shorter than the merging alone"
        );
    }

    #[test]
    fn pairs_are_kept_whole_with_several_depots() {
        let depots = [(0.0, 0.0), (100.0, 0.0)];
        let customers = customers(json!([
            { "id": 1, "x": 5.0, "y": 0.0, "demand": 2, "delivery": 2 },
            { "id": 2, "x": 9.0, "y": 4.0 },
            { "id": 3, "x": 95.0, "y": 0.0, "demand": 2, "delivery": 4 },
            { "id": 4, "x": 91.0, "y": 4.0 },
            { "id": 5, "x": 50.0, "y": 0.0, "demand": 1, "delivery": 6 },
            { "id": 6, "x": 52.0, "y": 3.0 },
        ]));
        let vehicles = fleet(json!([
            { "capacity": 4, "depot": 0 },
            { "capacity": 4, "depot": 1 },
        ]));
        for method in ["nn", "savings", "alns"] {
            let out = solve_pairs(&depots, &customers, &vehicles, method).expect(method);
            assert!(out.unassigned.is_empty(), "{method}: {out:?}");
            assert_pairs_are_kept(&customers, &vehicles, &out, method);
        }
    }

    #[test]
    fn a_pair_whose_pickup_no_vehicle_can_carry_is_left_out_whole() {
        let customers = customers(json!([
            { "id": 1, "x": 10.0, "y": 0.0, "demand": 9, "delivery": 2 },
            { "id": 2, "x": 12.0, "y": 0.0 },
            { "id": 3, "x": 20.0, "y": 0.0, "demand": 2, "delivery": 4 },
            { "id": 4, "x": 22.0, "y": 0.0 },
        ]));
        let vehicles = fleet(json!([{ "capacity": 3 }]));
        for method in ["nn", "savings", "alns"] {
            let out = solve_pairs(&[(0.0, 0.0)], &customers, &vehicles, method).expect(method);
            let mut left_out = out.unassigned.clone();
            left_out.sort_unstable();
            assert_eq!(left_out, [1, 2], "{method}: {out:?}");
            assert_pairs_are_kept(&customers, &vehicles, &out, method);
        }
    }

    #[test]
    fn a_pair_that_the_fixed_fleet_cannot_both_serve_is_unassigned_as_a_unit() {
        // One vehicle for two pairs that cannot share a route: each is 24 long alone, both 64.
        let customers = customers(json!([
            { "id": 1, "x": 10.0, "y": 0.0, "demand": 2, "delivery": 2 },
            { "id": 2, "x": 12.0, "y": 0.0 },
            { "id": 3, "x": 30.0, "y": 0.0, "demand": 2, "delivery": 4 },
            { "id": 4, "x": 32.0, "y": 0.0 },
        ]));
        let vehicles = fleet(json!([{ "capacity": 2, "max_distance": 30 }]));
        let config: InputConfig = serde_json::from_value(json!({ "max_vehicles": 1 })).unwrap();
        let out = solve(
            &ProblemInput {
                depots: &[(0.0, 0.0)],
                customers: &customers,
                vehicles: &vehicles,
                distances: None,
                durations: None,
            },
            "savings",
            &config,
        )
        .expect("a fixed fleet");
        assert_pairs_are_kept(&customers, &vehicles, &out, "max_vehicles 1");
        assert_eq!(out.unassigned.len(), 2, "{out:?}");
    }

    fn refusal(
        customers: &[InputCustomer],
        vehicles: &[InputVehicle],
        method: &str,
    ) -> ServiceError {
        solve_pairs(&[(0.0, 0.0)], customers, vehicles, method).expect_err("a refusal")
    }

    #[test]
    fn a_delivery_that_cannot_be_one_is_refused_by_where_it_is_given() {
        let vehicles = fleet(json!([{ "capacity": 5 }]));
        // Names no customer.
        let err = refusal(
            &customers(json!([{ "id": 1, "x": 1.0, "y": 0.0, "demand": 1, "delivery": 9 }])),
            &vehicles,
            "nn",
        );
        assert_eq!(err.code(), Code::InvalidOption.name());
        assert_eq!(err.fields["parameter"], "customers[0].delivery");
        // Is itself.
        let err = refusal(
            &customers(json!([{ "id": 1, "x": 1.0, "y": 0.0, "demand": 1, "delivery": 1 }])),
            &vehicles,
            "nn",
        );
        assert_eq!(err.code(), Code::InvalidOption.name());
        assert_eq!(err.fields["parameter"], "customers[0].delivery");
        // Asks for something itself.
        let err = refusal(
            &customers(json!([
                { "id": 1, "x": 1.0, "y": 0.0, "demand": 1, "delivery": 2 },
                { "id": 2, "x": 2.0, "y": 0.0, "demand": 3 },
            ])),
            &vehicles,
            "nn",
        );
        assert_eq!(err.code(), Code::InvalidOption.name());
        assert_eq!(err.fields["parameter"], "customers[1].demand");
        // Is a pickup too.
        let err = refusal(
            &customers(json!([
                { "id": 1, "x": 1.0, "y": 0.0, "demand": 1, "delivery": 2 },
                { "id": 2, "x": 2.0, "y": 0.0, "delivery": 3 },
                { "id": 3, "x": 3.0, "y": 0.0 },
            ])),
            &vehicles,
            "nn",
        );
        assert_eq!(err.code(), Code::InvalidOption.name());
        assert_eq!(err.fields["parameter"], "customers[1].delivery");
        // Is named by two pickups.
        let err = refusal(
            &customers(json!([
                { "id": 1, "x": 1.0, "y": 0.0, "demand": 1, "delivery": 3 },
                { "id": 2, "x": 2.0, "y": 0.0, "demand": 1, "delivery": 3 },
                { "id": 3, "x": 3.0, "y": 0.0 },
            ])),
            &vehicles,
            "nn",
        );
        assert_eq!(err.code(), Code::DuplicateId.name());
        assert_eq!(
            (
                &err.fields["id"],
                &err.fields["first"],
                &err.fields["second"]
            ),
            (&json!(3), &json!(0), &json!(1))
        );
        // A negative id is a range refusal, as everywhere else.
        let err = refusal(
            &customers(json!([{ "id": 1, "x": 1.0, "y": 0.0, "demand": 1, "delivery": -1 }])),
            &vehicles,
            "nn",
        );
        assert_eq!(err.code(), Code::ParameterOutOfRange.name());
        assert_eq!(err.fields["parameter"], "customers[0].delivery");
    }

    #[test]
    fn what_the_planner_cannot_do_with_pairs_is_refused_not_planned_wrongly() {
        let pairs = four_pairs();
        // The genetic algorithm.
        let err = refusal(&pairs, &fleet(json!([{ "capacity": 2 }])), "ga");
        assert_eq!(err.code(), Code::InvalidOption.name());
        assert_eq!(err.fields["parameter"], "method");
        // Several kinds at one depot.
        let err = refusal(
            &pairs,
            &fleet(json!([{ "capacity": 2 }, { "capacity": 5, "fixed_cost": 3 }])),
            "savings",
        );
        assert_eq!(err.code(), Code::InvalidOption.name());
        assert_eq!(err.fields["parameter"], "vehicles");
        // Without pairs the same requests are fine.
        let plain = customers(json!([
            { "id": 1, "x": 10.0, "y": 0.0, "demand": 1 },
            { "id": 2, "x": 20.0, "y": 0.0, "demand": 1 },
        ]));
        for (method, vehicles) in [
            ("ga", fleet(json!([{ "capacity": 2 }]))),
            (
                "savings",
                fleet(json!([{ "capacity": 2 }, { "capacity": 5 }])),
            ),
        ] {
            solve_pairs(&[(0.0, 0.0)], &plain, &vehicles, method).expect("no pairs");
        }
    }

    #[test]
    fn the_checker_names_a_pair_that_is_split_or_out_of_order() {
        let customers = four_pairs();
        let vehicles = fleet(json!([{ "capacity": 4 }]));
        let problem = ProblemInput {
            depots: &[(0.0, 0.0)],
            customers: &customers,
            vehicles: &vehicles,
            distances: None,
            durations: None,
        };
        let plan = |routes: serde_json::Value| -> InputSolution {
            serde_json::from_value(json!({ "routes": routes })).unwrap()
        };
        let checked = |solution: &InputSolution| {
            check(&problem, "savings", &config("savings"), solution).expect("a check")
        };
        // Everything whole and in order.
        let good = plan(json!([[1, 11, 2, 12], [3, 13, 4, 14]]));
        assert!(checked(&good).feasible);
        // A delivery before its pickup.
        let out = checked(&plan(json!([[11, 1, 2, 12], [3, 13, 4, 14]])));
        assert!(
            out.violations.iter().any(|v| matches!(
                v,
                PlanViolation::PairOrder {
                    route: 0,
                    pickup: 1,
                    delivery: 11
                }
            )),
            "{:?}",
            out.violations
        );
        // A pair on two routes.
        let out = checked(&plan(json!([[1, 2, 12], [11, 3, 13, 4, 14]])));
        assert!(
            out.violations.iter().any(|v| matches!(
                v,
                PlanViolation::PairSplit {
                    pickup: 1,
                    delivery: 11
                }
            )),
            "{:?}",
            out.violations
        );
    }
}
