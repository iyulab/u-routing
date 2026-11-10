//! Route feasibility checking and cost evaluation.

mod check;
mod evaluator;

pub use check::{check_solution, Fleet};
pub use evaluator::RouteEvaluator;

use crate::distance::DistanceMatrix;
use crate::models::{Customer, Vehicle};

/// Whether any customer carries a time window.
pub fn has_time_windows(customers: &[Customer]) -> bool {
    customers.iter().any(|c| c.time_window().is_some())
}

/// Whether any customer has a time window whose end is soft
/// ([`Customer::with_lateness_penalty`]) -- the case where a route's lateness is a cost the
/// searches have to weigh, not a limit they check.
pub fn has_soft_windows(customers: &[Customer]) -> bool {
    customers
        .iter()
        .any(|c| c.time_window().is_some() && c.lateness_penalty().is_some())
}

/// What a vehicle allows one route besides its capacity: how far it may
/// travel and how long it may take, depot to depot -- the limits
/// [`RouteEvaluator::build_route`] reports as violations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RouteLimits {
    /// Longest total distance, including the return to the depot.
    pub max_distance: Option<f64>,
    /// Latest return to the depot, on the clock that starts at 0.
    pub max_duration: Option<f64>,
}

impl RouteLimits {
    /// No limit: every route is allowed whatever its length.
    pub const NONE: RouteLimits = RouteLimits {
        max_distance: None,
        max_duration: None,
    };

    /// The limits `vehicle` sets.
    pub fn of(vehicle: &Vehicle) -> Self {
        RouteLimits {
            max_distance: vehicle.max_distance(),
            max_duration: vehicle.max_duration(),
        }
    }

    /// Whether neither limit is set.
    pub fn is_unlimited(&self) -> bool {
        self.max_distance.is_none() && self.max_duration.is_none()
    }

    /// Whether a route that travels `distance` in all and is back at the depot
    /// at `duration` keeps both limits.
    pub fn allow(&self, distance: f64, duration: f64) -> bool {
        self.max_distance.is_none_or(|max| distance <= max)
            && self.max_duration.is_none_or(|max| duration <= max)
    }
}

/// Whether a search has to ask [`route_feasible`] about its candidates at all:
/// only when a customer has a time window or the vehicle sets a limit. Without
/// either, every sequence is feasible apart from capacity.
pub fn route_checks_needed(customers: &[Customer], limits: &RouteLimits) -> bool {
    has_time_windows(customers) || !limits.is_unlimited()
}

/// Whether a route `depot → route[0] → … → depot` reaches every customer by
/// its due time -- or, at a customer whose window is soft
/// ([`Customer::with_lateness_penalty`]), reaches it at all -- and keeps
/// `limits`, with the same clock as
/// [`RouteEvaluator::build_route`]: departure at 0, travel times from
/// [`DistanceMatrix::travel_time`] (the distances unless it has times of its
/// own), waiting for a window that has not opened, and service before
/// moving on. Capacity is not checked here. O(n) and allocation-free, so a
/// search can ask it about every candidate move.
pub fn route_feasible(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
) -> bool {
    let mut time = 0.0;
    let mut distance = 0.0;
    let mut prev = depot;
    for &cid in route {
        distance += distances.get(prev, cid);
        let arrival = time + distances.travel_time(prev, cid);
        let customer = &customers[cid];
        let start = match customer.time_window() {
            Some(tw) => {
                if tw.is_violated(arrival) && customer.lateness_penalty().is_none() {
                    return false;
                }
                arrival + tw.waiting_time(arrival)
            }
            None => arrival,
        };
        time = start + customer.service_duration();
        prev = cid;
    }
    limits.allow(
        distance + distances.get(prev, depot),
        time + distances.travel_time(prev, depot),
    )
}

/// What [`route_eval`] reports of a route that is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RouteEval {
    /// Total distance, depot to depot.
    pub distance: f64,
    /// Return to the depot, on the clock that starts at 0.
    pub duration: f64,
    /// Time the vehicle is late, summed over the customers whose time window
    /// is soft ([`Customer::with_lateness_penalty`]); a late arrival at a hard
    /// window is not allowed at all.
    pub lateness: f64,
    /// That lateness times each customer's penalty, summed.
    pub lateness_cost: f64,
}

/// The distance, duration and lateness of a route `depot → route[0] → … →
/// depot`, or `None` if it is not allowed: a late arrival at a hard window, or a
/// limit of `limits` broken. The clock and the allowed set are those of
/// [`route_feasible`] -- `route_eval(…).is_some()` is `route_feasible(…)` -- and
/// the totals are those of [`RouteEvaluator::build_route`]. Capacity is not
/// checked here. O(n) and allocation-free.
pub fn route_eval(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
) -> Option<RouteEval> {
    let mut time = 0.0;
    let mut distance = 0.0;
    let mut lateness = 0.0;
    let mut lateness_cost = 0.0;
    let mut prev = depot;
    for &cid in route {
        distance += distances.get(prev, cid);
        let arrival = time + distances.travel_time(prev, cid);
        let customer = &customers[cid];
        let start = match customer.time_window() {
            Some(tw) => {
                if tw.is_violated(arrival) {
                    let penalty = customer.lateness_penalty()?;
                    let late = tw.lateness(arrival);
                    lateness += late;
                    lateness_cost += penalty * late;
                }
                arrival + tw.waiting_time(arrival)
            }
            None => arrival,
        };
        time = start + customer.service_duration();
        prev = cid;
    }
    let distance = distance + distances.get(prev, depot);
    let duration = time + distances.travel_time(prev, depot);
    limits.allow(distance, duration).then_some(RouteEval {
        distance,
        duration,
        lateness,
        lateness_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    fn line() -> (Vec<Customer>, DistanceMatrix) {
        // depot at 0, customers at 3 and 6 on a line, 1 unit of service each.
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 0.0, 1, 1.0),
            Customer::new(2, 6.0, 0.0, 1, 1.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn limits_bound_the_whole_round_trip() {
        let (customers, dm) = line();
        // 0 → 1 → 2 → 0 travels 12 and returns at 14 (two services of 1).
        let route = [1, 2];
        assert!(route_feasible(
            &route,
            0,
            &dm,
            &customers,
            &RouteLimits::NONE
        ));
        let d = |max| RouteLimits {
            max_distance: Some(max),
            ..RouteLimits::NONE
        };
        let t = |max| RouteLimits {
            max_duration: Some(max),
            ..RouteLimits::NONE
        };
        assert!(route_feasible(&route, 0, &dm, &customers, &d(12.0)));
        assert!(!route_feasible(&route, 0, &dm, &customers, &d(11.9)));
        assert!(route_feasible(&route, 0, &dm, &customers, &t(14.0)));
        assert!(!route_feasible(&route, 0, &dm, &customers, &t(13.9)));
        // The same as the evaluator says, at the same boundaries.
        for max in [11.9, 12.0] {
            let vehicle = Vehicle::new(0, 100).with_max_distance(max);
            let (_, violations) =
                RouteEvaluator::new(&customers, &dm, &vehicle).build_route(&route);
            assert_eq!(
                violations.is_empty(),
                route_feasible(&route, 0, &dm, &customers, &RouteLimits::of(&vehicle)),
                "max_distance {max}"
            );
        }
    }

    #[test]
    fn checks_are_needed_only_for_windows_or_limits() {
        let (mut customers, _) = line();
        assert!(!route_checks_needed(&customers, &RouteLimits::NONE));
        let limited = RouteLimits {
            max_duration: Some(5.0),
            ..RouteLimits::NONE
        };
        assert!(route_checks_needed(&customers, &limited));
        customers[1] = customers[1]
            .clone()
            .with_time_window(TimeWindow::new(0.0, 1.0).expect("valid"));
        assert!(route_checks_needed(&customers, &RouteLimits::NONE));
    }

    /// A route's figures by the plainest clock there is: a simulation that
    /// carries the time and the lateness along, written without sharing a line
    /// of the evaluators it checks.
    fn simulate(
        route: &[usize],
        customers: &[Customer],
        dm: &DistanceMatrix,
    ) -> (f64, f64, f64, f64, bool) {
        let (mut t, mut d, mut late, mut cost, mut hard_late) = (0.0, 0.0, 0.0, 0.0, false);
        let mut at = 0;
        for &c in route {
            d += dm.get(at, c);
            t += dm.travel_time(at, c);
            if let Some(tw) = customers[c].time_window() {
                if t < tw.ready() {
                    t = tw.ready();
                }
                if t > tw.due() {
                    match customers[c].lateness_penalty() {
                        Some(p) => {
                            late += t - tw.due();
                            cost += p * (t - tw.due());
                        }
                        None => hard_late = true,
                    }
                }
            }
            t += customers[c].service_duration();
            at = c;
        }
        d += dm.get(at, 0);
        t += dm.travel_time(at, 0);
        (d, t, late, cost, hard_late)
    }

    fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut all = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let head = rest.remove(i);
            for mut tail in permutations(&rest) {
                tail.insert(0, head);
                all.push(tail);
            }
        }
        all
    }

    #[test]
    fn a_soft_window_lets_a_late_arrival_through_and_charges_for_it() {
        let (mut customers, dm) = line();
        // 0 → 1 → 2: arrives at 3, serves to 4, arrives at 2 at 7.
        let window = TimeWindow::new(0.0, 5.0).expect("valid");
        customers[2] = customers[2].clone().with_time_window(window);
        let route = [1, 2];
        assert!(!route_feasible(
            &route,
            0,
            &dm,
            &customers,
            &RouteLimits::NONE
        ));
        assert_eq!(
            route_eval(&route, 0, &dm, &customers, &RouteLimits::NONE),
            None
        );

        customers[2] = customers[2]
            .clone()
            .with_lateness_penalty(4.0)
            .expect("valid");
        assert!(route_feasible(
            &route,
            0,
            &dm,
            &customers,
            &RouteLimits::NONE
        ));
        let eval = route_eval(&route, 0, &dm, &customers, &RouteLimits::NONE).expect("soft");
        assert_eq!(eval.lateness, 2.0);
        assert_eq!(eval.lateness_cost, 8.0);
        assert_eq!(eval.distance, 12.0);
        assert_eq!(eval.duration, 14.0);
    }

    #[test]
    fn a_penalty_without_a_window_is_inert() {
        let (mut customers, dm) = line();
        customers[2] = customers[2]
            .clone()
            .with_lateness_penalty(100.0)
            .expect("valid");
        let eval = route_eval(&[1, 2], 0, &dm, &customers, &RouteLimits::NONE).expect("allowed");
        assert_eq!((eval.lateness, eval.lateness_cost), (0.0, 0.0));
        assert!(!has_time_windows(&customers));
    }

    #[test]
    fn the_evaluators_agree_with_a_plain_simulation_on_every_order() {
        // Mixed hard and soft windows, early and late, on a small instance per seed.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bound: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * bound
        };
        let (mut checked, mut soft_late) = (0, 0);
        for _ in 0..200 {
            let mut customers = vec![Customer::depot(0.0, 0.0)];
            for id in 1..=4 {
                let mut c = Customer::new(id, next(20.0), next(20.0), 1, next(3.0));
                if next(1.0) < 0.8 {
                    let ready = next(25.0);
                    let tw = TimeWindow::new(ready, ready + next(15.0)).expect("valid");
                    c = c.with_time_window(tw);
                    if next(1.0) < 0.6 {
                        c = c.with_lateness_penalty(next(5.0)).expect("valid");
                    }
                }
                customers.push(c);
            }
            let dm = DistanceMatrix::from_customers(&customers);
            let vehicle = Vehicle::new(0, 100);
            let limits = RouteLimits {
                max_duration: Some(60.0),
                ..RouteLimits::NONE
            };
            for route in permutations(&[1, 2, 3, 4]) {
                let (d, t, late, cost, hard_late) = simulate(&route, &customers, &dm);
                let allowed = !hard_late && t <= 60.0;
                let eval = route_eval(&route, 0, &dm, &customers, &limits);
                assert_eq!(eval.is_some(), allowed, "route {route:?}");
                assert_eq!(route_feasible(&route, 0, &dm, &customers, &limits), allowed);
                if let Some(e) = eval {
                    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
                    assert!(
                        close(e.distance, d) && close(e.duration, t),
                        "route {route:?}"
                    );
                    assert!(close(e.lateness, late) && close(e.lateness_cost, cost));
                    // The route the evaluator builds says the same, and its violations are the
                    // hard ones only.
                    let (built, violations) =
                        RouteEvaluator::new(&customers, &dm, &vehicle).build_route(&route);
                    assert!(close(built.total_distance(), d));
                    assert!(close(built.total_lateness(), late));
                    assert!(close(built.lateness_cost(), cost));
                    assert!(violations.iter().all(|v| !matches!(
                        v.kind,
                        crate::models::ViolationType::TimeWindowViolated { .. }
                    )));
                    checked += 1;
                    soft_late += usize::from(e.lateness > 0.0);
                }
            }
        }
        assert!(
            checked > 300,
            "too few allowed routes to mean anything: {checked}"
        );
        assert!(soft_late > 100, "too few late soft routes: {soft_late}");
    }
}
