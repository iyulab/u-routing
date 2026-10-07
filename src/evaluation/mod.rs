//! Route feasibility checking and cost evaluation.

mod evaluator;

pub use evaluator::RouteEvaluator;

use crate::distance::DistanceMatrix;
use crate::models::{Customer, Vehicle};

/// Whether any customer carries a time window.
pub fn has_time_windows(customers: &[Customer]) -> bool {
    customers.iter().any(|c| c.time_window().is_some())
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
/// its due time and keeps `limits`, with the same clock as
/// [`RouteEvaluator::build_route`]: departure at 0, travel time equal to
/// distance, waiting for a window that has not opened, and service before
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
        let travel = distances.get(prev, cid);
        distance += travel;
        let arrival = time + travel;
        let customer = &customers[cid];
        let start = match customer.time_window() {
            Some(tw) => {
                if tw.is_violated(arrival) {
                    return false;
                }
                arrival + tw.waiting_time(arrival)
            }
            None => arrival,
        };
        time = start + customer.service_duration();
        prev = cid;
    }
    let back = distances.get(prev, depot);
    limits.allow(distance + back, time + back)
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
}
