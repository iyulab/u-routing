//! Checking a whole plan against its problem.

use super::RouteEvaluator;
use crate::distance::DistanceMatrix;
use crate::models::{num_depots, Customer, Solution, Vehicle, Violation, ViolationType};

/// The vehicles a plan may use.
#[derive(Debug, Clone, Copy)]
pub struct Fleet<'a> {
    /// The vehicle types; a route names one by its index here.
    pub vehicles: &'a [Vehicle],
    /// How many vehicles of each type there are, in the order of `vehicles`:
    /// at most that many routes may use the type. `None` is without limit, and
    /// so is an empty slice.
    pub counts: &'a [Option<usize>],
    /// The most routes the plan may use, if bounded.
    pub max_routes: Option<usize>,
}

/// Relative tolerance of a route's reported distance against its own.
const DISTANCE_TOLERANCE: f64 = 1e-6;

/// Every way `solution` breaks its problem: each customer served exactly once
/// or reported unassigned (and nothing else reported), each route's vehicle in
/// the fleet and used as the fleet allows, no more routes than the fleet
/// allows, each route within its vehicle's capacity, time windows, distance
/// and duration limits, and each route's reported distance
/// (`Route::total_distance`) equal to its own. Empty means the plan is
/// feasible as reported.
///
/// Customer ids are indices into `customers`, whose first entries are the
/// depots (one, or several -- see [`num_depots`]); the rest are the customers.
/// Each route is judged with its own vehicle (`Route::vehicle_id`), from its
/// customer sequence alone, starting and ending at that vehicle's depot
/// ([`Vehicle::depot_id`], which has to be one of the depots) -- the visit
/// timings a route carries are recomputed, not trusted.
///
/// # Complexity
/// O(n + r) for n visits and r routes, after the O(1) distance lookups.
///
/// # Examples
///
/// ```
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::evaluation::{check_solution, Fleet};
/// use u_routing::models::{Customer, Route, Solution, Vehicle, Visit, ViolationType};
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 3.0, 4.0, 15, 0.0),
///     Customer::new(2, 6.0, 8.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = [Vehicle::new(0, 20)];
/// let fleet = Fleet { vehicles: &vehicles, counts: &[], max_routes: None };
///
/// let mut route = Route::new(0);
/// for id in [1, 2] {
///     route.push_visit(Visit { customer_id: id, arrival_time: 0.0, departure_time: 0.0, load_after: 0 });
/// }
/// route.set_total_distance(20.0); // 5 out, 5 across, 10 back
/// let mut plan = Solution::new();
/// plan.add_route(route);
///
/// let violations = check_solution(&customers, &dm, &fleet, &plan);
/// // 15 + 10 is over a capacity of 20.
/// assert!(matches!(violations[0].kind, ViolationType::CapacityExceeded { load: 25, capacity: 20, .. }));
/// ```
pub fn check_solution(
    customers: &[Customer],
    distances: &DistanceMatrix,
    fleet: &Fleet<'_>,
    solution: &Solution,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    let mut push = |kind| violations.push(Violation::new(kind));
    let depots = num_depots(customers);
    let is_customer = |id: usize| id >= depots && id < customers.len();

    if let Some(max_routes) = fleet.max_routes {
        if solution.num_routes() > max_routes {
            push(ViolationType::TooManyRoutes {
                routes: solution.num_routes(),
                max_routes,
            });
        }
    }

    let mut served = vec![false; customers.len()];
    let mut vehicle_used = vec![0usize; fleet.vehicles.len()];
    for (route_index, route) in solution.routes().iter().enumerate() {
        let ids = route.customer_ids();
        let mut known = true;
        for &id in &ids {
            if !is_customer(id) {
                known = false;
                push(ViolationType::UnknownCustomer {
                    route_index,
                    customer_id: id,
                });
            } else if served[id] {
                push(ViolationType::CustomerVisitedTwice {
                    customer_id: id,
                    route_index,
                });
            } else {
                served[id] = true;
            }
        }

        let vehicle_id = route.vehicle_id();
        let Some(vehicle) = fleet.vehicles.get(vehicle_id) else {
            push(ViolationType::UnknownVehicle {
                route_index,
                vehicle_id,
            });
            continue;
        };
        vehicle_used[vehicle_id] += 1;
        if let Some(Some(count)) = fleet.counts.get(vehicle_id) {
            if vehicle_used[vehicle_id] == count + 1 {
                push(ViolationType::FleetCountExceeded {
                    route_index,
                    vehicle_id,
                    count: *count,
                });
            }
        }
        // A route through an id that is not a customer cannot be timed.
        if !known {
            continue;
        }
        let (timed, route_violations) =
            RouteEvaluator::new(customers, distances, vehicle).build_route(&ids);
        for mut v in route_violations {
            match &mut v.kind {
                ViolationType::CapacityExceeded { route_index: r, .. }
                | ViolationType::TimeWindowViolated { route_index: r, .. }
                | ViolationType::MaxDistanceExceeded { route_index: r, .. }
                | ViolationType::MaxDurationExceeded { route_index: r, .. }
                | ViolationType::BreakNotPlaceable { route_index: r } => *r = route_index,
                _ => {}
            }
            push(v.kind);
        }
        let (reported, actual) = (route.total_distance(), timed.total_distance());
        if (reported - actual).abs() > DISTANCE_TOLERANCE * actual.abs() {
            push(ViolationType::DistanceMismatch {
                route_index,
                reported,
                actual,
            });
        }
        let (reported, actual) = (route.lateness_cost(), timed.lateness_cost());
        if (reported - actual).abs() > DISTANCE_TOLERANCE * actual.abs() {
            push(ViolationType::LatenessMismatch {
                route_index,
                reported,
                actual,
            });
        }
    }

    let mut reported_unassigned = vec![false; customers.len()];
    for &id in solution.unassigned() {
        if !is_customer(id) || served[id] || reported_unassigned[id] {
            push(ViolationType::WronglyUnassigned { customer_id: id });
        } else {
            reported_unassigned[id] = true;
        }
    }
    for id in depots..customers.len() {
        if !served[id] && !reported_unassigned[id] {
            push(ViolationType::CustomerMissing { customer_id: id });
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Route, TimeWindow, Visit};

    fn four() -> (Vec<Customer>, DistanceMatrix) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 5, 0.0),
            Customer::new(2, 2.0, 0.0, 5, 0.0),
            Customer::new(3, 0.0, 3.0, 5, 0.0),
            Customer::new(4, 0.0, 4.0, 5, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    /// A route through `ids` reporting its own distance, depot to depot.
    fn route(vehicle: usize, ids: &[usize], dm: &DistanceMatrix) -> Route {
        let mut r = Route::new(vehicle);
        let (mut prev, mut total) = (0, 0.0);
        for &id in ids {
            r.push_visit(Visit {
                customer_id: id,
                arrival_time: 0.0,
                departure_time: 0.0,
                load_after: 0,
            });
            if id < dm.size() {
                total += dm.get(prev, id);
                prev = id;
            }
        }
        r.set_total_distance(total + dm.get(prev, 0));
        r
    }

    fn plan(routes: Vec<Route>, unassigned: &[usize]) -> Solution {
        let mut s = Solution::new();
        for r in routes {
            s.add_route(r);
        }
        for &u in unassigned {
            s.add_unassigned(u);
        }
        s
    }

    fn kinds(v: &[Violation]) -> Vec<&ViolationType> {
        v.iter().map(|v| &v.kind).collect()
    }

    /// Two depots on a line, depot 0 at x = 0 and depot 1 at x = 100, and
    /// customers 2..=5 at x = 10, 20, 90 and 80.
    fn two_depots() -> (Vec<Customer>, DistanceMatrix) {
        let at = |id: usize, x: f64| Customer::new(id, x, 0.0, 5, 0.0);
        let customers = vec![
            Customer::depot_at(0, 0.0, 0.0),
            Customer::depot_at(1, 100.0, 0.0),
            at(2, 10.0),
            at(3, 20.0),
            at(4, 90.0),
            at(5, 80.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    /// A route through `ids` out of `depot`, reporting its own distance.
    fn route_from(vehicle: usize, depot: usize, ids: &[usize], dm: &DistanceMatrix) -> Route {
        let mut r = Route::new(vehicle);
        let (mut prev, mut total) = (depot, 0.0);
        for &id in ids {
            r.push_visit(Visit {
                customer_id: id,
                arrival_time: 0.0,
                departure_time: 0.0,
                load_after: 0,
            });
            total += dm.get(prev, id);
            prev = id;
        }
        r.set_total_distance(total + dm.get(prev, depot));
        r
    }

    #[test]
    fn each_route_runs_from_its_vehicles_depot() {
        let (c, dm) = two_depots();
        let vehicles = [Vehicle::new(0, 20), Vehicle::new(1, 20).with_depot(1)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: Some(2),
        };
        // 0 -> 2 -> 3 -> 0 is 40, and so is 1 -> 4 -> 5 -> 1.
        let feasible = plan(
            vec![
                route_from(0, 0, &[2, 3], &dm),
                route_from(1, 1, &[4, 5], &dm),
            ],
            &[],
        );
        assert!((feasible.routes()[0].total_distance() - 40.0).abs() < 1e-9);
        assert!((feasible.routes()[1].total_distance() - 40.0).abs() < 1e-9);
        assert!(check_solution(&c, &dm, &fleet, &feasible).is_empty());

        // Each route reported as if it ran out of the other vehicle's depot
        // (1 -> 2 -> 3 -> 1 is 180, 0 -> 4 -> 5 -> 0 is 180): both are wrong.
        let swapped = plan(
            vec![
                route_from(0, 1, &[2, 3], &dm),
                route_from(1, 0, &[4, 5], &dm),
            ],
            &[],
        );
        let v = check_solution(&c, &dm, &fleet, &swapped);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(v
            .iter()
            .all(|v| matches!(v.kind, ViolationType::DistanceMismatch { .. })));
        // Only the second one: the first is where its vehicle is.
        let one_wrong = plan(
            vec![
                route_from(0, 0, &[2, 3], &dm),
                route_from(1, 0, &[4, 5], &dm),
            ],
            &[],
        );
        assert!(matches!(
            check_solution(&c, &dm, &fleet, &one_wrong)[..],
            [Violation {
                kind: ViolationType::DistanceMismatch { route_index: 1, .. }
            }]
        ));
    }

    #[test]
    fn a_depot_is_no_customer_and_the_customers_start_after_the_depots() {
        let (c, dm) = two_depots();
        let vehicles = [Vehicle::new(0, 100)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: None,
        };
        // Both depots are refused as a visit; customer 2 is the first customer,
        // reported missing; ids 0 and 1 are not customers to report either.
        let s = plan(
            vec![
                route_from(0, 0, &[3, 4, 5], &dm),
                route_from(0, 0, &[1], &dm),
            ],
            &[0],
        );
        let v = check_solution(&c, &dm, &fleet, &s);
        let expect = [
            ViolationType::UnknownCustomer {
                route_index: 1,
                customer_id: 1,
            },
            ViolationType::WronglyUnassigned { customer_id: 0 },
            ViolationType::CustomerMissing { customer_id: 2 },
        ];
        for e in &expect {
            assert!(kinds(&v).contains(&e), "missing {e:?} in {v:?}");
        }
        // Nothing is said about the depots as customers going missing.
        assert!(!v.iter().any(|v| matches!(
            v.kind,
            ViolationType::CustomerMissing { customer_id: 0 | 1 }
        )));
        assert_eq!(v.len(), 3, "{v:?}");
    }

    #[test]
    fn a_feasible_plan_has_no_violations() {
        let (c, dm) = four();
        let vehicles = [Vehicle::new(0, 10)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: Some(2),
        };
        let s = plan(vec![route(0, &[1, 2], &dm), route(0, &[3, 4], &dm)], &[]);
        assert!(check_solution(&c, &dm, &fleet, &s).is_empty());
    }

    #[test]
    fn every_customer_is_served_once_or_reported_unassigned() {
        let (c, dm) = four();
        let vehicles = [Vehicle::new(0, 100)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: None,
        };
        // 1 twice, 4 missing, 2 served and reported unassigned, 9 and the
        // depot are not customers.
        let s = plan(
            vec![route(0, &[1, 2], &dm), route(0, &[1, 3], &dm)],
            &[2, 9],
        );
        let v = check_solution(&c, &dm, &fleet, &s);
        assert_eq!(
            kinds(&v),
            [
                &ViolationType::CustomerVisitedTwice {
                    customer_id: 1,
                    route_index: 1
                },
                &ViolationType::WronglyUnassigned { customer_id: 2 },
                &ViolationType::WronglyUnassigned { customer_id: 9 },
                &ViolationType::CustomerMissing { customer_id: 4 },
            ]
        );
        let depot = plan(vec![route(0, &[1, 0, 2, 3, 4], &dm)], &[]);
        assert!(kinds(&check_solution(&c, &dm, &fleet, &depot)).contains(
            &&ViolationType::UnknownCustomer {
                route_index: 0,
                customer_id: 0
            }
        ));
    }

    #[test]
    fn each_route_is_judged_with_its_own_vehicle() {
        let (c, dm) = four();
        // Vehicle 1 carries 5, vehicle 0 carries 10: [3, 4] fits only vehicle 0.
        let vehicles = [Vehicle::new(0, 10), Vehicle::new(1, 5)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[Some(1), Some(1)],
            max_routes: None,
        };
        let ok = plan(vec![route(0, &[3, 4], &dm), route(1, &[1], &dm)], &[2]);
        assert!(check_solution(&c, &dm, &fleet, &ok).is_empty());
        let swapped = plan(vec![route(1, &[3, 4], &dm), route(0, &[1, 2], &dm)], &[]);
        assert_eq!(
            kinds(&check_solution(&c, &dm, &fleet, &swapped)),
            [&ViolationType::CapacityExceeded {
                route_index: 0,
                load: 10,
                capacity: 5
            }]
        );
    }

    #[test]
    fn the_fleet_bounds_routes_and_vehicle_use() {
        let (c, dm) = four();
        let vehicles = [Vehicle::new(0, 10)];
        let s = plan(
            vec![
                route(0, &[1, 2], &dm),
                route(0, &[3], &dm),
                route(3, &[4], &dm),
            ],
            &[],
        );
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[Some(1)],
            max_routes: Some(2),
        };
        assert_eq!(
            kinds(&check_solution(&c, &dm, &fleet, &s)),
            [
                &ViolationType::TooManyRoutes {
                    routes: 3,
                    max_routes: 2
                },
                &ViolationType::FleetCountExceeded {
                    route_index: 1,
                    vehicle_id: 0,
                    count: 1
                },
                &ViolationType::UnknownVehicle {
                    route_index: 2,
                    vehicle_id: 3
                },
            ]
        );
    }

    #[test]
    fn limits_windows_and_the_reported_distance_are_checked() {
        let mut c = four().0;
        c[2] = Customer::new(2, 2.0, 0.0, 5, 0.0)
            .with_time_window(TimeWindow::new(0.0, 1.0).expect("a window"));
        let dm = DistanceMatrix::from_customers(&c);
        let vehicles = [Vehicle::new(0, 100)
            .with_max_distance(5.0)
            .with_max_duration(5.0)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: None,
        };
        let mut short = route(0, &[1, 2, 3, 4], &dm);
        short.set_total_distance(1.0);
        let s = plan(vec![short], &[]);
        let v = check_solution(&c, &dm, &fleet, &s);
        let k = kinds(&v);
        assert!(matches!(
            k[0],
            ViolationType::TimeWindowViolated {
                route_index: 0,
                customer_id: 2,
                ..
            }
        ));
        assert!(k
            .iter()
            .any(|k| matches!(k, ViolationType::MaxDistanceExceeded { route_index: 0, .. })));
        assert!(k
            .iter()
            .any(|k| matches!(k, ViolationType::MaxDurationExceeded { route_index: 0, .. })));
        assert!(matches!(
            k.last(),
            Some(ViolationType::DistanceMismatch { route_index: 0, reported, .. }) if *reported == 1.0
        ));
    }

    #[test]
    fn a_late_arrival_is_a_violation_only_at_a_hard_window() {
        let (mut c, dm) = four();
        // Customer 2 is reached at 2 (via customer 1), due at 1: one unit late.
        let window = TimeWindow::new(0.0, 1.0).expect("valid");
        c[2] = c[2].clone().with_time_window(window);
        let vehicles = [Vehicle::new(0, 100)];
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: None,
        };
        let mut late = route(0, &[1, 2], &dm);
        let s = plan(vec![late.clone(), route(0, &[3, 4], &dm)], &[]);
        assert!(matches!(
            kinds(&check_solution(&c, &dm, &fleet, &s))[..],
            [ViolationType::TimeWindowViolated { customer_id: 2, .. }]
        ));

        // Soft: the arrival is allowed, and the route has to say what it costs.
        c[2] = c[2].clone().with_lateness_penalty(3.0).expect("valid");
        let s = plan(vec![late.clone(), route(0, &[3, 4], &dm)], &[]);
        let found = check_solution(&c, &dm, &fleet, &s);
        assert!(
            matches!(
                kinds(&found)[..],
                [ViolationType::LatenessMismatch { route_index: 0, reported, actual }]
                    if *reported == 0.0 && (*actual - 3.0).abs() < 1e-12
            ),
            "{found:?}"
        );
        late.set_lateness(1.0, 3.0);
        let s = plan(vec![late, route(0, &[3, 4], &dm)], &[]);
        assert!(check_solution(&c, &dm, &fleet, &s).is_empty());
    }
}
