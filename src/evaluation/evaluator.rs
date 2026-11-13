//! Route evaluator that computes timing, load, and feasibility.

use crate::models::{Customer, Route, Vehicle, Violation, ViolationType, Visit};

/// Evaluates routes by computing visit timing, cumulative load, total distance,
/// and checking constraints (capacity, time windows, max distance/duration).
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle, TimeWindow};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::evaluation::RouteEvaluator;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 3.0, 4.0, 10, 5.0),
///     Customer::new(2, 6.0, 8.0, 20, 5.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicle = Vehicle::new(0, 100);
///
/// let evaluator = RouteEvaluator::new(&customers, &dm, &vehicle);
/// let (route, violations) = evaluator.build_route(&[1, 2]);
/// assert_eq!(route.len(), 2);
/// assert!(violations.is_empty());
/// ```
pub struct RouteEvaluator<'a> {
    customers: &'a [Customer],
    distances: &'a crate::distance::DistanceMatrix,
    vehicle: &'a Vehicle,
}

impl<'a> RouteEvaluator<'a> {
    /// Creates a new evaluator for the given problem data.
    pub fn new(
        customers: &'a [Customer],
        distances: &'a crate::distance::DistanceMatrix,
        vehicle: &'a Vehicle,
    ) -> Self {
        Self {
            customers,
            distances,
            vehicle,
        }
    }

    /// Builds a route from a sequence of customer IDs, computing timing and load.
    ///
    /// Returns the constructed route and any constraint violations found.
    pub fn build_route(&self, customer_ids: &[usize]) -> (Route, Vec<Violation>) {
        let mut route = Route::new(self.vehicle.id());
        let mut violations = Vec::new();
        let depot_id = self.vehicle.depot_id();
        let mut current_time = 0.0;
        let mut current_load: i32 = 0;
        let mut total_distance = 0.0;
        let mut total_lateness = 0.0;
        let mut lateness_cost = 0.0;
        let mut prev = depot_id;

        // The driver's break, placed where the route finishes earliest.
        let rest = self.vehicle.rest();
        let placed = rest.as_ref().and_then(|b| {
            super::schedule::schedule(customer_ids, depot_id, self.distances, self.customers, b)
        });
        let rest_at = placed.and_then(|s| s.rest);

        for (index, &cid) in customer_ids.iter().enumerate() {
            total_distance += self.distances.get(prev, cid);
            let arrival = current_time + self.distances.travel_time(prev, cid);
            // At the stop the break comes before, the service can start only after it.
            let ready_at = match rest_at {
                Some((at, start)) if at == index => start + rest.map_or(0.0, |b| b.duration()),
                _ => arrival,
            };

            let customer = &self.customers[cid];

            // Check time window
            let service_start = if let Some(tw) = customer.time_window() {
                let arrival = ready_at;
                if tw.is_violated(arrival) {
                    match customer.lateness_penalty() {
                        // A soft window: late is allowed and costs.
                        Some(penalty) => {
                            let late = tw.lateness(arrival);
                            total_lateness += late;
                            lateness_cost += penalty * late;
                        }
                        None => {
                            violations.push(Violation::new(ViolationType::TimeWindowViolated {
                                route_index: 0,
                                customer_id: cid,
                                arrival,
                                due: tw.due(),
                            }))
                        }
                    }
                }
                arrival + tw.waiting_time(arrival)
            } else {
                ready_at
            };

            let departure = service_start + customer.service_duration();
            current_load += customer.demand();

            route.push_visit(Visit {
                customer_id: cid,
                arrival_time: arrival,
                departure_time: departure,
                load_after: current_load,
            });

            current_time = departure;
            prev = cid;
        }

        // Return to depot
        total_distance += self.distances.get(prev, depot_id);
        let total_duration = current_time + self.distances.travel_time(prev, depot_id);

        route.set_total_distance(total_distance);
        route.set_total_duration(total_duration);
        route.set_lateness(total_lateness, lateness_cost);
        route.set_rest(rest_at);
        // No placement of the break keeps the windows and the break's own: if no hard window is
        // reported closed on the timing above, the break is what cannot be placed.
        if rest.is_some()
            && placed.is_none()
            && !violations
                .iter()
                .any(|v| matches!(v.kind, ViolationType::TimeWindowViolated { .. }))
        {
            violations.push(Violation::new(ViolationType::BreakNotPlaceable {
                route_index: 0,
            }));
        }

        // Check capacity
        if current_load > self.vehicle.capacity() {
            violations.push(Violation::new(ViolationType::CapacityExceeded {
                route_index: 0,
                load: current_load,
                capacity: self.vehicle.capacity(),
            }));
        }

        // Check max distance
        if let Some(max_d) = self.vehicle.max_distance() {
            if total_distance > max_d {
                violations.push(Violation::new(ViolationType::MaxDistanceExceeded {
                    route_index: 0,
                    distance: total_distance,
                    max_distance: max_d,
                }));
            }
        }

        // Check max duration
        if let Some(max_t) = self.vehicle.max_duration() {
            if total_duration > max_t {
                violations.push(Violation::new(ViolationType::MaxDurationExceeded {
                    route_index: 0,
                    duration: total_duration,
                    max_duration: max_t,
                }));
            }
        }

        (route, violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMatrix;
    use crate::models::TimeWindow;

    fn setup() -> (Vec<Customer>, DistanceMatrix, Vehicle) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 4.0, 10, 5.0),
            Customer::new(2, 6.0, 8.0, 20, 5.0),
            Customer::new(3, 0.0, 10.0, 15, 5.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 50);
        (customers, dm, vehicle)
    }

    #[test]
    fn test_build_route_empty() {
        let (customers, dm, vehicle) = setup();
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        let (route, violations) = eval.build_route(&[]);
        assert!(route.is_empty());
        assert!(violations.is_empty());
        assert_eq!(route.total_distance(), 0.0);
    }

    #[test]
    fn test_build_route_single() {
        let (customers, dm, vehicle) = setup();
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        let (route, violations) = eval.build_route(&[1]);
        assert_eq!(route.len(), 1);
        assert!(violations.is_empty());
        // depot->1 = 5.0, 1->depot = 5.0
        assert!((route.total_distance() - 10.0).abs() < 1e-10);
        assert_eq!(route.total_load(), 10);
    }

    #[test]
    fn test_build_route_capacity_violated() {
        let (customers, dm, _) = setup();
        let small_vehicle = Vehicle::new(0, 25);
        let eval = RouteEvaluator::new(&customers, &dm, &small_vehicle);
        // 10 + 20 + 15 = 45 > 25
        let (route, violations) = eval.build_route(&[1, 2, 3]);
        assert_eq!(route.len(), 3);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0].kind,
            ViolationType::CapacityExceeded {
                load: 45,
                capacity: 25,
                ..
            }
        ));
    }

    #[test]
    fn test_build_route_within_capacity() {
        let (customers, dm, vehicle) = setup();
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        // 10 + 20 = 30 <= 50
        let (_, violations) = eval.build_route(&[1, 2]);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_build_route_time_window_ok() {
        let tw = TimeWindow::new(0.0, 100.0).expect("valid");
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 4.0, 10, 5.0).with_time_window(tw),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        let (_, violations) = eval.build_route(&[1]);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_build_route_time_window_violated() {
        let tw = TimeWindow::new(0.0, 3.0).expect("valid");
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 4.0, 10, 5.0).with_time_window(tw),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        // Travel time = 5.0 > due = 3.0
        let (_, violations) = eval.build_route(&[1]);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0].kind,
            ViolationType::TimeWindowViolated { customer_id: 1, .. }
        ));
    }

    #[test]
    fn test_build_route_waiting() {
        let tw = TimeWindow::new(20.0, 100.0).expect("valid");
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 4.0, 10, 5.0).with_time_window(tw),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        let (route, violations) = eval.build_route(&[1]);
        assert!(violations.is_empty());
        let visit = &route.visits()[0];
        // Arrival at 5.0, wait until 20.0, service 5.0, depart at 25.0
        assert!((visit.arrival_time - 5.0).abs() < 1e-10);
        assert!((visit.departure_time - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_route_max_distance_violated() {
        let (customers, dm, _) = setup();
        let vehicle = Vehicle::new(0, 100).with_max_distance(8.0);
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        // depot->1->depot = 10.0 > 8.0
        let (_, violations) = eval.build_route(&[1]);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0].kind,
            ViolationType::MaxDistanceExceeded { .. }
        ));
    }

    #[test]
    fn test_build_route_max_duration_violated() {
        let (customers, dm, _) = setup();
        let vehicle = Vehicle::new(0, 100).with_max_duration(5.0);
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        // travel 5 + service 5 + return 5 = 15 > 5.0
        let (_, violations) = eval.build_route(&[1]);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0].kind,
            ViolationType::MaxDurationExceeded { .. }
        ));
    }

    #[test]
    fn test_timing_chain() {
        let (customers, dm, vehicle) = setup();
        let eval = RouteEvaluator::new(&customers, &dm, &vehicle);
        let (route, _) = eval.build_route(&[1, 2]);
        let v1 = &route.visits()[0];
        let v2 = &route.visits()[1];
        // Customer 1 departs, then travels to customer 2
        let expected_arrival_2 = v1.departure_time + dm.get(1, 2);
        assert!((v2.arrival_time - expected_arrival_2).abs() < 1e-10);
    }
}
