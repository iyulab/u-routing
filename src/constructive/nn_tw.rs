//! Time-window-aware nearest-neighbor heuristic.
//!
//! # Algorithm
//!
//! Extension of the greedy nearest-neighbor that also checks time window
//! feasibility before inserting a customer. Only customers whose time window
//! can still be satisfied (arrival ≤ due) are considered as candidates.
//! Among feasible candidates, the nearest one is selected.
//!
//! # Complexity
//!
//! O(n²) where n = number of customers.
//!
//! # Reference
//!
//! Solomon, M.M. (1987). "Algorithms for the Vehicle Routing and Scheduling
//! Problems with Time Window Constraints", *Operations Research* 35(2), 254-265.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{Customer, Solution, Vehicle};

/// Constructs a VRPTW solution using a time-window-aware nearest-neighbor.
///
/// At each step, selects the nearest unvisited customer whose time window
/// is still reachable (arrival ≤ due). Starts a new route when no feasible
/// customer can be added (due to capacity or time).
///
/// # Arguments
///
/// * `customers` — All locations (index 0 = depot, with time windows)
/// * `distances` — Distance matrix
/// * `vehicles` — Available vehicles
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle, TimeWindow};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::constructive::nearest_neighbor_tw;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 10.0).unwrap()),
///     Customer::new(2, 2.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 20.0).unwrap()),
///     Customer::new(3, 3.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 30.0).unwrap()),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 30)];
///
/// let solution = nearest_neighbor_tw(&customers, &dm, &vehicles);
/// assert_eq!(solution.num_served(), 3);
/// ```
pub fn nearest_neighbor_tw(
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicles: &[Vehicle],
) -> Solution {
    let n = customers.len();
    if n <= 1 {
        return Solution::new();
    }

    let mut visited = vec![false; n];
    visited[0] = true; // depot

    let mut solution = Solution::new();
    let mut vehicle_idx = 0;

    loop {
        if vehicle_idx >= vehicles.len() {
            for (i, &v) in visited.iter().enumerate() {
                if !v && i > 0 {
                    solution.add_unassigned(i);
                }
            }
            break;
        }

        let vehicle = &vehicles[vehicle_idx];
        let evaluator = RouteEvaluator::new(customers, distances, vehicle);
        let depot = vehicle.depot_id();
        let mut current = depot;
        let mut current_time = 0.0;
        let mut current_load: i32 = 0;
        let mut route_customers = Vec::new();

        loop {
            let mut best: Option<(usize, f64)> = None;

            for i in 1..n {
                if visited[i] {
                    continue;
                }

                // Check capacity
                let demand = customers[i].demand();
                if current_load + demand > vehicle.capacity() {
                    continue;
                }

                // Check time window feasibility
                let travel = distances.get(current, i);
                let arrival = current_time + travel;

                if let Some(tw) = customers[i].time_window() {
                    if arrival > tw.due() {
                        continue; // cannot arrive before window closes
                    }
                }

                // Among feasible customers, pick nearest
                let d = distances.get(current, i);
                if best.is_none_or(|(_, best_d)| d < best_d) {
                    best = Some((i, d));
                }
            }

            match best {
                Some((next, _)) => {
                    visited[next] = true;
                    route_customers.push(next);

                    let travel = distances.get(current, next);
                    let arrival = current_time + travel;

                    // Update time considering waiting
                    let service_start = if let Some(tw) = customers[next].time_window() {
                        arrival + tw.waiting_time(arrival)
                    } else {
                        arrival
                    };

                    current_time = service_start + customers[next].service_duration();
                    current_load += customers[next].demand();
                    current = next;
                }
                None => break,
            }
        }

        if !route_customers.is_empty() {
            let (route, _) = evaluator.build_route(&route_customers);
            solution.add_route(route);
        }

        vehicle_idx += 1;

        if visited.iter().skip(1).all(|&v| v) {
            break;
        }
    }

    let total_dist = solution.total_distance();
    solution.set_total_cost(total_dist);
    solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    #[test]
    fn test_nn_tw_all_feasible() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 2.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
            Customer::new(2, 2.0, 0.0, 10, 2.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
            Customer::new(3, 3.0, 0.0, 10, 2.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 0);
        assert_eq!(sol.num_routes(), 1);
    }

    #[test]
    fn test_nn_tw_tight_windows_force_split() {
        // Customer 1 has a tight early window, customer 2 has a tight late window
        // Both are near depot but can't share a route due to time
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 2.0).expect("valid")),
            Customer::new(2, -1.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 2.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100), Vehicle::new(1, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        assert_eq!(sol.num_served(), 2);
        assert_eq!(sol.num_unassigned(), 0);
        // After visiting 1 (arrive=1, wait=0, depart=6), reaching 2 at time 8 > due 2
        // So customer 2 needs separate route
        assert_eq!(sol.num_routes(), 2);
    }

    #[test]
    fn test_nn_tw_infeasible_window() {
        // Customer 2 has a window that closes before we can reach it
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
            Customer::new(2, 100.0, 0.0, 10, 0.0)
                .with_time_window(TimeWindow::new(0.0, 5.0).expect("valid")), // too far, window closes at 5
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        // Customer 2 is 100 units away, window closes at 5 → infeasible
        // With 1 vehicle: after visiting customer 1 at time 1+0=1, going to 2 takes 99, arrival=100 > due=5
        // From depot directly: arrival=100 > due=5
        // So customer 2 is unassigned
        assert_eq!(sol.num_served(), 1);
        assert_eq!(sol.num_unassigned(), 1);
    }

    #[test]
    fn test_nn_tw_waiting_time() {
        // Customer has a late window — vehicle arrives early and waits
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(10.0, 20.0).expect("valid")),
            Customer::new(2, 2.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(16.0, 30.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        assert_eq!(sol.num_served(), 2);
        assert_eq!(sol.num_unassigned(), 0);
        // Customer 1: arrive=1, wait until 10, service 5, depart=15
        // Customer 2: arrive=15+1=16, within [16,30], service 5, depart=21
        let visits = sol.routes()[0].visits();
        assert!((visits[0].arrival_time - 1.0).abs() < 1e-10);
        assert!((visits[0].departure_time - 15.0).abs() < 1e-10);
        assert!((visits[1].arrival_time - 16.0).abs() < 1e-10);
        assert!((visits[1].departure_time - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_nn_tw_no_time_windows() {
        // Falls back to standard nearest-neighbor behavior
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        assert_eq!(sol.num_served(), 2);
        assert_eq!(sol.num_routes(), 1);
    }

    #[test]
    fn test_nn_tw_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        assert_eq!(sol.num_routes(), 0);
    }

    #[test]
    fn test_nn_tw_selects_nearest_feasible() {
        // Customer 1 is far but feasible, customer 2 is near but infeasible
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 0.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
            Customer::new(2, 1.0, 0.0, 10, 0.0)
                .with_time_window(TimeWindow::new(0.0, 0.5).expect("valid")), // window closes before arrival
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor_tw(&customers, &dm, &vehicles);
        // Customer 2 is nearest but arrives at t=1 > due=0.5
        // So only customer 1 is served
        assert_eq!(sol.num_served(), 1);
        assert_eq!(sol.routes()[0].customer_ids(), vec![1]);
    }
}
