//! Time-window-aware split algorithm for VRPTW.
//!
//! # Algorithm
//!
//! Extension of the Prins (2004) split that additionally checks time window
//! feasibility. An edge (i, j) in the auxiliary graph is only valid if the
//! sub-route tour[i..=j] can be executed without violating any time window.
//!
//! For each sub-route candidate, simulates the timing forward from the depot:
//! arrival → wait (if early) → service → next customer. If any customer's
//! arrival exceeds its due date, the sub-route is infeasible and pruned.
//!
//! # Complexity
//!
//! O(n²) — same as the original split.
//!
//! # Reference
//!
//! Prins, C. (2004). "A simple and effective evolutionary algorithm for the
//! vehicle routing problem", *Computers & Operations Research* 31(12), 1985-2002.
//!
//! Solomon, M.M. (1987). "Algorithms for the Vehicle Routing and Scheduling
//! Problems with Time Window Constraints", *Operations Research* 35(2), 254-265.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteLimits;
use crate::models::Customer;

use super::split::SplitResult;

/// Splits a giant tour into sub-routes respecting both capacity and time windows.
///
/// Each sub-route starts and ends at the depot, and all time window constraints
/// are satisfied (arrival ≤ due for each customer).
///
/// A customer that cannot be served even alone (too heavy for the vehicle, a
/// time window out of reach, a limit it breaks by itself) is left out of the
/// routes; the customers around it are still split.
///
/// # Arguments
///
/// * `tour` — Customer IDs in giant-tour order (excluding depot)
/// * `customers` — All locations (index 0 = depot, with optional time windows)
/// * `distances` — Distance matrix
/// * `capacity` — Vehicle capacity
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, TimeWindow};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::ga::split_tw;
/// use u_routing::evaluation::RouteLimits;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 20.0).unwrap()),
///     Customer::new(2, 2.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 20.0).unwrap()),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// let result = split_tw(&[1, 2], &customers, &dm, 30, &RouteLimits::NONE, 0.0);
/// assert_eq!(result.routes.len(), 1);
/// ```
pub fn split_tw(
    tour: &[usize],
    customers: &[Customer],
    distances: &DistanceMatrix,
    capacity: i32,
    limits: &RouteLimits,
    route_cost: f64,
) -> SplitResult {
    let n = tour.len();

    if n == 0 {
        return SplitResult {
            routes: vec![],
            total_distance: 0.0,
        };
    }

    let depot = 0;

    let mut cost = vec![f64::INFINITY; n + 1];
    let mut dist = vec![0.0; n + 1];
    let mut pred = vec![0usize; n + 1];
    cost[0] = 0.0;

    for i in 0..n {
        if cost[i] == f64::INFINITY {
            continue;
        }

        let mut load = 0i32;
        let mut route_dist = 0.0;
        let mut time = 0.0;

        for j in i..n {
            let cid = tour[j];
            load += customers[cid].demand();

            if load > capacity {
                break;
            }

            // Compute distance
            if j == i {
                route_dist = distances.get(depot, cid);
                time = distances.travel_time(depot, cid);
            } else {
                route_dist += distances.get(tour[j - 1], cid);
                time += distances.travel_time(tour[j - 1], cid);
            }

            // Check time window
            if let Some(tw) = customers[cid].time_window() {
                if time > tw.due() {
                    break;
                }
                // Wait if early
                time = time.max(tw.ready());
            }

            // Add service time
            time += customers[cid].service_duration();

            // Complete route cost: ... → cid → depot
            let back = distances.get(cid, depot);
            let total_route = route_dist + back;
            // A route over the vehicle's distance or duration limit is not an
            // arc of the split graph. A longer one may still be (its return
            // leg can be shorter), so the scan goes on.
            if !limits.allow(total_route, time + distances.travel_time(cid, depot)) {
                continue;
            }
            let new_cost = cost[i] + route_cost + total_route;

            if new_cost < cost[j + 1] {
                cost[j + 1] = new_cost;
                dist[j + 1] = dist[i] + total_route;
                pred[j + 1] = i;
            }
        }
    }

    // The first position no segment reaches is a customer that cannot even ride
    // alone (too heavy, window out of reach, limits it breaks by itself). Leave
    // it out and split the rest -- the backtrack from an unreachable end would
    // otherwise cut the tour there and drop every customer after it as well.
    if let Some(stuck) = (1..=n).find(|&j| cost[j] == f64::INFINITY) {
        let rest: Vec<usize> = tour
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != stuck - 1)
            .map(|(_, &c)| c)
            .collect();
        return split_tw(&rest, customers, distances, capacity, limits, route_cost);
    }

    let mut routes = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = pred[j];
        routes.push(tour[i..j].to_vec());
        j = i;
    }
    routes.reverse();

    SplitResult {
        routes,
        total_distance: dist[n],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_distance_leaves_the_route_cost_out() {
        // As `split` reports it: the routes' length, with the route cost only
        // deciding where to cut. It used to include the cost, which the GA's
        // fitness then added again.
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 0.0, 1, 0.0),
            Customer::new(2, -3.0, 0.0, 1, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let tw = split_tw(&[1, 2], &customers, &dm, 1, &RouteLimits::NONE, 100.0);
        let plain = super::super::split::split(&[1, 2], &customers, &dm, 1, 100.0);
        assert_eq!(tw.routes.len(), 2);
        assert!(
            (tw.total_distance - 12.0).abs() < 1e-9,
            "{}",
            tw.total_distance
        );
        assert!((tw.total_distance - plain.total_distance).abs() < 1e-9);
    }
    use crate::models::TimeWindow;

    #[test]
    fn test_split_tw_all_feasible() {
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
        let result = split_tw(&[1, 2, 3], &customers, &dm, 30, &RouteLimits::NONE, 0.0);
        assert_eq!(result.routes.len(), 1);
    }

    #[test]
    fn test_split_tw_forces_split() {
        // Customer 2 has tight window that forces it on a separate route
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 6.0).expect("valid")),
            Customer::new(2, -5.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 6.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        // Tour [1, 2]: after visiting 1 (arrive=5, service=5, depart=10), travel to 2 takes 10, arrive=20 > due=6
        let result = split_tw(&[1, 2], &customers, &dm, 100, &RouteLimits::NONE, 0.0);
        assert_eq!(result.routes.len(), 2);
    }

    #[test]
    fn test_split_tw_no_time_windows() {
        // Without TW, behaves like regular split
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let result = split_tw(&[1, 2, 3], &customers, &dm, 30, &RouteLimits::NONE, 0.0);
        assert_eq!(result.routes.len(), 1);
        assert!((result.total_distance - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_tw_waiting() {
        // Customer has late window — waiting is fine, still feasible
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 2.0)
                .with_time_window(TimeWindow::new(10.0, 20.0).expect("valid")),
            Customer::new(2, 2.0, 0.0, 10, 2.0)
                .with_time_window(TimeWindow::new(14.0, 30.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        // Cust 1: arrive=1, wait to 10, service=2, depart=12
        // Cust 2: arrive=12+1=13, wait to 14, service=2, depart=16
        let result = split_tw(&[1, 2], &customers, &dm, 30, &RouteLimits::NONE, 0.0);
        assert_eq!(result.routes.len(), 1);
    }

    #[test]
    fn test_split_tw_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let result = split_tw(&[], &customers, &dm, 30, &RouteLimits::NONE, 0.0);
        assert!(result.routes.is_empty());
        assert_eq!(result.total_distance, 0.0);
    }

    #[test]
    fn test_split_tw_unservable_customer_does_not_drop_the_tour_after_it() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 99, 0.0), // heavier than the vehicle
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let result = split_tw(&[1, 2, 3], &customers, &dm, 20, &RouteLimits::NONE, 0.0);
        let served: Vec<usize> = result.routes.iter().flatten().copied().collect();
        assert_eq!(served, vec![1, 3], "{:?}", result.routes);
        // 0 -> 1 -> 3 -> 0
        assert!((result.total_distance - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_split_tw_capacity_and_tw_combined() {
        // Both capacity and time window constraints active
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 15, 0.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
            Customer::new(2, 2.0, 0.0, 15, 0.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
            Customer::new(3, 3.0, 0.0, 15, 0.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        // Capacity 25: can hold at most 1 customer each (15+15=30>25)
        let result = split_tw(&[1, 2, 3], &customers, &dm, 25, &RouteLimits::NONE, 0.0);
        assert!(result.routes.len() >= 2);
    }
}
