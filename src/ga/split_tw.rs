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
use crate::models::Customer;

use super::split::SplitResult;

/// Splits a giant tour into sub-routes respecting both capacity and time windows.
///
/// Each sub-route starts and ends at the depot, and all time window constraints
/// are satisfied (arrival ≤ due for each customer).
///
/// If time windows make it impossible to serve all customers in any partition,
/// the algorithm will skip infeasible sub-routes (some customers may remain
/// unserved, indicated by infinite cost at unreachable positions).
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
/// let result = split_tw(&[1, 2], &customers, &dm, 30);
/// assert_eq!(result.routes.len(), 1);
/// ```
pub fn split_tw(
    tour: &[usize],
    customers: &[Customer],
    distances: &DistanceMatrix,
    capacity: i32,
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
                time = route_dist;
            } else {
                let travel = distances.get(tour[j - 1], cid);
                route_dist += travel;
                time += travel;
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
            let total_route = route_dist + distances.get(cid, depot);
            let new_cost = cost[i] + total_route;

            if new_cost < cost[j + 1] {
                cost[j + 1] = new_cost;
                pred[j + 1] = i;
            }
        }
    }

    // Backtrack to find routes
    if cost[n] == f64::INFINITY {
        // Try to recover as many customers as possible
        // Find the last reachable position
        let mut last = 0;
        for j in (0..=n).rev() {
            if cost[j] < f64::INFINITY {
                last = j;
                break;
            }
        }

        let mut routes = Vec::new();
        let mut j = last;
        while j > 0 {
            let i = pred[j];
            routes.push(tour[i..j].to_vec());
            j = i;
        }
        routes.reverse();

        let total = if last > 0 { cost[last] } else { 0.0 };
        return SplitResult {
            routes,
            total_distance: total,
        };
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
        total_distance: cost[n],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let result = split_tw(&[1, 2, 3], &customers, &dm, 30);
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
        let result = split_tw(&[1, 2], &customers, &dm, 100);
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
        let result = split_tw(&[1, 2, 3], &customers, &dm, 30);
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
        let result = split_tw(&[1, 2], &customers, &dm, 30);
        assert_eq!(result.routes.len(), 1);
    }

    #[test]
    fn test_split_tw_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let result = split_tw(&[], &customers, &dm, 30);
        assert!(result.routes.is_empty());
        assert_eq!(result.total_distance, 0.0);
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
        let result = split_tw(&[1, 2, 3], &customers, &dm, 25);
        assert!(result.routes.len() >= 2);
    }
}
