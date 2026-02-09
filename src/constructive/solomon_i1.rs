//! Solomon's I1 insertion heuristic for VRPTW.
//!
//! # Algorithm
//!
//! A sequential insertion heuristic that iteratively inserts the "best"
//! unrouted customer into the current route. The insertion criterion
//! combines distance cost with time-based urgency:
//!
//! c1(i,u,j) = α₁·d(i,u) + α₂·d(u,j) - μ·d(i,j)
//!
//! where (i,j) is the edge being broken and u is the customer to insert.
//! The customer with the best (lowest) insertion cost is chosen.
//!
//! When no more customers can be feasibly inserted, a new route is opened.
//!
//! # Complexity
//!
//! O(n² · m) where n = customers, m = routes.
//!
//! # Reference
//!
//! Solomon, M.M. (1987). "Algorithms for the Vehicle Routing and Scheduling
//! Problems with Time Window Constraints", *Operations Research* 35(2), 254-265.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{Customer, Solution, Vehicle};

/// Constructs a VRPTW solution using Solomon's I1 insertion heuristic.
///
/// Builds routes one at a time. For each unrouted customer, evaluates
/// all feasible insertion positions, selecting the customer-position pair
/// with the lowest cost increase. Opens a new route when no feasible
/// insertion remains.
///
/// # Arguments
///
/// * `customers` — All locations (index 0 = depot, with time windows)
/// * `distances` — Distance matrix
/// * `vehicle` — Vehicle type (homogeneous fleet, unlimited count)
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle, TimeWindow};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::constructive::solomon_i1;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 20.0).unwrap()),
///     Customer::new(2, 2.0, 0.0, 10, 2.0)
///         .with_time_window(TimeWindow::new(0.0, 20.0).unwrap()),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicle = Vehicle::new(0, 30);
///
/// let solution = solomon_i1(&customers, &dm, &vehicle);
/// assert_eq!(solution.num_served(), 2);
/// ```
pub fn solomon_i1(
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Solution {
    let n = customers.len();
    if n <= 1 {
        return Solution::new();
    }

    let depot = vehicle.depot_id();
    let evaluator = RouteEvaluator::new(customers, distances, vehicle);

    let mut unrouted: Vec<usize> = (1..n).collect();
    let mut solution = Solution::new();

    while !unrouted.is_empty() {
        // Start a new route: pick the farthest unrouted customer as seed
        let seed_idx = farthest_from_depot(&unrouted, depot, distances);
        let seed = unrouted.remove(seed_idx);
        let mut route_customers = vec![seed];

        // Iteratively insert customers into this route
        loop {
            let mut best_insert: Option<(usize, usize, f64)> = None; // (unrouted_idx, position, cost)

            for (ui, &cid) in unrouted.iter().enumerate() {
                // Check capacity
                let current_load: i32 = route_customers
                    .iter()
                    .map(|&c| customers[c].demand())
                    .sum();
                if current_load + customers[cid].demand() > vehicle.capacity() {
                    continue;
                }

                // Try inserting at every position
                for pos in 0..=route_customers.len() {
                    let prev = if pos == 0 { depot } else { route_customers[pos - 1] };
                    let next = if pos == route_customers.len() {
                        depot
                    } else {
                        route_customers[pos]
                    };

                    // Distance cost
                    let cost = distances.get(prev, cid) + distances.get(cid, next)
                        - distances.get(prev, next);

                    // Check time window feasibility
                    let mut test_route = route_customers.clone();
                    test_route.insert(pos, cid);
                    if !is_tw_feasible(&test_route, depot, customers, distances) {
                        continue;
                    }

                    if best_insert.as_ref().is_none_or(|b| cost < b.2) {
                        best_insert = Some((ui, pos, cost));
                    }
                }
            }

            match best_insert {
                Some((ui, pos, _)) => {
                    let cid = unrouted.remove(ui);
                    route_customers.insert(pos, cid);
                }
                None => break, // No feasible insertion — close this route
            }
        }

        let (route, _) = evaluator.build_route(&route_customers);
        solution.add_route(route);
    }

    let total_dist = solution.total_distance();
    solution.set_total_cost(total_dist);
    solution
}

/// Finds the index of the farthest customer from the depot.
fn farthest_from_depot(
    unrouted: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
) -> usize {
    let mut best_idx = 0;
    let mut best_dist = 0.0;
    for (i, &cid) in unrouted.iter().enumerate() {
        let d = distances.get(depot, cid);
        if d > best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Checks whether a route is feasible with respect to time windows.
fn is_tw_feasible(
    route: &[usize],
    depot: usize,
    customers: &[Customer],
    distances: &DistanceMatrix,
) -> bool {
    let mut time = 0.0;
    let mut prev = depot;

    for &cid in route {
        let travel = distances.get(prev, cid);
        let arrival = time + travel;

        if let Some(tw) = customers[cid].time_window() {
            if arrival > tw.due() {
                return false;
            }
            time = arrival + tw.waiting_time(arrival) + customers[cid].service_duration();
        } else {
            time = arrival + customers[cid].service_duration();
        }
        prev = cid;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    #[test]
    fn test_solomon_all_one_route() {
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
        let vehicle = Vehicle::new(0, 100);
        let sol = solomon_i1(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 0);
    }

    #[test]
    fn test_solomon_capacity_split() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 15, 0.0),
            Customer::new(2, 2.0, 0.0, 15, 0.0),
            Customer::new(3, 3.0, 0.0, 15, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 25);
        let sol = solomon_i1(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 3);
        assert!(sol.num_routes() >= 2);
    }

    #[test]
    fn test_solomon_tw_split() {
        // Time windows force separate routes
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 6.0).expect("valid")),
            Customer::new(2, -5.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 6.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = solomon_i1(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 2);
        // After visiting 1 (arrive=5, service=5, depart=10), can't reach 2 by due=6
        assert_eq!(sol.num_routes(), 2);
    }

    #[test]
    fn test_solomon_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = solomon_i1(&customers, &dm, &vehicle);
        assert_eq!(sol.num_routes(), 0);
    }

    #[test]
    fn test_solomon_single_customer() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 0.0)
                .with_time_window(TimeWindow::new(0.0, 100.0).expect("valid")),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = solomon_i1(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 1);
        assert_eq!(sol.num_routes(), 1);
    }

    #[test]
    fn test_solomon_no_tw() {
        // Works without time windows too
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = solomon_i1(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 2);
    }

    #[test]
    fn test_solomon_seeds_farthest() {
        // Verify farthest customer is used as seed
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 10.0, 0.0, 10, 0.0), // farthest
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let unrouted = vec![1, 2];
        let idx = farthest_from_depot(&unrouted, 0, &dm);
        assert_eq!(unrouted[idx], 2);
    }
}
