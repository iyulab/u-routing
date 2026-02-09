//! Clarke-Wright savings algorithm.
//!
//! # Algorithm
//!
//! The savings algorithm (Clarke & Wright, 1964) starts with each customer
//! on its own route (depot → customer → depot). It then merges routes by
//! computing the "savings" of combining the end of one route with the start
//! of another:
//!
//! ```text
//! s(i, j) = d(0, i) + d(0, j) - d(i, j)
//! ```
//!
//! Routes are merged in decreasing order of savings, subject to capacity
//! constraints.
//!
//! # Complexity
//!
//! O(n² log n) where n = number of customers (dominated by sorting savings).
//!
//! # Reference
//!
//! Clarke, G. & Wright, J.W. (1964). "Scheduling of Vehicles from a Central
//! Depot to a Number of Delivery Points", *Operations Research* 12(4), 568-581.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{Customer, Solution, Vehicle};

/// A savings value for merging two customers' routes.
#[derive(Debug)]
struct Saving {
    i: usize,
    j: usize,
    value: f64,
}

/// Constructs a VRP solution using the Clarke-Wright savings algorithm.
///
/// Starts with one route per customer, then merges routes in order of
/// decreasing savings while respecting vehicle capacity.
///
/// # Arguments
///
/// * `customers` — All locations (index 0 = depot)
/// * `distances` — Distance matrix
/// * `vehicle` — Vehicle type (homogeneous fleet, unlimited count)
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::constructive::clarke_wright_savings;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 3.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicle = Vehicle::new(0, 30);
///
/// let solution = clarke_wright_savings(&customers, &dm, &vehicle);
/// assert_eq!(solution.num_served(), 3);
/// ```
pub fn clarke_wright_savings(
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Solution {
    let n = customers.len();
    if n <= 1 {
        return Solution::new();
    }

    let depot = vehicle.depot_id();
    let num_customers = n - 1;

    // Compute savings
    let mut savings = Vec::with_capacity(num_customers * (num_customers - 1) / 2);
    for i in 1..n {
        for j in (i + 1)..n {
            let s = distances.get(depot, i) + distances.get(depot, j) - distances.get(i, j);
            if s > 0.0 {
                savings.push(Saving { i, j, value: s });
            }
        }
    }

    // Sort by decreasing savings
    savings.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .expect("savings should not be NaN")
    });

    // Each customer starts in its own route
    // route_of[customer_id] = route index, route_head[r] = first customer,
    // route_tail[r] = last customer
    let mut route_of = vec![0usize; n];
    let mut route_load = vec![0i32; n];
    let mut route_members: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 1..n {
        route_of[i] = i;
        route_load[i] = customers[i].demand();
        route_members[i].push(i);
    }

    // Merge routes
    for saving in &savings {
        let ri = route_of[saving.i];
        let rj = route_of[saving.j];

        // Skip if same route
        if ri == rj {
            continue;
        }

        // Check capacity
        let combined_load = route_load[ri] + route_load[rj];
        if combined_load > vehicle.capacity() {
            continue;
        }

        // Check that i is at the end of its route and j is at the start (or vice versa)
        let i_at_end = route_members[ri].last() == Some(&saving.i);
        let j_at_start = route_members[rj].first() == Some(&saving.j);
        let i_at_start = route_members[ri].first() == Some(&saving.i);
        let j_at_end = route_members[rj].last() == Some(&saving.j);

        let (merge_from, merge_into, reverse_from, reverse_into) =
            if i_at_end && j_at_start {
                (rj, ri, false, false)
            } else if j_at_end && i_at_start {
                (ri, rj, false, false)
            } else if i_at_end && j_at_end {
                (rj, ri, true, false)
            } else if i_at_start && j_at_start {
                (rj, ri, false, true)
            } else {
                continue;
            };

        // Merge: append members of merge_from into merge_into
        let mut from_members = std::mem::take(&mut route_members[merge_from]);
        if reverse_from {
            from_members.reverse();
        }

        if reverse_into {
            route_members[merge_into].reverse();
        }

        route_members[merge_into].append(&mut from_members);
        route_load[merge_into] = combined_load;
        route_load[merge_from] = 0;

        // Update route assignments
        for &cid in &route_members[merge_into] {
            route_of[cid] = merge_into;
        }
    }

    // Build solution from merged routes
    let evaluator = RouteEvaluator::new(customers, distances, vehicle);
    let mut solution = Solution::new();
    let mut visited = vec![false; n];

    for members in &route_members {
        if members.is_empty() {
            continue;
        }
        let (route, _) = evaluator.build_route(members);
        for &cid in members {
            visited[cid] = true;
        }
        solution.add_route(route);
    }

    for (i, &is_visited) in visited.iter().enumerate().skip(1) {
        if !is_visited {
            solution.add_unassigned(i);
        }
    }

    let total_dist = solution.total_distance();
    solution.set_total_cost(total_dist);
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cw_line() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 30);
        let sol = clarke_wright_savings(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 0);
        // With savings, should merge all into one route
        assert_eq!(sol.num_routes(), 1);
        // Optimal: 0→1→2→3→0 = 6.0
        assert!((sol.total_distance() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_cw_capacity_split() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 15, 0.0),
            Customer::new(2, 2.0, 0.0, 15, 0.0),
            Customer::new(3, 3.0, 0.0, 15, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 25);
        let sol = clarke_wright_savings(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 3);
        // Can't fit all in one route (45 > 25), needs at least 2 routes
        assert!(sol.num_routes() >= 2);
    }

    #[test]
    fn test_cw_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = clarke_wright_savings(&customers, &dm, &vehicle);
        assert_eq!(sol.num_routes(), 0);
    }

    #[test]
    fn test_cw_single_customer() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = clarke_wright_savings(&customers, &dm, &vehicle);
        assert_eq!(sol.num_routes(), 1);
        assert_eq!(sol.num_served(), 1);
        assert!((sol.total_distance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cw_better_than_nn() {
        // Triangle: depot at origin, two customers forming a triangle
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 0.0),  // east
            Customer::new(2, 0.0, 5.0, 10, 0.0),  // north
            Customer::new(3, 5.0, 5.0, 10, 0.0),  // northeast
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = clarke_wright_savings(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 3);
        // Clarke-Wright should produce a reasonably good tour
        assert!(sol.total_distance() < 25.0);
    }

    #[test]
    fn test_cw_savings_computation() {
        // Verify that savings formula is correct
        // s(i,j) = d(0,i) + d(0,j) - d(i,j)
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 0.0, 5, 0.0),
            Customer::new(2, 4.0, 0.0, 5, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        // s(1,2) = 3 + 4 - 1 = 6 > 0, should merge
        let vehicle = Vehicle::new(0, 100);
        let sol = clarke_wright_savings(&customers, &dm, &vehicle);
        assert_eq!(sol.num_routes(), 1);
        // 0→1→2→0 = 3 + 1 + 4 = 8, vs separate = 6 + 8 = 14
        assert!((sol.total_distance() - 8.0).abs() < 1e-10);
    }
}
