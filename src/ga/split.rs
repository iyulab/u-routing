//! Split algorithm for partitioning a giant tour into feasible routes.
//!
//! # Algorithm
//!
//! Given a giant tour (permutation of customers), finds the optimal partition
//! into sub-routes such that each route respects vehicle capacity and the
//! total distance is minimized.
//!
//! Models the problem as a shortest-path problem on an auxiliary graph where
//! node i represents the boundary after customer i, and edge (i, j) represents
//! serving customers i+1..=j in one route.
//!
//! # Complexity
//!
//! O(n²) where n = number of customers (worst case when all customers fit
//! in a single route). In practice much faster due to capacity pruning.
//!
//! # Reference
//!
//! Prins, C. (2004). "A simple and effective evolutionary algorithm for the
//! vehicle routing problem", *Computers & Operations Research* 31(12), 1985-2002.

use crate::distance::DistanceMatrix;
use crate::models::Customer;

/// Result of the split algorithm.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Routes as sequences of customer IDs.
    pub routes: Vec<Vec<usize>>,
    /// Total distance of all routes.
    pub total_distance: f64,
}

/// Splits a giant tour into optimal sub-routes using dynamic programming.
///
/// Each sub-route starts and ends at the depot and respects the given
/// vehicle capacity.
///
/// # Arguments
///
/// * `tour` — Customer IDs in giant-tour order (excluding depot)
/// * `customers` — All locations (index 0 = depot)
/// * `distances` — Distance matrix
/// * `capacity` — Vehicle capacity
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::ga::split;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 3.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// // Giant tour: [1, 2, 3] — customers in order
/// let result = split(&[1, 2, 3], &customers, &dm, 30);
/// assert_eq!(result.routes.len(), 1); // all fit in one route
/// assert!((result.total_distance - 6.0).abs() < 1e-10);
/// ```
pub fn split(
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

    // cost[i] = minimum total distance to serve tour[0..i]
    // pred[i] = predecessor index (start of the last route ending at i)
    let mut cost = vec![f64::INFINITY; n + 1];
    let mut pred = vec![0usize; n + 1];
    cost[0] = 0.0;

    for i in 0..n {
        if cost[i] == f64::INFINITY {
            continue;
        }

        let mut load = 0i32;
        let mut route_dist = 0.0;

        for j in i..n {
            let cid = tour[j];
            load += customers[cid].demand();

            if load > capacity {
                break;
            }

            // Add distance: prev → cid
            if j == i {
                // First customer in route: depot → cid
                route_dist = distances.get(depot, cid);
            } else {
                // Extend: prev_customer → cid
                route_dist += distances.get(tour[j - 1], cid);
            }

            // Complete route: ... → cid → depot
            let total_route = route_dist + distances.get(cid, depot);
            let new_cost = cost[i] + total_route;

            if new_cost < cost[j + 1] {
                cost[j + 1] = new_cost;
                pred[j + 1] = i;
            }
        }
    }

    // Backtrack to find routes
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

    fn line_customers() -> (Vec<Customer>, DistanceMatrix) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn test_split_single_route() {
        let (cust, dm) = line_customers();
        let result = split(&[1, 2, 3], &cust, &dm, 30);
        assert_eq!(result.routes.len(), 1);
        assert_eq!(result.routes[0], vec![1, 2, 3]);
        // 0→1→2→3→0 = 1+1+1+3 = 6
        assert!((result.total_distance - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_forced_two_routes() {
        let (cust, dm) = line_customers();
        // Capacity 20 can hold 2 customers (demand 10 each)
        let result = split(&[1, 2, 3], &cust, &dm, 20);
        assert_eq!(result.routes.len(), 2);
        // Optimal split: [1]+[2,3] = (0→1→0)+(0→2→3→0) = 2+6 = 8
        assert_eq!(result.routes[0], vec![1]);
        assert_eq!(result.routes[1], vec![2, 3]);
        assert!((result.total_distance - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_each_alone() {
        let (cust, dm) = line_customers();
        // Capacity 10 — each customer alone
        let result = split(&[1, 2, 3], &cust, &dm, 10);
        assert_eq!(result.routes.len(), 3);
        // (0→1→0)+(0→2→0)+(0→3→0) = 2+4+6 = 12
        assert!((result.total_distance - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_empty() {
        let (cust, dm) = line_customers();
        let result = split(&[], &cust, &dm, 30);
        assert!(result.routes.is_empty());
        assert_eq!(result.total_distance, 0.0);
    }

    #[test]
    fn test_split_single_customer() {
        let (cust, dm) = line_customers();
        let result = split(&[2], &cust, &dm, 30);
        assert_eq!(result.routes.len(), 1);
        assert_eq!(result.routes[0], vec![2]);
        // 0→2→0 = 4
        assert!((result.total_distance - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_reversed_order() {
        let (cust, dm) = line_customers();
        // Tour [3,2,1] — reversed
        let result = split(&[3, 2, 1], &cust, &dm, 30);
        assert_eq!(result.routes.len(), 1);
        // 0→3→2→1→0 = 3+1+1+1 = 6
        assert!((result.total_distance - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_optimal_partition() {
        // Cluster layout: two clusters far apart
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 10.0, 0.0, 10, 0.0),
            Customer::new(4, 11.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        // Tour: [1,2,3,4], capacity 20 → split into [1,2] + [3,4]
        let result = split(&[1, 2, 3, 4], &customers, &dm, 20);
        assert_eq!(result.routes.len(), 2);
        assert_eq!(result.routes[0], vec![1, 2]);
        assert_eq!(result.routes[1], vec![3, 4]);
        // [1,2]: 0→1→2→0 = 1+1+2 = 4
        // [3,4]: 0→3→4→0 = 10+1+11 = 22
        // Total: 26
        assert!((result.total_distance - 26.0).abs() < 1e-10);
    }
}
