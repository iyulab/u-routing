//! Sweep constructive heuristic.
//!
//! # Algorithm
//!
//! Sorts customers by polar angle relative to the depot, then groups them
//! into routes by sweeping through angles while respecting capacity. This
//! exploits geographic clustering: nearby customers tend to have similar
//! angles and are placed on the same route.
//!
//! # Complexity
//!
//! O(n log n) where n = number of customers (dominated by angle sorting).
//!
//! # Reference
//!
//! Gillett, B.E. & Miller, L.R. (1974). "A Heuristic Algorithm for the
//! Vehicle-Dispatch Problem", *Operations Research* 22(2), 340-349.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{Customer, Solution, Vehicle};

/// Constructs a VRP solution using the sweep heuristic.
///
/// Sorts customers by polar angle from the depot, then packs them into
/// routes sequentially until capacity is reached. Each full route starts
/// a new vehicle.
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
/// use u_routing::constructive::sweep;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 1.0, 10, 0.0),
///     Customer::new(2, -1.0, 1.0, 10, 0.0),
///     Customer::new(3, -1.0, -1.0, 10, 0.0),
///     Customer::new(4, 1.0, -1.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicle = Vehicle::new(0, 30);
///
/// let solution = sweep(&customers, &dm, &vehicle);
/// assert_eq!(solution.num_served(), 4);
/// ```
pub fn sweep(customers: &[Customer], distances: &DistanceMatrix, vehicle: &Vehicle) -> Solution {
    let n = customers.len();
    if n <= 1 {
        return Solution::new();
    }

    let depot = &customers[vehicle.depot_id()];
    let depot_x = depot.x();
    let depot_y = depot.y();

    // Compute polar angle for each non-depot customer
    let mut angle_order: Vec<(usize, f64)> = (1..n)
        .map(|i| {
            let dx = customers[i].x() - depot_x;
            let dy = customers[i].y() - depot_y;
            let angle = dy.atan2(dx);
            (i, angle)
        })
        .collect();

    // Sort by angle (ascending)
    angle_order.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("angles should not be NaN"));

    // Build routes by sweeping through sorted customers
    let evaluator = RouteEvaluator::new(customers, distances, vehicle);
    let mut solution = Solution::new();
    let mut current_load: i32 = 0;
    let mut current_route: Vec<usize> = Vec::new();

    for &(cid, _) in &angle_order {
        let demand = customers[cid].demand();

        if current_load + demand > vehicle.capacity() && !current_route.is_empty() {
            // Finalize current route
            let (route, _) = evaluator.build_route(&current_route);
            solution.add_route(route);
            current_route.clear();
            current_load = 0;
        }

        if demand <= vehicle.capacity() {
            current_route.push(cid);
            current_load += demand;
        } else {
            // Single customer exceeds capacity — mark unassigned
            solution.add_unassigned(cid);
        }
    }

    // Add remaining route
    if !current_route.is_empty() {
        let (route, _) = evaluator.build_route(&current_route);
        solution.add_route(route);
    }

    let total_dist = solution.total_distance();
    solution.set_total_cost(total_dist);
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sweep_all_one_route() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 0);
        assert_eq!(sol.num_routes(), 1);
    }

    #[test]
    fn test_sweep_splits_by_capacity() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 15, 0.0),
            Customer::new(2, -1.0, 1.0, 15, 0.0),
            Customer::new(3, -1.0, -1.0, 15, 0.0),
            Customer::new(4, 1.0, -1.0, 15, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 25);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 4);
        assert_eq!(sol.num_unassigned(), 0);
        // Capacity 25 with demand 15 each: at most 1 per route = 4 routes
        assert!(sol.num_routes() >= 2);
    }

    #[test]
    fn test_sweep_clusters_geographically() {
        // Customers in two clusters: NE quadrant and SW quadrant
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 10, 0.0),   // NE ~45°
            Customer::new(2, 1.5, 1.5, 10, 0.0),   // NE ~45°
            Customer::new(3, -1.0, -1.0, 10, 0.0), // SW ~-135°
            Customer::new(4, -1.5, -1.5, 10, 0.0), // SW ~-135°
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 20);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 4);
        assert_eq!(sol.num_unassigned(), 0);
        assert_eq!(sol.num_routes(), 2);
        // Verify clustering: each route should have 2 nearby customers
        for route in sol.routes() {
            assert_eq!(route.len(), 2);
        }
    }

    #[test]
    fn test_sweep_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_routes(), 0);
        assert_eq!(sol.num_served(), 0);
    }

    #[test]
    fn test_sweep_single_customer() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_routes(), 1);
        assert_eq!(sol.num_served(), 1);
        assert!((sol.total_distance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sweep_oversized_customer() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 200, 0.0), // exceeds capacity
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 2);
        assert_eq!(sol.num_unassigned(), 1);
    }

    #[test]
    fn test_sweep_angular_ordering() {
        // Customers at known angles: 0°, 90°, 180°, 270°
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),  // 0°
            Customer::new(2, 0.0, 1.0, 10, 0.0),  // 90°
            Customer::new(3, -1.0, 0.0, 10, 0.0), // 180°
            Customer::new(4, 0.0, -1.0, 10, 0.0), // -90° (=270°)
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let sol = sweep(&customers, &dm, &vehicle);
        assert_eq!(sol.num_served(), 4);
        // All in one route, ordered by angle: 4(-90°), 1(0°), 2(90°), 3(180°)
        assert_eq!(sol.num_routes(), 1);
        let ids = sol.routes()[0].customer_ids();
        assert_eq!(ids, vec![4, 1, 2, 3]);
    }
}
