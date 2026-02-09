//! Nearest-neighbor constructive heuristic.
//!
//! Builds routes greedily: starting from the depot, always visit the nearest
//! unvisited customer. When capacity is exhausted, start a new route.
//!
//! # Complexity
//!
//! O(n²) where n = number of customers.
//!
//! # Reference
//!
//! This is the simplest constructive heuristic for VRP. While solution
//! quality is typically 15-25% above optimal, it provides a fast baseline.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{Customer, Solution, Vehicle};

/// Constructs a VRP solution using the nearest-neighbor heuristic.
///
/// Starting from the depot, greedily visits the nearest unvisited customer.
/// Opens a new route when adding the next customer would violate capacity.
///
/// # Arguments
///
/// * `customers` — All locations (index 0 = depot)
/// * `distances` — Distance matrix
/// * `vehicles` — Available vehicles (homogeneous fleet assumed)
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::constructive::nearest_neighbor;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 3.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 30)];
///
/// let solution = nearest_neighbor(&customers, &dm, &vehicles);
/// assert_eq!(solution.num_served(), 3);
/// assert!(solution.num_unassigned() == 0);
/// ```
pub fn nearest_neighbor(
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
            // No more vehicles — mark remaining as unassigned
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
        let mut route_customers = Vec::new();
        let mut current_load: i32 = 0;

        loop {
            // Find nearest unvisited customer that fits capacity
            let mut best: Option<(usize, f64)> = None;
            for i in 1..n {
                if visited[i] {
                    continue;
                }
                let demand = customers[i].demand();
                if current_load + demand > vehicle.capacity() {
                    continue;
                }
                let d = distances.get(current, i);
                if best.is_none() || d < best.expect("checked is_none").1 {
                    best = Some((i, d));
                }
            }

            match best {
                Some((next, _)) => {
                    visited[next] = true;
                    route_customers.push(next);
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

        // Check if all customers are visited
        if visited.iter().skip(1).all(|&v| v) {
            break;
        }
    }

    // Compute total cost
    let total_dist = solution.total_distance();
    solution.set_total_cost(total_dist);

    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_customers() -> (Vec<Customer>, DistanceMatrix, Vec<Vehicle>) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        (customers, dm, vehicles)
    }

    #[test]
    fn test_nn_all_on_one_route() {
        let (customers, dm, vehicles) = line_customers();
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        assert_eq!(sol.num_routes(), 1);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 0);
        // Should visit in order 1, 2, 3 (nearest each time)
        assert_eq!(sol.routes()[0].customer_ids(), vec![1, 2, 3]);
        // Distance: 0→1 + 1→2 + 2→3 + 3→0 = 1 + 1 + 1 + 3 = 6
        assert!((sol.routes()[0].total_distance() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_nn_split_routes() {
        let (customers, dm, _) = line_customers();
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        // Capacity 20: first route takes customers 1, 2 (20), second takes 3 (10)
        assert_eq!(sol.num_routes(), 2);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 0);
    }

    #[test]
    fn test_nn_insufficient_vehicles() {
        let (customers, dm, _) = line_customers();
        let vehicles = vec![Vehicle::new(0, 15)]; // Only fits 1 customer
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        assert_eq!(sol.num_routes(), 1);
        assert!(sol.num_unassigned() > 0);
    }

    #[test]
    fn test_nn_empty() {
        let customers = vec![Customer::depot(0.0, 0.0)];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        assert_eq!(sol.num_routes(), 0);
        assert_eq!(sol.num_served(), 0);
    }

    #[test]
    fn test_nn_chooses_nearest() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 10.0, 0.0, 5, 0.0), // far
            Customer::new(2, 1.0, 0.0, 5, 0.0),  // near
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        // Should pick customer 2 first (distance 1) then 1 (distance 9)
        assert_eq!(sol.routes()[0].customer_ids(), vec![2, 1]);
    }

    #[test]
    fn test_nn_total_cost() {
        let (customers, dm, vehicles) = line_customers();
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        assert!((sol.total_cost() - sol.total_distance()).abs() < 1e-10);
    }
}
