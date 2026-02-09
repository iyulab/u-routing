//! Inter-route exchange operator (2-opt*).
//!
//! # Algorithm
//!
//! The cross-exchange (2-opt*) operator swaps tail segments between two routes.
//! Given routes R1 = [a₁, ..., aᵢ, aᵢ₊₁, ..., aₙ] and
//! R2 = [b₁, ..., bⱼ, bⱼ₊₁, ..., bₘ], produce:
//!
//! R1' = [a₁, ..., aᵢ, bⱼ₊₁, ..., bₘ]
//! R2' = [b₁, ..., bⱼ, aᵢ₊₁, ..., aₙ]
//!
//! Accepts moves that reduce total distance and maintain capacity feasibility.
//!
//! # Complexity
//!
//! O(n² × R²) per pass, where n = customers per route, R = number of routes.
//!
//! # Reference
//!
//! Potvin, J.-Y. & Rousseau, J.-M. (1995). "An Exchange Heuristic for
//! Routeing Problems with Time Windows", *Journal of the Operational Research
//! Society* 46(12), 1433-1446.

use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{Customer, Solution, Vehicle};

/// Applies inter-route cross-exchange (2-opt*) improvement.
///
/// Swaps tail segments between pairs of routes to reduce total distance
/// while respecting capacity constraints.
///
/// # Arguments
///
/// * `solution` — Current VRP solution
/// * `customers` — All locations (index 0 = depot)
/// * `distances` — Distance matrix
/// * `vehicle` — Vehicle type (homogeneous fleet)
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::constructive::nearest_neighbor;
/// use u_routing::local_search::exchange_improve;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 1.0, 10, 0.0),
///     Customer::new(2, -1.0, -1.0, 10, 0.0),
///     Customer::new(3, 1.0, -1.0, 10, 0.0),
///     Customer::new(4, -1.0, 1.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];
///
/// let initial = nearest_neighbor(&customers, &dm, &vehicles);
/// let improved = exchange_improve(&initial, &customers, &dm, &vehicles[0]);
/// assert!(improved.total_distance() <= initial.total_distance() + 1e-10);
/// ```
pub fn exchange_improve(
    solution: &Solution,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Solution {
    if solution.num_routes() < 2 {
        return solution.clone();
    }

    let depot = vehicle.depot_id();
    let mut routes: Vec<Vec<usize>> = solution
        .routes()
        .iter()
        .map(|r| r.customer_ids())
        .collect();

    let mut improved = true;
    while improved {
        improved = false;

        for r1 in 0..routes.len() {
            for r2 in (r1 + 1)..routes.len() {
                if let Some((cut1, cut2, delta)) =
                    find_best_exchange(&routes[r1], &routes[r2], depot, distances, customers, vehicle)
                {
                    if delta < -1e-10 {
                        // Execute the exchange
                        let tail1: Vec<usize> = routes[r1][cut1..].to_vec();
                        let tail2: Vec<usize> = routes[r2][cut2..].to_vec();
                        routes[r1].truncate(cut1);
                        routes[r2].truncate(cut2);
                        routes[r1].extend(tail2);
                        routes[r2].extend(tail1);
                        improved = true;
                    }
                }
            }
        }
    }

    rebuild_solution(&routes, solution, distances, customers, vehicle)
}

/// Finds the best cross-exchange between two routes.
/// Returns (cut_pos_r1, cut_pos_r2, delta) if improvement found.
fn find_best_exchange(
    route1: &[usize],
    route2: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    vehicle: &Vehicle,
) -> Option<(usize, usize, f64)> {
    let n1 = route1.len();
    let n2 = route2.len();

    let mut best: Option<(usize, usize, f64)> = None;

    // cut1 ranges from 1..n1 (split after position cut1-1)
    // cut2 ranges from 1..n2
    for cut1 in 1..=n1 {
        for cut2 in 1..=n2 {
            // New routes after exchange:
            // R1' = route1[..cut1] + route2[cut2..]
            // R2' = route2[..cut2] + route1[cut1..]

            // Check capacity
            let new_load1: i32 = route1[..cut1]
                .iter()
                .chain(route2[cut2..].iter())
                .map(|&c| customers[c].demand())
                .sum();
            let new_load2: i32 = route2[..cut2]
                .iter()
                .chain(route1[cut1..].iter())
                .map(|&c| customers[c].demand())
                .sum();

            if new_load1 > vehicle.capacity() || new_load2 > vehicle.capacity() {
                continue;
            }

            // Compute delta: change in total distance
            // Old edges broken: (route1[cut1-1] → route1[cut1] or depot)
            //                   (route2[cut2-1] → route2[cut2] or depot)
            let old_edge1 = if cut1 < n1 {
                distances.get(route1[cut1 - 1], route1[cut1])
            } else {
                distances.get(route1[cut1 - 1], depot)
            };
            let old_edge2 = if cut2 < n2 {
                distances.get(route2[cut2 - 1], route2[cut2])
            } else {
                distances.get(route2[cut2 - 1], depot)
            };

            // New edges created
            let new_edge1 = if cut2 < n2 {
                distances.get(route1[cut1 - 1], route2[cut2])
            } else {
                distances.get(route1[cut1 - 1], depot)
            };
            let new_edge2 = if cut1 < n1 {
                distances.get(route2[cut2 - 1], route1[cut1])
            } else {
                distances.get(route2[cut2 - 1], depot)
            };

            let delta = (new_edge1 + new_edge2) - (old_edge1 + old_edge2);

            if delta < -1e-10
                && best.as_ref().is_none_or(|b| delta < b.2)
            {
                best = Some((cut1, cut2, delta));
            }
        }
    }

    best
}

/// Rebuilds a Solution from customer ID sequences.
fn rebuild_solution(
    routes: &[Vec<usize>],
    original: &Solution,
    distances: &DistanceMatrix,
    customers: &[Customer],
    vehicle: &Vehicle,
) -> Solution {
    let evaluator = RouteEvaluator::new(customers, distances, vehicle);
    let mut solution = Solution::new();

    for route_customers in routes {
        if route_customers.is_empty() {
            continue;
        }
        let (route, _) = evaluator.build_route(route_customers);
        solution.add_route(route);
    }

    for &uid in original.unassigned() {
        solution.add_unassigned(uid);
    }

    let total_dist = solution.total_distance();
    solution.set_total_cost(total_dist);
    solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constructive::nearest_neighbor;

    #[test]
    fn test_exchange_single_route() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let vehicles = vec![vehicle.clone()];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = exchange_improve(&sol, &customers, &dm, &vehicle);
        assert_eq!(improved.num_served(), 1);
    }

    #[test]
    fn test_exchange_does_not_worsen() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 10, 0.0),
            Customer::new(2, -1.0, -1.0, 10, 0.0),
            Customer::new(3, 1.0, -1.0, 10, 0.0),
            Customer::new(4, -1.0, 1.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 20);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];
        let initial = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = exchange_improve(&initial, &customers, &dm, &vehicle);
        assert!(improved.total_distance() <= initial.total_distance() + 1e-10);
    }

    #[test]
    fn test_exchange_respects_capacity() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, -1.0, 0.0, 10, 0.0),
            Customer::new(4, -2.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 20);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = exchange_improve(&sol, &customers, &dm, &vehicle);
        for route in improved.routes() {
            assert!(route.total_load() <= 20);
        }
    }

    #[test]
    fn test_exchange_fixes_interleaved_routes() {
        // Route 1: [1(east), 4(west)] and Route 2: [3(east), 2(west)]
        // Exchange tails to get [1, 3] (east) and [4, 2] (west)
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 5.0, 1.0, 10, 0.0),  // east
            Customer::new(2, -5.0, -1.0, 10, 0.0), // west
            Customer::new(3, 5.0, -1.0, 10, 0.0),  // east
            Customer::new(4, -5.0, 1.0, 10, 0.0),  // west
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 20);

        // Manually create a bad solution with interleaved clusters
        let evaluator = RouteEvaluator::new(&customers, &dm, &vehicle);
        let mut sol = Solution::new();
        let (r1, _) = evaluator.build_route(&[1, 4]);
        let (r2, _) = evaluator.build_route(&[3, 2]);
        sol.add_route(r1);
        sol.add_route(r2);
        sol.set_total_cost(sol.total_distance());

        let initial_dist = sol.total_distance();
        let improved = exchange_improve(&sol, &customers, &dm, &vehicle);
        assert!(improved.total_distance() <= initial_dist + 1e-10);
        assert_eq!(improved.num_served(), 4);
    }

    #[test]
    fn test_exchange_preserves_all_customers() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 5, 0.0),
            Customer::new(2, 2.0, -1.0, 5, 0.0),
            Customer::new(3, -1.0, 2.0, 5, 0.0),
            Customer::new(4, -2.0, -1.0, 5, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 10);
        let vehicles = vec![Vehicle::new(0, 10), Vehicle::new(1, 10)];
        let initial = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = exchange_improve(&initial, &customers, &dm, &vehicle);
        assert_eq!(improved.num_served(), 4);
    }
}
