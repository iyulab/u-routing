//! Inter-route customer relocation operator.
//!
//! # Algorithm
//!
//! Tries moving each customer from its current route to the best insertion
//! position in another route. Accepts moves that reduce total distance and
//! maintain capacity feasibility.
//!
//! # Complexity
//!
//! O(n² × R) per pass where n = customers per route, R = number of routes.
//!
//! # Reference
//!
//! Or, I. (1976). "Traveling Salesman-Type Combinatorial Problems and Their
//! Relation to the Logistics of Blood Banking". PhD thesis.

use crate::distance::DistanceMatrix;
use crate::models::{Customer, Solution, Vehicle};

/// A relocate move: move customer from one route to another.
#[derive(Debug, Clone)]
struct RelocateMove {
    from_route: usize,
    from_pos: usize,
    to_route: usize,
    to_pos: usize,
    delta: f64,
}

/// Applies inter-route relocate improvement to a solution.
///
/// Iteratively moves customers between routes to reduce total distance,
/// while respecting vehicle capacity constraints.
///
/// # Arguments
///
/// * `solution` — Current solution (modified in-place style, returns new)
/// * `customers` — All locations
/// * `distances` — Distance matrix
/// * `vehicle` — Vehicle type (homogeneous fleet)
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::constructive::nearest_neighbor;
/// use u_routing::local_search::relocate_improve;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 0.0, 3.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];
///
/// let initial = nearest_neighbor(&customers, &dm, &vehicles);
/// let improved = relocate_improve(&initial, &customers, &dm, &vehicles[0]);
/// assert!(improved.total_distance() <= initial.total_distance() + 1e-10);
/// ```
pub fn relocate_improve(
    solution: &Solution,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Solution {
    if solution.num_routes() < 2 {
        return solution.clone();
    }

    // Extract route customer sequences
    let mut routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();

    let mut improved = true;
    while improved {
        improved = false;
        let best_move = find_best_relocate(&routes, customers, distances, vehicle);

        if let Some(mv) = best_move {
            if mv.delta < -1e-10 {
                let customer_id = routes[mv.from_route][mv.from_pos];
                routes[mv.from_route].remove(mv.from_pos);
                routes[mv.to_route].insert(mv.to_pos, customer_id);
                improved = true;
            }
        }
    }

    // Rebuild solution
    rebuild_solution(&routes, solution, distances, customers, vehicle)
}

/// Finds the best single relocate move across all route pairs.
fn find_best_relocate(
    routes: &[Vec<usize>],
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Option<RelocateMove> {
    let depot = vehicle.depot_id();
    let mut best: Option<RelocateMove> = None;

    for from_r in 0..routes.len() {
        for from_pos in 0..routes[from_r].len() {
            let cid = routes[from_r][from_pos];
            let removal_delta = removal_cost(&routes[from_r], from_pos, depot, distances);

            for (to_r, to_route) in routes.iter().enumerate() {
                if to_r == from_r {
                    continue;
                }

                // Check capacity
                let to_load: i32 = to_route.iter().map(|&c| customers[c].demand()).sum();
                if to_load + customers[cid].demand() > vehicle.capacity() {
                    continue;
                }

                // Try all insertion positions
                for to_pos in 0..=to_route.len() {
                    let insertion_delta = insertion_cost(to_route, to_pos, cid, depot, distances);
                    let delta = removal_delta + insertion_delta;

                    if delta < -1e-10 {
                        let is_better = best.as_ref().is_none_or(|b| delta < b.delta);
                        if is_better {
                            best = Some(RelocateMove {
                                from_route: from_r,
                                from_pos,
                                to_route: to_r,
                                to_pos,
                                delta,
                            });
                        }
                    }
                }
            }
        }
    }

    best
}

/// Cost of removing customer at `pos` from route.
fn removal_cost(route: &[usize], pos: usize, depot: usize, distances: &DistanceMatrix) -> f64 {
    let prev = if pos == 0 { depot } else { route[pos - 1] };
    let next = if pos == route.len() - 1 {
        depot
    } else {
        route[pos + 1]
    };
    let cid = route[pos];

    // Old: prev → cid → next
    // New: prev → next
    distances.get(prev, next) - distances.get(prev, cid) - distances.get(cid, next)
}

/// Cost of inserting `customer_id` at `pos` in route.
fn insertion_cost(
    route: &[usize],
    pos: usize,
    customer_id: usize,
    depot: usize,
    distances: &DistanceMatrix,
) -> f64 {
    let prev = if pos == 0 { depot } else { route[pos - 1] };
    let next = if pos == route.len() {
        depot
    } else {
        route[pos]
    };

    // Old: prev → next
    // New: prev → customer_id → next
    distances.get(prev, customer_id) + distances.get(customer_id, next) - distances.get(prev, next)
}

/// Rebuilds a Solution from customer ID sequences.
fn rebuild_solution(
    routes: &[Vec<usize>],
    original: &Solution,
    distances: &DistanceMatrix,
    customers: &[Customer],
    vehicle: &Vehicle,
) -> Solution {
    use crate::evaluation::RouteEvaluator;

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
    fn test_relocate_single_route() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let vehicles = vec![vehicle.clone()];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = relocate_improve(&sol, &customers, &dm, &vehicle);
        assert_eq!(improved.num_served(), 1);
    }

    #[test]
    fn test_relocate_no_improvement_needed() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 100);
        let vehicles = vec![vehicle.clone()];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = relocate_improve(&sol, &customers, &dm, &vehicle);
        assert!((improved.total_distance() - sol.total_distance()).abs() < 1e-10);
    }

    #[test]
    fn test_relocate_improves_bad_split() {
        // Customer 2 is closer to customer 3 than to customer 1
        // Force bad split by capacity
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 10.0, 0.0, 10, 0.0),
            Customer::new(2, 5.0, 5.0, 5, 0.0), // between
            Customer::new(3, 0.0, 10.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 20);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];

        let initial = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = relocate_improve(&initial, &customers, &dm, &vehicle);
        assert!(improved.total_distance() <= initial.total_distance() + 1e-10);
        assert_eq!(improved.num_served(), 3);
    }

    #[test]
    fn test_relocate_respects_capacity() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 15);
        let vehicles = vec![
            Vehicle::new(0, 15),
            Vehicle::new(1, 15),
            Vehicle::new(2, 15),
        ];
        let sol = nearest_neighbor(&customers, &dm, &vehicles);
        let improved = relocate_improve(&sol, &customers, &dm, &vehicle);
        // Each route should have at most capacity 15 (1 customer each)
        for route in improved.routes() {
            assert!(route.total_load() <= 15);
        }
    }

    #[test]
    fn test_removal_cost() {
        let route = vec![1, 2, 3];
        let dm = DistanceMatrix::from_data(
            4,
            vec![
                0.0, 5.0, 8.0, 12.0, 5.0, 0.0, 3.0, 7.0, 8.0, 3.0, 0.0, 4.0, 12.0, 7.0, 4.0, 0.0,
            ],
        )
        .expect("valid");

        // Removing customer 2 (pos=1): was 1→2→3, becomes 1→3
        let cost = removal_cost(&route, 1, 0, &dm);
        // Old: d(1,2) + d(2,3) = 3 + 4 = 7
        // New: d(1,3) = 7
        // Delta: 7 - 7 = 0
        assert!((cost - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_insertion_cost() {
        let route = vec![1, 3];
        let dm = DistanceMatrix::from_data(
            4,
            vec![
                0.0, 5.0, 8.0, 12.0, 5.0, 0.0, 3.0, 7.0, 8.0, 3.0, 0.0, 4.0, 12.0, 7.0, 4.0, 0.0,
            ],
        )
        .expect("valid");

        // Inserting customer 2 at pos=1: route becomes [1, 2, 3]
        let cost = insertion_cost(&route, 1, 2, 0, &dm);
        // Old: d(1,3) = 7
        // New: d(1,2) + d(2,3) = 3 + 4 = 7
        // Delta: 7 - 7 = 0
        assert!((cost - 0.0).abs() < 1e-10);
    }
}
