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
use crate::evaluation::{
    close_pairs, closes_feasibly, has_pairs, load_change, route_feasible, RouteEvaluator,
    RouteLimits,
};
use crate::models::{Customer, Route, Solution, Vehicle};

/// The vehicle of `route` in `vehicles` (by id; the first when the id is not
/// in the list).
pub(crate) fn fleet_vehicle<'v>(vehicles: &'v [Vehicle], route: &Route) -> &'v Vehicle {
    vehicles
        .iter()
        .find(|v| v.id() == route.vehicle_id())
        .unwrap_or(&vehicles[0])
}

/// Constructs a VRP solution using the nearest-neighbor heuristic.
///
/// Starting from the depot, greedily visits the nearest unvisited customer.
/// Opens a new route when adding the next customer would violate capacity.
///
/// # Arguments
///
/// * `customers` — All locations (index 0 = depot)
/// * `distances` — Distance matrix
/// * `vehicles` — Available vehicles, used in order, each with its own capacity;
///   customers left when they run out are reported as unassigned
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

    let paired = has_pairs(customers);
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
        let limits = RouteLimits::of(vehicle);
        let mut current = depot;
        let mut route_customers = Vec::new();
        let mut current_load: i32 = 0;
        // Travelled so far, and the clock after serving `current`.
        let mut travelled = 0.0;
        let mut clock = 0.0;

        loop {
            // Find the nearest unvisited customer that fits capacity and still
            // lets the route get back to the depot within the vehicle's limits.
            let mut best: Option<(usize, f64)> = None;
            for i in 1..n {
                if visited[i] {
                    continue;
                }
                let demand = load_change(customers, i);
                if current_load + demand > vehicle.capacity() {
                    continue;
                }
                // Whatever the route owes at its end (a pickup's delivery) has to fit too.
                if paired
                    && !closes_feasibly(
                        &mut route_customers,
                        i,
                        depot,
                        distances,
                        customers,
                        &limits,
                    )
                {
                    continue;
                }
                let d = distances.get(current, i);
                let back = distances.get(i, depot);
                if limits.rest.is_some() {
                    // A driver's break is placed on the whole route: ask the route's clock.
                    route_customers.push(i);
                    let fits =
                        route_feasible(&route_customers, depot, distances, customers, &limits);
                    route_customers.pop();
                    if !fits {
                        continue;
                    }
                } else if !limits.allow(
                    travelled + d + back,
                    clock
                        + distances.travel_time(current, i)
                        + customers[i].service_duration()
                        + distances.travel_time(i, depot),
                ) {
                    continue;
                }
                if best.is_none_or(|(_, best_d)| d < best_d) {
                    best = Some((i, d));
                }
            }

            match best {
                Some((next, d)) => {
                    visited[next] = true;
                    route_customers.push(next);
                    current_load += load_change(customers, next);
                    travelled += d;
                    clock +=
                        distances.travel_time(current, next) + customers[next].service_duration();
                    current = next;
                }
                None => break,
            }
        }

        // The route ends with the deliveries it still owes (each was checked to fit).
        if paired {
            let before = route_customers.len();
            close_pairs(&mut route_customers, customers);
            for &delivery in &route_customers[before..] {
                visited[delivery] = true;
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

    solution.price(|r| fleet_vehicle(vehicles, r));

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

    /// A driver's break is time on the route: with `max_duration` the stops it leaves no room for go
    /// on another route.
    #[test]
    fn the_drivers_break_counts_against_the_duration_limit() {
        use crate::evaluation::route_feasible;
        use crate::models::VehicleBreak;
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for i in 1..=4 {
            customers.push(Customer::new(i, 5.0 * i as f64, 0.0, 1, 1.0));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        // Out and back to the fourth stop is 40 and four services 4: 44 without the break.
        let free = Vehicle::new(0, 10).with_max_duration(67.0);
        let with_break = free.clone().with_break(VehicleBreak::new(2.0, 30.0, 25.0));
        let vehicles = |v: &Vehicle| vec![v.clone(), v.clone(), v.clone(), v.clone()];
        let sol = nearest_neighbor(&customers, &dm, &vehicles(&free));
        assert_eq!(sol.routes().len(), 1, "one route of 44 fits in 67");
        let sol = nearest_neighbor(&customers, &dm, &vehicles(&with_break));
        assert!(
            sol.routes().len() > 1,
            "44 + 25 does not fit: {:?}",
            sol.routes()
                .iter()
                .map(|r| r.customer_ids())
                .collect::<Vec<_>>()
        );
        let limits = RouteLimits::of(&with_break);
        for route in sol.routes() {
            assert!(
                route_feasible(&route.customer_ids(), 0, &dm, &customers, &limits),
                "{:?}",
                route.customer_ids()
            );
        }
        assert_eq!(sol.num_served(), 4);
    }
}
