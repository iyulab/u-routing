//! Pair relocation: lift a pickup-delivery pair out of its route and put it back where the plan
//! costs least.
//!
//! The single-customer moves cannot move a pair at all (each half alone is not a route), so a
//! plan with pairs would otherwise stay as the construction made it. This move takes both
//! stops of a pair, tries every pair of positions in every route (the pickup first), and keeps
//! the cheapest placement that keeps the routes feasible: windows, limits, the pair's order and
//! the vehicle's capacity along the way.
//!
//! # Complexity
//!
//! O(P · R · L³) per pass for P pairs and R routes of at most L stops (each of the L² placements
//! in a route is evaluated in O(L)).

use super::gain_floor;
use crate::distance::DistanceMatrix;
use crate::evaluation::{effective_cost, route_peak_load, RouteEvaluator, RouteLimits};
use crate::models::{Customer, Solution, Vehicle};

/// The most passes over the pairs: each pass takes every pair that has an improving placement.
const MAX_PASSES: usize = 50;

/// Moves each pickup-delivery pair of `solution` to its cheapest placement, until no pair has an
/// improving one: the pair leaves its route and both stops are inserted -- the pickup first -- at
/// the positions of any route (its own included) that keep the route within the vehicle's
/// capacity, windows and limits. The cost compared is the routes' priced distance (distance plus
/// the cost of lateness at a soft window). The plan is returned as it is when it has no pairs.
///
/// # Examples
///
/// ```
/// use u_routing::constructive::nearest_neighbor;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::local_search::pair_relocate_improve;
/// use u_routing::models::{Customer, Vehicle};
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 5, 0.0).pickup_of(2),
///     Customer::new(2, 2.0, 0.0, 0, 0.0).delivery_of(1),
///     Customer::new(3, 1.0, 1.0, 5, 0.0).pickup_of(4),
///     Customer::new(4, 2.0, 1.0, 0, 0.0).delivery_of(3),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 5), Vehicle::new(1, 5)];
///
/// let before = nearest_neighbor(&customers, &dm, &vehicles);
/// let after = u_routing::local_search::pair_relocate_improve(&before, &customers, &dm, &vehicles[0]);
/// assert!(after.total_distance() <= before.total_distance() + 1e-10);
/// ```
pub fn pair_relocate_improve(
    solution: &Solution,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Solution {
    let pickups: Vec<usize> = (0..customers.len())
        .filter(|&c| customers[c].delivery_index().is_some())
        .collect();
    if pickups.is_empty() {
        return solution.clone();
    }

    let depot = vehicle.depot_id();
    let limits = RouteLimits::of(vehicle);
    let floor = gain_floor(distances);
    let cost = |route: &[usize]| effective_cost(route, depot, distances, customers, &limits);
    let mut routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
    let mut costs: Vec<f64> = routes.iter().map(|r| cost(r)).collect();

    for _ in 0..MAX_PASSES {
        let mut improved = false;
        for &pickup in &pickups {
            let delivery = customers[pickup]
                .delivery_index()
                .expect("a pickup has a delivery");
            let Some(from) = routes.iter().position(|r| r.contains(&pickup)) else {
                continue;
            };
            if !routes[from].contains(&delivery) {
                continue;
            }
            let without: Vec<usize> = routes[from]
                .iter()
                .copied()
                .filter(|&c| c != pickup && c != delivery)
                .collect();
            let without_cost = cost(&without);

            // The best placement over every route, the pair's own included.
            let mut best: Option<(f64, usize, Vec<usize>, f64)> = None;
            for to in 0..routes.len() {
                let base = if to == from { &without } else { &routes[to] };
                let (old_total, other) = if to == from {
                    (costs[from], 0.0)
                } else {
                    (costs[from] + costs[to], without_cost)
                };
                for i in 0..=base.len() {
                    for j in i..=base.len() {
                        let mut candidate = Vec::with_capacity(base.len() + 2);
                        candidate.extend_from_slice(&base[..i]);
                        candidate.push(pickup);
                        candidate.extend_from_slice(&base[i..j]);
                        candidate.push(delivery);
                        candidate.extend_from_slice(&base[j..]);
                        if route_peak_load(&candidate, customers) > vehicle.capacity() {
                            continue;
                        }
                        let placed = cost(&candidate);
                        // A placement the route does not allow costs infinity: no gain.
                        let gain = old_total - (other + placed);
                        if gain > floor && best.as_ref().is_none_or(|b| gain > b.0) {
                            best = Some((gain, to, candidate, placed));
                        }
                    }
                }
            }

            if let Some((_, to, candidate, placed)) = best {
                if to == from {
                    routes[from] = candidate;
                    costs[from] = placed;
                } else {
                    routes[from] = without;
                    costs[from] = without_cost;
                    routes[to] = candidate;
                    costs[to] = placed;
                }
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    let evaluator = RouteEvaluator::new(customers, distances, vehicle);
    let mut rebuilt = Solution::new();
    for route in routes.iter().filter(|r| !r.is_empty()) {
        rebuilt.add_route(evaluator.build_route(route).0);
    }
    for &u in solution.unassigned() {
        rebuilt.add_unassigned(u);
    }
    rebuilt.price(|_| vehicle);
    rebuilt
}
