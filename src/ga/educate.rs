//! Local search on a giant tour -- the improvement step of a hybrid GA.
//!
//! A permutation GA with only crossover and random mutation does not reach
//! good routing solutions: the split of a random tour opens many short
//! routes, and recombination alone rarely closes them. Prins (2004) replaces
//! mutation with local search on the split solution and writes the improved
//! routes back into the chromosome (a Lamarckian step), so the population
//! carries what the search learned.
//!
//! [`educate`] splits the tour, improves the routes with the inter-route moves
//! (relocate, then 2-opt* tail exchange) and the intra-route ones (2-opt,
//! or-opt), and concatenates them back. Every move keeps capacity, time
//! windows and the route limits, so the routes stay feasible; splitting the
//! returned tour again costs at most what the improved routes cost, since
//! their cuts are among the ones the split considers.
//!
//! # Reference
//!
//! Prins, C. (2004). "A simple and effective evolutionary algorithm for the
//! vehicle routing problem", *Computers & Operations Research* 31(12), 1985-2002.

use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, RouteEvaluator, RouteLimits};
use crate::local_search::{exchange_improve, or_opt_improve, relocate_improve, two_opt_improve};
use crate::models::{Customer, Solution, Vehicle};

use super::split::split;
use super::split_tw::split_tw;

/// The vehicle every route of the GA is planned with: one capacity and one
/// set of limits.
pub(crate) fn route_vehicle(capacity: i32, limits: &RouteLimits) -> Vehicle {
    let mut vehicle = Vehicle::new(0, capacity);
    if let Some(max) = limits.max_distance {
        vehicle = vehicle.with_max_distance(max);
    }
    if let Some(max) = limits.max_duration {
        vehicle = vehicle.with_max_duration(max);
    }
    vehicle
}

/// Routes as one giant tour: the routes in order, then any customer none of
/// them serves, so the tour stays a permutation of every customer.
pub(crate) fn concatenate(routes: &[Vec<usize>], num_customers: usize) -> Vec<usize> {
    let mut tour: Vec<usize> = routes.iter().flatten().copied().collect();
    let mut served = vec![false; num_customers + 1];
    for &c in &tour {
        served[c] = true;
    }
    tour.extend((1..=num_customers).filter(|&c| !served[c]));
    tour
}

/// Improves the solution `tour` splits into, and returns it as a giant tour.
pub(crate) fn educate(
    tour: &[usize],
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Vec<usize> {
    let limits = RouteLimits::of(vehicle);
    let fixed = vehicle.fixed_cost_in_distance();
    let routes = if route_checks_needed(customers, &limits) {
        split_tw(
            tour,
            customers,
            distances,
            vehicle.capacity(),
            &limits,
            fixed,
        )
        .routes
    } else {
        split(tour, customers, distances, vehicle.capacity(), fixed).routes
    };

    let evaluator = RouteEvaluator::new(customers, distances, vehicle);
    let mut solution = Solution::new();
    for route in &routes {
        solution.add_route(evaluator.build_route(route).0);
    }
    let solution = relocate_improve(&solution, customers, distances, vehicle);
    let solution = exchange_improve(&solution, customers, distances, vehicle);

    let depot = vehicle.depot_id();
    let routes: Vec<Vec<usize>> = solution
        .routes()
        .iter()
        .map(|r| {
            let (route, _) =
                two_opt_improve(&r.customer_ids(), depot, distances, customers, &limits);
            or_opt_improve(&route, depot, distances, customers, &limits).0
        })
        .filter(|r| !r.is_empty())
        .collect();
    concatenate(&routes, customers.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    #[test]
    fn concatenate_keeps_every_customer() {
        assert_eq!(concatenate(&[vec![3, 1], vec![4]], 5), vec![3, 1, 4, 2, 5]);
    }

    #[test]
    fn educate_returns_a_permutation_no_worse_than_its_input() {
        // Two clusters, interleaved in the tour: the split of the input opens
        // a route per customer pair, education regroups them.
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for i in 1..=8 {
            let x = if i % 2 == 0 { 10.0 } else { -10.0 };
            customers.push(
                Customer::new(i, x, i as f64, 1, 0.0)
                    .with_time_window(TimeWindow::new(0.0, 1000.0).expect("valid window")),
            );
        }
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = route_vehicle(4, &RouteLimits::NONE);
        let tour: Vec<usize> = (1..=8).collect();
        let limits = RouteLimits::NONE;
        let before = split_tw(&tour, &customers, &dm, 4, &limits, 0.0).total_distance;

        let educated = educate(&tour, &customers, &dm, &vehicle);
        let mut sorted = educated.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (1..=8).collect::<Vec<_>>());
        let after = split_tw(&educated, &customers, &dm, 4, &limits, 0.0).total_distance;
        assert!(after < before - 1e-9, "{after} !< {before}");
    }
}
