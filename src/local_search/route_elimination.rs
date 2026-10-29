//! Route elimination: empty a route by inserting its customers into the others.
//!
//! # Algorithm
//!
//! A route is removed and each of its customers, the one with the largest
//! demand first, goes to the cheapest feasible position in the remaining
//! routes (capacity, time windows and the route limits kept). When all of
//! them find a place and the distance they add is less than the route's own
//! distance plus its fixed cost, the routes without it replace the old ones.
//! The shortest routes are tried first, and the search starts over after each
//! elimination until no route can go.
//!
//! Relocating one customer at a time cannot do this: the moves that empty a
//! route each lengthen the others before the last one saves the fixed cost,
//! so a descent never takes the first of them. Removing whole routes and
//! reinserting their customers is the route-minimisation step of the
//! vehicle-routing heuristics that rank fewer vehicles first (Bräysy &
//! Gendreau 2005 survey it; Nagata & Bräysy 2009 extend it with an ejection
//! pool).
//!
//! # Complexity
//!
//! O(R × k × n × L) per attempt, R routes, k customers of the route tried,
//! n positions and L the length of a feasibility check.
//!
//! # Reference
//!
//! Bräysy, O. & Gendreau, M. (2005). "Vehicle Routing Problem with Time
//! Windows, Part II: Metaheuristics", *Transportation Science* 39(1), 119-139.

use super::gain_floor;
use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, route_feasible, RouteLimits};
use crate::models::{Customer, Vehicle};

use super::route_distance;

/// Removes every route of `routes` whose customers the others can take for
/// less than the route costs (its distance plus `vehicle`'s fixed cost), and
/// returns what is left. Every route returned keeps capacity, time windows and
/// the route limits when the routes given do.
pub(crate) fn eliminate_routes(
    mut routes: Vec<Vec<usize>>,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
) -> Vec<Vec<usize>> {
    let depot = vehicle.depot_id();
    let limits = RouteLimits::of(vehicle);
    let constrained = route_checks_needed(customers, &limits);
    let fixed = vehicle.fixed_cost_in_distance();
    let load = |route: &[usize]| route.iter().map(|&c| customers[c].demand()).sum::<i32>();

    'search: loop {
        routes.retain(|r| !r.is_empty());
        let mut order: Vec<usize> = (0..routes.len()).collect();
        order.sort_by_key(|&r| routes[r].len());
        for r in order {
            let saved = route_distance(&routes[r], depot, distances) + fixed;
            let mut rest: Vec<Vec<usize>> = routes
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != r)
                .map(|(_, route)| route.clone())
                .collect();
            let mut loads: Vec<i32> = rest.iter().map(|route| load(route)).collect();
            let mut moving = routes[r].clone();
            moving.sort_by_key(|&c| std::cmp::Reverse(customers[c].demand()));
            let mut added = 0.0;
            let mut placed_all = true;
            for c in moving {
                // Positions cheapest first, so the first feasible one is the
                // cheapest (the first in route order among equals), and the
                // search stops where no cheaper position could still pay.
                let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
                for (k, route) in rest.iter().enumerate() {
                    if loads[k] + customers[c].demand() > vehicle.capacity() {
                        continue;
                    }
                    for pos in 0..=route.len() {
                        let prev = if pos == 0 { depot } else { route[pos - 1] };
                        let next = route.get(pos).copied().unwrap_or(depot);
                        let cost = distances.get(prev, c) + distances.get(c, next)
                            - distances.get(prev, next);
                        candidates.push((cost, k, pos));
                    }
                }
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut best: Option<(f64, usize, usize)> = None;
                for (cost, k, pos) in candidates {
                    if added + cost >= saved {
                        break;
                    }
                    if constrained {
                        let mut trial = rest[k].clone();
                        trial.insert(pos, c);
                        if !route_feasible(&trial, depot, distances, customers, &limits) {
                            continue;
                        }
                    }
                    best = Some((cost, k, pos));
                    break;
                }
                match best {
                    Some((cost, k, pos)) => {
                        rest[k].insert(pos, c);
                        loads[k] += customers[c].demand();
                        added += cost;
                    }
                    None => {
                        placed_all = false;
                        break;
                    }
                }
                if added >= saved {
                    placed_all = false;
                    break;
                }
            }
            if placed_all && added < saved - gain_floor(distances) {
                routes = rest;
                continue 'search;
            }
        }
        return routes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    fn line(n: usize) -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for i in 1..=n {
            customers.push(Customer::new(i, i as f64, 0.0, 1, 0.0));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn a_route_goes_when_its_fixed_cost_outweighs_the_detour() {
        // Customers 1..4 on a line; [1, 2] and [3, 4] as two routes. Merged,
        // the distance falls from 4 + 8 to 8 anyway.
        let (customers, dm) = line(4);
        let vehicle = Vehicle::new(0, 10).with_fixed_cost(100.0);
        let routes = eliminate_routes(vec![vec![1, 2], vec![3, 4]], &customers, &dm, &vehicle);
        assert_eq!(routes.len(), 1);
        let mut served: Vec<usize> = routes.concat();
        served.sort_unstable();
        assert_eq!(served, vec![1, 2, 3, 4]);
    }

    #[test]
    fn capacity_and_time_windows_keep_a_route() {
        let (customers, dm) = line(4);
        // Capacity 2: neither route can take another customer.
        let tight = Vehicle::new(0, 2).with_fixed_cost(100.0);
        let routes = eliminate_routes(vec![vec![1, 2], vec![3, 4]], &customers, &dm, &tight);
        assert_eq!(routes.len(), 2);

        // Customer 1 must be reached by time 1 and takes 10 to serve,
        // customer 4 by time 4: no single route visits both in time.
        let mut windowed = customers.clone();
        windowed[1] = Customer::new(1, 1.0, 0.0, 1, 10.0)
            .with_time_window(TimeWindow::new(0.0, 1.0).expect("valid window"));
        windowed[4] = Customer::new(4, 4.0, 0.0, 1, 0.0)
            .with_time_window(TimeWindow::new(0.0, 4.0).expect("valid window"));
        let roomy = Vehicle::new(0, 10).with_fixed_cost(100.0);
        let routes = eliminate_routes(vec![vec![1], vec![4]], &windowed, &dm, &roomy);
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn without_a_fixed_cost_a_route_goes_only_when_distance_falls() {
        // Two customers on opposite sides: one route each is shorter.
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        customers.push(Customer::new(1, -5.0, 0.0, 1, 0.0));
        customers.push(Customer::new(2, 5.0, 0.0, 1, 0.0));
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 10);
        let routes = eliminate_routes(vec![vec![1], vec![2]], &customers, &dm, &vehicle);
        assert_eq!(routes.len(), 2);
    }
}
