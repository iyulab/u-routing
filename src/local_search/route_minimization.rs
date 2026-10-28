//! Route minimisation with an ejection pool.
//!
//! # Algorithm
//!
//! A route is removed and its customers go to an *ejection pool*. One at a
//! time, the last customer put in the pool is inserted at the cheapest
//! feasible position of the remaining routes (capacity, time windows and the
//! route limits kept). When there is none, it goes in anyway at the position
//! where ejecting one customer -- or, failing that, two -- of the same route
//! makes the route feasible again, and the ejected customers join the pool.
//! Each customer carries a counter of how often it could not be inserted; the
//! ejection chosen is the one whose ejected customers have the smallest sum of
//! counters (ties: the cheaper insertion), so the customers that are hard to
//! place stay in and the easy ones move. The route is gone when the pool
//! empties within the iteration budget; then the next route is tried, the
//! shortest first, until none can go.
//!
//! Inserting the customers of a removed route one by one (the route
//! elimination of the GA's local search) fails as soon as one of them fits
//! nowhere, which on tight time windows is the usual case for the last route
//! a solution could save: the ejection makes room for it.
//!
//! This is the ejection-pool step of Nagata & Bräysy (2009) without its
//! squeeze (infeasible insertion repaired by local search) and perturbation
//! phases, and with at most two customers ejected at a time.
//!
//! # Complexity
//!
//! O(I × R × L³) per route tried, I the iteration budget, R routes and L the
//! length of a route (pairs ejected × positions × a feasibility check).
//!
//! # Reference
//!
//! Nagata, Y. & Bräysy, O. (2009). "A powerful route minimization heuristic
//! for the vehicle routing problem with time windows", *Operations Research
//! Letters* 37(5), 333-338.

use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, route_feasible, RouteLimits};
use crate::models::{Customer, Vehicle};

use super::route_distance;

/// Removes routes of `routes` while the ejection pool can place their
/// customers in the others within `max_iterations` insertions per route, and
/// removing them lowers the cost (distance plus `vehicle`'s fixed cost per
/// route). Every route returned keeps capacity, time windows and the route
/// limits when the routes given do, and serves the same customers.
pub(crate) fn minimize_routes(
    mut routes: Vec<Vec<usize>>,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
    max_iterations: usize,
) -> Vec<Vec<usize>> {
    let depot = vehicle.depot_id();
    let fixed = vehicle.fixed_cost_in_distance();
    let cost = |routes: &[Vec<usize>]| {
        routes
            .iter()
            .map(|r| route_distance(r, depot, distances) + fixed)
            .sum::<f64>()
    };
    let pool = Pool {
        customers,
        distances,
        vehicle,
        limits: RouteLimits::of(vehicle),
        constrained: route_checks_needed(customers, &RouteLimits::of(vehicle)),
    };

    'search: loop {
        routes.retain(|r| !r.is_empty());
        let before = cost(&routes);
        let mut order: Vec<usize> = (0..routes.len()).collect();
        order.sort_by_key(|&r| routes[r].len());
        for r in order {
            let rest: Vec<Vec<usize>> = routes
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != r)
                .map(|(_, route)| route.clone())
                .collect();
            if let Some(rest) = pool.empty_into(rest, &routes[r], max_iterations) {
                if cost(&rest) < before - 1e-9 {
                    routes = rest;
                    continue 'search;
                }
            }
        }
        return routes;
    }
}

/// An insertion that ejects customers: the route it changes, that route
/// afterwards, the customers ejected, and its rank (failure counters of the
/// ejected, then insertion cost).
struct Ejection {
    key: (u32, f64),
    route: usize,
    into: Vec<usize>,
    ejected: Vec<usize>,
}

struct Pool<'a> {
    customers: &'a [Customer],
    distances: &'a DistanceMatrix,
    vehicle: &'a Vehicle,
    limits: RouteLimits,
    constrained: bool,
}

impl Pool<'_> {
    fn load(&self, route: &[usize]) -> i32 {
        route.iter().map(|&c| self.customers[c].demand()).sum()
    }

    fn feasible(&self, route: &[usize]) -> bool {
        self.load(route) <= self.vehicle.capacity()
            && (!self.constrained
                || route_feasible(
                    route,
                    self.vehicle.depot_id(),
                    self.distances,
                    self.customers,
                    &self.limits,
                ))
    }

    fn insertion_cost(&self, route: &[usize], pos: usize, c: usize) -> f64 {
        let depot = self.vehicle.depot_id();
        let prev = if pos == 0 { depot } else { route[pos - 1] };
        let next = route.get(pos).copied().unwrap_or(depot);
        self.distances.get(prev, c) + self.distances.get(c, next) - self.distances.get(prev, next)
    }

    /// The cheapest feasible `(route, position)` for `c`.
    fn cheapest(&self, routes: &[Vec<usize>], c: usize) -> Option<(usize, usize)> {
        let mut best: Option<(f64, usize, usize)> = None;
        for (k, route) in routes.iter().enumerate() {
            if self.load(route) + self.customers[c].demand() > self.vehicle.capacity() {
                continue;
            }
            for pos in 0..=route.len() {
                let cost = self.insertion_cost(route, pos, c);
                if best.is_some_and(|(b, _, _)| cost >= b) {
                    continue;
                }
                let mut trial = route.clone();
                trial.insert(pos, c);
                if self.feasible(&trial) {
                    best = Some((cost, k, pos));
                }
            }
        }
        best.map(|(_, k, pos)| (k, pos))
    }

    /// The route `c` goes into with the ejected customers, chosen by the
    /// smallest sum of failure counters (then insertion cost): one customer
    /// ejected if any single ejection works, else two.
    fn eject(&self, routes: &[Vec<usize>], c: usize, failures: &[u32]) -> Option<Ejection> {
        for size in 1..=2 {
            let mut best: Option<Ejection> = None;
            for (k, route) in routes.iter().enumerate() {
                let n = route.len();
                let pairs: Vec<Vec<usize>> = if size == 1 {
                    (0..n).map(|i| vec![i]).collect()
                } else {
                    (0..n)
                        .flat_map(|i| (i + 1..n).map(move |j| vec![i, j]))
                        .collect()
                };
                for out in pairs {
                    let penalty: u32 = out.iter().map(|&i| failures[route[i]]).sum();
                    if best.as_ref().is_some_and(|b| penalty > b.key.0) {
                        continue;
                    }
                    let kept: Vec<usize> = route
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !out.contains(i))
                        .map(|(_, &x)| x)
                        .collect();
                    if self.load(&kept) + self.customers[c].demand() > self.vehicle.capacity() {
                        continue;
                    }
                    for pos in 0..=kept.len() {
                        let key = (penalty, self.insertion_cost(&kept, pos, c));
                        if best.as_ref().is_some_and(|b| key >= b.key) {
                            continue;
                        }
                        let mut trial = kept.clone();
                        trial.insert(pos, c);
                        if self.feasible(&trial) {
                            best = Some(Ejection {
                                key,
                                route: k,
                                into: trial,
                                ejected: out.iter().map(|&i| route[i]).collect(),
                            });
                        }
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// `routes` with every customer of `removed` inserted, or `None` when the
    /// pool does not empty within `max_iterations` insertions.
    fn empty_into(
        &self,
        mut routes: Vec<Vec<usize>>,
        removed: &[usize],
        max_iterations: usize,
    ) -> Option<Vec<Vec<usize>>> {
        let mut failures = vec![1u32; self.customers.len()];
        let mut pool: Vec<usize> = removed.to_vec();
        let mut iterations = 0;
        while let Some(c) = pool.pop() {
            if iterations == max_iterations {
                return None;
            }
            iterations += 1;
            if let Some((k, pos)) = self.cheapest(&routes, c) {
                routes[k].insert(pos, c);
                continue;
            }
            failures[c] += 1;
            let ejection = self.eject(&routes, c, &failures)?;
            routes[ejection.route] = ejection.into;
            pool.extend(ejection.ejected);
        }
        Some(routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served(routes: &[Vec<usize>]) -> Vec<usize> {
        let mut all: Vec<usize> = routes.concat();
        all.sort_unstable();
        all
    }

    /// Customers 1..=4 on a line with demands 2, 1, 1, 2.
    fn four() -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for (i, d) in [2, 1, 1, 2].into_iter().enumerate() {
            customers.push(Customer::new(i + 1, (i + 1) as f64, 0.0, d, 0.0));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    fn pool<'a>(customers: &'a [Customer], dm: &'a DistanceMatrix, v: &'a Vehicle) -> Pool<'a> {
        Pool {
            customers,
            distances: dm,
            vehicle: v,
            limits: RouteLimits::of(v),
            constrained: route_checks_needed(customers, &RouteLimits::of(v)),
        }
    }

    #[test]
    fn ejecting_a_customer_frees_the_place_the_removed_one_needs() {
        // Capacity 3: customer 1 (demand 2) fits neither [4] (load 2) nor
        // [2, 3] (load 2) directly. Ejecting 2 or 3 makes room, and the
        // ejected one (demand 1) then fits beside 4.
        let (customers, dm) = four();
        let vehicle = Vehicle::new(0, 3).with_fixed_cost(100.0);
        let p = pool(&customers, &dm, &vehicle);
        assert!(p.cheapest(&[vec![4], vec![2, 3]], 1).is_none());
        let routes = p
            .empty_into(vec![vec![4], vec![2, 3]], &[1], 100)
            .expect("the pool empties");
        assert_eq!(routes.len(), 2);
        assert_eq!(served(&routes), vec![1, 2, 3, 4]);
        for route in &routes {
            assert!(
                p.load(route) <= 3,
                "route {route:?} carries {}",
                p.load(route)
            );
        }
    }

    #[test]
    fn the_pool_gives_up_when_the_demand_cannot_fit() {
        // Capacity 2 and total demand 6: two routes cannot serve it, so the
        // pool cycles until the budget runs out and the routes stay.
        let (customers, dm) = four();
        let vehicle = Vehicle::new(0, 2).with_fixed_cost(100.0);
        let p = pool(&customers, &dm, &vehicle);
        assert!(p.empty_into(vec![vec![4], vec![2, 3]], &[1], 50).is_none());
        let routes = minimize_routes(
            vec![vec![1], vec![2, 3], vec![4]],
            &customers,
            &dm,
            &vehicle,
            50,
        );
        assert_eq!(routes.len(), 3);
        assert_eq!(served(&routes), vec![1, 2, 3, 4]);
    }

    #[test]
    fn minimize_routes_removes_routes_while_the_fixed_cost_pays() {
        let (customers, dm) = four();
        let vehicle = Vehicle::new(0, 3).with_fixed_cost(100.0);
        let routes = minimize_routes(
            vec![vec![1], vec![4], vec![2, 3]],
            &customers,
            &dm,
            &vehicle,
            100,
        );
        assert_eq!(routes.len(), 2);
        assert_eq!(served(&routes), vec![1, 2, 3, 4]);
    }

    #[test]
    fn without_a_fixed_cost_a_longer_plan_is_refused() {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        customers.push(Customer::new(1, -5.0, 0.0, 1, 0.0));
        customers.push(Customer::new(2, 5.0, 0.0, 1, 0.0));
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 10);
        let routes = minimize_routes(vec![vec![1], vec![2]], &customers, &dm, &vehicle, 100);
        assert_eq!(routes.len(), 2);
    }
}
