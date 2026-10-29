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
//! After every ejection the routes are perturbed: random inter-route moves --
//! exchanging two customers, or swapping the tails of two routes (2-opt*) --
//! each taken only when both routes stay feasible. The ejection always
//! chooses from the neighbourhood of the routes as they are; the perturbation
//! moves the search to a different arrangement of the same customers, so the
//! next insertion can find a place the last one could not.
//!
//! This is the ejection-pool step and the perturbation of Nagata & Bräysy
//! (2009) without its squeeze (infeasible insertion repaired by local
//! search), and with at most two customers ejected at a time.
//!
//! # Complexity
//!
//! O(I × (R × L³ + P × L)) per route tried, I the iteration budget, R routes,
//! L the length of a route and P the perturbation moves per ejection: for
//! each pair ejected, the latest arrival each remaining customer allows
//! (Savelsbergh 1992) is O(L), and then every insertion position is checked
//! against it in O(1); only a position that passes builds the route.
//!
//! Savelsbergh, M. W. P. (1992). "The vehicle routing problem with time
//! windows: minimizing route duration", *ORSA Journal on Computing* 4(2),
//! 146-154.
//!
//! # Reference
//!
//! Nagata, Y. & Bräysy, O. (2009). "A powerful route minimization heuristic
//! for the vehicle routing problem with time windows", *Operations Research
//! Letters* 37(5), 333-338.

use rand::{Rng, RngExt};

use u_metaheur::Deadline;

use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, route_feasible, RouteLimits};
use crate::models::{Customer, Vehicle};

use super::route_distance;

/// Random moves tried after each ejection (Nagata & Bräysy's I_rand).
const PERTURBATION_MOVES: usize = 100;

/// Removes routes of `routes` while the ejection pool can place their
/// customers in the others within `max_iterations` insertions per route, and
/// removing them lowers the cost (distance plus `vehicle`'s fixed cost per
/// route). Every route returned keeps capacity, time windows and the route
/// limits when the routes given do, and serves the same customers. `rng`
/// drives the perturbation; once `deadline` passes, the routes reached so far
/// are returned.
pub(crate) fn minimize_routes<R: Rng>(
    mut routes: Vec<Vec<usize>>,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicle: &Vehicle,
    max_iterations: usize,
    rng: &mut R,
    deadline: &Deadline,
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
            if let Some(rest) = pool.empty_into(rest, &routes[r], max_iterations, rng, deadline) {
                if cost(&rest) < before - 1e-12 * before.abs() {
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

    /// The time each customer of `route` is left (after its service), in
    /// order, or `None` when one of them is reached after its window closes.
    ///
    /// The clock only moves forward -- service starts at the arrival or the
    /// window's opening, whichever is later -- so inserting a customer can
    /// only delay the ones after it: a route already late stays late
    /// wherever a customer goes in.
    fn departures(&self, route: &[usize]) -> Option<Vec<f64>> {
        let mut time = 0.0;
        let mut prev = self.vehicle.depot_id();
        let mut out = Vec::with_capacity(route.len());
        for &cid in route {
            time = self.served_at(time + self.distances.travel_time(prev, cid), cid)?;
            out.push(time);
            prev = cid;
        }
        Some(out)
    }

    /// When the service of `c` ends after arriving at `arrival`, or `None`
    /// when its window has closed.
    fn served_at(&self, arrival: f64, c: usize) -> Option<f64> {
        let customer = &self.customers[c];
        let start = match customer.time_window() {
            Some(tw) if tw.is_violated(arrival) => return None,
            Some(tw) => arrival + tw.waiting_time(arrival),
            None => arrival,
        };
        Some(start + customer.service_duration())
    }

    /// The latest arrival at each customer of `route` that still lets it and
    /// every customer after it start within their windows (Savelsbergh 1992):
    /// service at a customer must end in time to reach the next by the next's
    /// latest arrival, and cannot start before the window opens -- when the
    /// opening is already too late, no arrival works (−∞).
    fn latest_arrivals(&self, route: &[usize]) -> Vec<f64> {
        let depot = self.vehicle.depot_id();
        let mut latest = vec![f64::INFINITY; route.len()];
        // The depot has no window: the return is bounded by the route limits
        // only, which the full check applies.
        let mut next_latest = f64::INFINITY;
        for i in (0..route.len()).rev() {
            let c = route[i];
            let customer = &self.customers[c];
            let next = route.get(i + 1).copied().unwrap_or(depot);
            let start_by =
                next_latest - self.distances.travel_time(c, next) - customer.service_duration();
            let (ready, due) = customer
                .time_window()
                .map_or((f64::NEG_INFINITY, f64::INFINITY), |tw| {
                    (tw.ready(), tw.due())
                });
            latest[i] = if ready > start_by {
                f64::NEG_INFINITY
            } else {
                due.min(start_by)
            };
            next_latest = latest[i];
        }
        latest
    }

    /// Whether `c`, inserted at `pos` of a route left at `departures` with
    /// `latest` arrivals, can keep every window: it starts within its own,
    /// and the customer after it is still reached by its latest arrival.
    /// Checked in O(1) without building the route, and a necessary condition
    /// only -- the slack absorbs rounding, so the full check decides.
    fn may_fit(
        &self,
        route: &[usize],
        departures: &[f64],
        latest: &[f64],
        pos: usize,
        c: usize,
    ) -> bool {
        let (left, prev) = if pos == 0 {
            (0.0, self.vehicle.depot_id())
        } else {
            (departures[pos - 1], route[pos - 1])
        };
        let Some(left_c) = self.served_at(left + self.distances.travel_time(prev, c), c) else {
            return false;
        };
        match route.get(pos) {
            None => true,
            Some(&next) => {
                let bound = latest[pos];
                left_c + self.distances.travel_time(c, next) <= bound + 1e-9 * bound.abs().max(1.0)
            }
        }
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
            let (departures, latest) = if self.constrained {
                match self.departures(route) {
                    Some(d) => (d, self.latest_arrivals(route)),
                    None => continue,
                }
            } else {
                (Vec::new(), Vec::new())
            };
            for pos in 0..=route.len() {
                let cost = self.insertion_cost(route, pos, c);
                if best.is_some_and(|(b, _, _)| cost >= b) {
                    continue;
                }
                if self.constrained && !self.may_fit(route, &departures, &latest, pos, c) {
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
                    let (departures, latest) = if self.constrained {
                        match self.departures(&kept) {
                            Some(d) => (d, self.latest_arrivals(&kept)),
                            None => continue,
                        }
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    for pos in 0..=kept.len() {
                        let key = (penalty, self.insertion_cost(&kept, pos, c));
                        if best.as_ref().is_some_and(|b| key >= b.key) {
                            continue;
                        }
                        if self.constrained && !self.may_fit(&kept, &departures, &latest, pos, c) {
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

    /// Up to `moves` random feasible inter-route moves on `routes`: an
    /// exchange of two customers or a 2-opt* (the tails of two routes
    /// swapped), each kept only when both routes stay feasible and non-empty.
    fn perturb<R: Rng>(&self, routes: &mut [Vec<usize>], moves: usize, rng: &mut R) {
        let n = routes.len();
        if n < 2 {
            return;
        }
        for _ in 0..moves {
            let a = rng.random_range(0..n as u64) as usize;
            let mut b = rng.random_range(0..(n - 1) as u64) as usize;
            if b >= a {
                b += 1;
            }
            let (la, lb) = (routes[a].len(), routes[b].len());
            if la == 0 || lb == 0 {
                continue;
            }
            let (ra, rb) = if rng.random_bool(0.5) {
                let i = rng.random_range(0..la as u64) as usize;
                let j = rng.random_range(0..lb as u64) as usize;
                let (mut ra, mut rb) = (routes[a].clone(), routes[b].clone());
                std::mem::swap(&mut ra[i], &mut rb[j]);
                (ra, rb)
            } else {
                let i = rng.random_range(0..=la as u64) as usize;
                let j = rng.random_range(0..=lb as u64) as usize;
                let ra: Vec<usize> = routes[a][..i]
                    .iter()
                    .chain(&routes[b][j..])
                    .copied()
                    .collect();
                let rb: Vec<usize> = routes[b][..j]
                    .iter()
                    .chain(&routes[a][i..])
                    .copied()
                    .collect();
                if ra.is_empty() || rb.is_empty() {
                    continue;
                }
                (ra, rb)
            };
            if self.feasible(&ra) && self.feasible(&rb) {
                routes[a] = ra;
                routes[b] = rb;
            }
        }
    }

    /// `routes` with every customer of `removed` inserted, or `None` when the
    /// pool does not empty within `max_iterations` insertions or before
    /// `deadline`.
    fn empty_into<R: Rng>(
        &self,
        mut routes: Vec<Vec<usize>>,
        removed: &[usize],
        max_iterations: usize,
        rng: &mut R,
        deadline: &Deadline,
    ) -> Option<Vec<Vec<usize>>> {
        let mut failures = vec![1u32; self.customers.len()];
        let mut pool: Vec<usize> = removed.to_vec();
        let mut iterations = 0;
        while let Some(c) = pool.pop() {
            if iterations == max_iterations || deadline.passed() {
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
            self.perturb(&mut routes, PERTURBATION_MOVES, rng);
        }
        Some(routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    fn rng() -> rand::rngs::SmallRng {
        u_numflow::random::create_rng(7)
    }

    fn unlimited() -> Deadline {
        Deadline::start(None)
    }

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

    proptest::proptest! {
        /// On a route that keeps its windows, the O(1) check agrees with
        /// building the route: an insertion it refuses breaks a window, and
        /// one it allows keeps them all (no route limits, ample capacity).
        #[test]
        fn may_fit_agrees_with_the_full_check(
            points in proptest::collection::vec(
                (0u8..40, 0u8..40, 0u16..200, 0u16..120, 0u8..10),
                2..9,
            ),
            c_at in (0u8..40, 0u8..40, 0u16..200, 0u16..120, 0u8..10),
        ) {
            let mut customers = vec![Customer::depot(20.0, 20.0)];
            for (i, &(x, y, ready, width, service)) in
                points.iter().chain(std::iter::once(&c_at)).enumerate()
            {
                let ready = f64::from(ready);
                customers.push(
                    Customer::new(i + 1, f64::from(x), f64::from(y), 1, f64::from(service))
                        .with_time_window(
                            TimeWindow::new(ready, ready + f64::from(width)).expect("valid"),
                        ),
                );
            }
            let dm = DistanceMatrix::from_customers(&customers);
            let vehicle = Vehicle::new(0, 1000);
            let pool = pool(&customers, &dm, &vehicle);
            let c = customers.len() - 1;
            // The longest prefix of the points that keeps its windows.
            let mut route: Vec<usize> = Vec::new();
            for k in 1..c {
                route.push(k);
                if pool.departures(&route).is_none() {
                    route.pop();
                }
            }
            let departures = pool.departures(&route).expect("route keeps its windows");
            let latest = pool.latest_arrivals(&route);
            for pos in 0..=route.len() {
                let mut trial = route.clone();
                trial.insert(pos, c);
                proptest::prop_assert_eq!(
                    pool.may_fit(&route, &departures, &latest, pos, c),
                    pool.feasible(&trial),
                    "pos {} of {:?}", pos, route
                );
            }
        }
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
            .empty_into(
                vec![vec![4], vec![2, 3]],
                &[1],
                100,
                &mut rng(),
                &unlimited(),
            )
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
        assert!(p
            .empty_into(
                vec![vec![4], vec![2, 3]],
                &[1],
                50,
                &mut rng(),
                &unlimited()
            )
            .is_none());
        let routes = minimize_routes(
            vec![vec![1], vec![2, 3], vec![4]],
            &customers,
            &dm,
            &vehicle,
            50,
            &mut rng(),
            &unlimited(),
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
            &mut rng(),
            &unlimited(),
        );
        assert_eq!(routes.len(), 2);
        assert_eq!(served(&routes), vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_passed_deadline_keeps_the_routes_given() {
        // The plan above loses a route with time to search; out of time, the
        // routes come back as they were.
        let (customers, dm) = four();
        let vehicle = Vehicle::new(0, 3).with_fixed_cost(100.0);
        let routes = minimize_routes(
            vec![vec![1], vec![4], vec![2, 3]],
            &customers,
            &dm,
            &vehicle,
            100,
            &mut rng(),
            &Deadline::start(Some(0)),
        );
        assert_eq!(routes, vec![vec![1], vec![4], vec![2, 3]]);
    }

    #[test]
    fn without_a_fixed_cost_a_longer_plan_is_refused() {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        customers.push(Customer::new(1, -5.0, 0.0, 1, 0.0));
        customers.push(Customer::new(2, 5.0, 0.0, 1, 0.0));
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicle = Vehicle::new(0, 10);
        let routes = minimize_routes(
            vec![vec![1], vec![2]],
            &customers,
            &dm,
            &vehicle,
            100,
            &mut rng(),
            &unlimited(),
        );
        assert_eq!(routes.len(), 2);
    }
}
