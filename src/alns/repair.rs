//! Repair operators for ALNS-based VRP optimization.
//!
//! # Operators
//!
//! - [`GreedyInsertion`] — Inserts each customer at the cheapest position
//! - [`RegretInsertion`] — Inserts the customer with highest regret value first
//!
//! # Reference
//!
//! Ropke, S. & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search
//! Heuristic for the Pickup and Delivery Problem with Time Windows",
//! *Transportation Science* 40(4), 455-472.

use rand::Rng;
use u_metaheur::alns::RepairOperator;
use u_metaheur::Deadline;

use crate::distance::DistanceMatrix;
use crate::evaluation::{effective_cost, route_feasible, RouteLimits};
use crate::fleet::{cheapest_in_route, InstanceKind};
use crate::models::Customer;

use super::solution_repr::RoutingSolution;

/// What every insertion of one repair is judged by: the instance, the vehicle's capacity and
/// limits, the price of opening a route, and what the instance says about how a position is
/// checked (read once -- each is a scan of every customer).
struct Context<'a> {
    distances: &'a DistanceMatrix,
    customers: &'a [Customer],
    capacity: i32,
    limits: &'a RouteLimits,
    kind: InstanceKind,
    route_cost: f64,
}

impl Context<'_> {
    /// The cheapest position of `customer_id` in `route`, as `(position, cost_increase)`.
    fn in_route(&self, route: &[usize], customer_id: usize) -> Option<(usize, f64)> {
        cheapest_in_route(
            route,
            self.capacity,
            self.limits,
            self.kind,
            customer_id,
            self.distances,
            self.customers,
            f64::INFINITY,
        )
    }

    /// Whether a route of this one customer reaches it on time and within the vehicle's limits.
    /// Always true with no window and no limit.
    fn open_route(&self, customer_id: usize) -> bool {
        !self.kind.constrained
            || route_feasible(
                &[customer_id],
                0,
                self.distances,
                self.customers,
                self.limits,
            )
    }

    /// What opening a route for `customer_id` alone costs: `route_cost` (the vehicle's fixed cost
    /// in distance units) plus the round trip, or `None` when even that route breaks a time
    /// window or a limit.
    fn new_route_cost(&self, customer_id: usize) -> Option<f64> {
        if self.kind.soft {
            // A route of its own is late where the customer cannot be reached in time, and that
            // costs.
            let alone = effective_cost(
                &[customer_id],
                0,
                self.distances,
                self.customers,
                self.limits,
            );
            return alone.is_finite().then_some(self.route_cost + alone);
        }
        self.open_route(customer_id).then(|| {
            self.route_cost
                + self.distances.get(0, customer_id)
                + self.distances.get(customer_id, 0)
        })
    }
}

/// The customers a repair still has to place, each with its cheapest insertion into every route
/// and the cost of a route of its own.
///
/// Placing a customer changes one route, so only that route's column is computed again: a
/// repair that re-priced every route for every customer after each placement spent most of its
/// time on routes that had not changed (O(u²·n) for `u` customers over `n`, now O(u·n + u²·L)
/// with `L` the length of a route).
struct Pending {
    ids: Vec<usize>,
    /// `table[u][r]`: the cheapest `(position, cost_increase)` of `ids[u]` in route `r`.
    table: Vec<Vec<Option<(usize, f64)>>>,
    /// `fresh[u]`: what a route of its own costs `ids[u]`.
    fresh: Vec<Option<f64>>,
}

impl Pending {
    /// Prices every customer of `ids` against `routes`; `None` when `deadline` passes first.
    fn new(
        ids: Vec<usize>,
        routes: &[Vec<usize>],
        cx: &Context<'_>,
        deadline: &Deadline,
    ) -> Option<Self> {
        let mut table = Vec::with_capacity(ids.len());
        let mut fresh = Vec::with_capacity(ids.len());
        for &cid in &ids {
            if deadline.passed() {
                return None;
            }
            table.push(routes.iter().map(|r| cx.in_route(r, cid)).collect());
            fresh.push(cx.new_route_cost(cid));
        }
        Some(Self { ids, table, fresh })
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Takes customer `u` out (the others keep their order); returns its id.
    fn remove(&mut self, u: usize) -> usize {
        self.table.remove(u);
        self.fresh.remove(u);
        self.ids.remove(u)
    }

    /// Prices every customer against route `r`, which has changed -- or is the new route after
    /// the last one. `false` when `deadline` passes first.
    fn refresh(
        &mut self,
        r: usize,
        route: &[usize],
        cx: &Context<'_>,
        deadline: &Deadline,
    ) -> bool {
        for u in 0..self.ids.len() {
            if deadline.passed() {
                return false;
            }
            let entry = cx.in_route(route, self.ids[u]);
            if r < self.table[u].len() {
                self.table[u][r] = entry;
            } else {
                self.table[u].push(entry);
            }
        }
        true
    }

    /// The cheapest existing route for customer `u` as `(route, position, cost)`: the least cost,
    /// the first route among equals.
    fn cheapest(&self, u: usize) -> Option<(usize, usize, f64)> {
        let mut best: Option<(usize, usize, f64)> = None;
        for (ri, entry) in self.table[u].iter().enumerate() {
            if let Some((pos, cost)) = *entry {
                if best.is_none_or(|b| cost < b.2) {
                    best = Some((ri, pos, cost));
                }
            }
        }
        best
    }
}

/// Greedy insertion: inserts each unassigned customer at its cheapest position.
///
/// Iteratively selects the unassigned customer with the lowest insertion cost
/// and places it at the best position. Creates new routes when needed.
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::alns::{RoutingSolution, repair::GreedyInsertion};
/// use u_metaheur::alns::RepairOperator;
///
/// let cust = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&cust);
/// let sol = RoutingSolution::new(vec![vec![1]], vec![2], &cust, &dm);
///
/// let op = GreedyInsertion::new(dm, cust, 100);
/// let mut rng = u_numflow::random::create_rng(42);
/// let repaired = op.repair(&sol, &mut rng);
/// assert!(repaired.unassigned().is_empty());
/// ```
pub struct GreedyInsertion {
    limits: RouteLimits,
    kind: InstanceKind,
    distances: DistanceMatrix,
    customers: Vec<Customer>,
    capacity: i32,
    route_cost: f64,
}

impl GreedyInsertion {
    /// Creates a new greedy insertion operator.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>, capacity: i32) -> Self {
        let kind = InstanceKind::of(&customers, &RouteLimits::NONE);
        Self {
            distances,
            customers,
            capacity,
            limits: RouteLimits::NONE,
            kind,
            route_cost: 0.0,
        }
    }

    /// Inserts only where the route keeps the vehicle's distance and duration
    /// limits (none by default).
    pub fn with_limits(mut self, limits: RouteLimits) -> Self {
        self.kind = InstanceKind::of(&self.customers, &limits);
        self.limits = limits;
        self
    }

    /// The cost of opening a route, in distance units (a vehicle's
    /// [`fixed_cost_in_distance`](crate::models::Vehicle::fixed_cost_in_distance);
    /// 0 by default). A new route is a candidate like any insertion, priced
    /// at this plus its round trip.
    pub fn with_route_cost(mut self, route_cost: f64) -> Self {
        self.route_cost = route_cost;
        self
    }
}

impl RepairOperator<RoutingSolution> for GreedyInsertion {
    fn name(&self) -> &str {
        "greedy_insertion"
    }

    fn repair<R: Rng>(&self, solution: &RoutingSolution, rng: &mut R) -> RoutingSolution {
        self.repair_within(solution, rng, &Deadline::start(None))
            .expect("a repair with no deadline completes")
    }

    /// Stops as soon as `deadline` passes: one repair over a large instance takes longer than
    /// a short budget, and the runner can only look between iterations.
    fn repair_within<R: Rng>(
        &self,
        solution: &RoutingSolution,
        _rng: &mut R,
        deadline: &Deadline,
    ) -> Option<RoutingSolution> {
        let mut sol = solution.clone();
        let unassigned = std::mem::take(sol.unassigned_mut());
        let cx = Context {
            distances: &self.distances,
            customers: &self.customers,
            capacity: self.capacity,
            limits: &self.limits,
            kind: self.kind,
            route_cost: self.route_cost,
        };
        let mut pending = Pending::new(unassigned, sol.routes(), &cx, deadline)?;

        while !pending.is_empty() {
            if deadline.passed() {
                return None;
            }
            // Find the unassigned customer with the cheapest insertion
            let mut best_cust_idx = 0;
            let mut best_route = 0;
            let mut best_pos = 0;
            let mut best_cost = f64::INFINITY;
            let mut found = false;

            for u in 0..pending.len() {
                if let Some((ri, pos, cost)) = pending.cheapest(u) {
                    if cost < best_cost {
                        best_cost = cost;
                        best_cust_idx = u;
                        best_route = ri;
                        best_pos = pos;
                        found = true;
                    }
                }
                // A route of its own is a candidate at its price.
                if let Some(cost) = pending.fresh[u] {
                    if cost < best_cost {
                        best_cost = cost;
                        best_cust_idx = u;
                        best_route = NEW_ROUTE;
                        best_pos = 0;
                        found = true;
                    }
                }
            }

            let changed = if !found {
                // No feasible insertion — open a route for the first unassigned
                // customer, or leave it unassigned if even that arrives late.
                let cid = pending.remove(0);
                if cx.open_route(cid) {
                    sol.routes_mut().push(vec![cid]);
                    Some(sol.routes().len() - 1)
                } else {
                    sol.unassigned_mut().push(cid);
                    None
                }
            } else {
                let cid = pending.remove(best_cust_idx);
                place(&mut sol, best_route, best_pos, cid);
                Some(if best_route == NEW_ROUTE {
                    sol.routes().len() - 1
                } else {
                    best_route
                })
            };
            if let Some(r) = changed {
                if !pending.refresh(r, &sol.routes()[r], &cx, deadline) {
                    return None;
                }
            }
        }

        sol.recalculate(&self.distances, &self.customers);
        Some(sol)
    }
}

/// The route index that stands for a new route in a candidate.
const NEW_ROUTE: usize = usize::MAX;

/// Puts `cid` at `pos` of route `route`, or on a new route of its own.
fn place(sol: &mut RoutingSolution, route: usize, pos: usize, cid: usize) {
    if route == NEW_ROUTE {
        sol.routes_mut().push(vec![cid]);
    } else {
        sol.routes_mut()[route].insert(pos, cid);
    }
}

/// Regret-k insertion: prioritizes customers with the highest regret value.
///
/// Regret-k is defined as the difference between the k-th best and the best
/// insertion cost. Customers with high regret have fewer good alternatives
/// and should be inserted first.
///
/// Uses k=2 (regret-2) by default.
pub struct RegretInsertion {
    limits: RouteLimits,
    kind: InstanceKind,
    distances: DistanceMatrix,
    customers: Vec<Customer>,
    capacity: i32,
    k: usize,
    route_cost: f64,
}

impl RegretInsertion {
    /// Creates a new regret-k insertion operator with k=2.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>, capacity: i32) -> Self {
        let kind = InstanceKind::of(&customers, &RouteLimits::NONE);
        Self {
            distances,
            customers,
            capacity,
            k: 2,
            limits: RouteLimits::NONE,
            kind,
            route_cost: 0.0,
        }
    }

    /// The cost of opening a route, as [`GreedyInsertion::with_route_cost`].
    pub fn with_route_cost(mut self, route_cost: f64) -> Self {
        self.route_cost = route_cost;
        self
    }

    /// Inserts only where the route keeps the vehicle's distance and duration
    /// limits (none by default).
    pub fn with_limits(mut self, limits: RouteLimits) -> Self {
        self.kind = InstanceKind::of(&self.customers, &limits);
        self.limits = limits;
        self
    }

    /// Creates a regret insertion operator with custom k.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(2);
        self
    }
}

impl RepairOperator<RoutingSolution> for RegretInsertion {
    fn name(&self) -> &str {
        "regret_insertion"
    }

    fn repair<R: Rng>(&self, solution: &RoutingSolution, rng: &mut R) -> RoutingSolution {
        self.repair_within(solution, rng, &Deadline::start(None))
            .expect("a repair with no deadline completes")
    }

    /// Stops as soon as `deadline` passes, as [`GreedyInsertion::repair_within`].
    fn repair_within<R: Rng>(
        &self,
        solution: &RoutingSolution,
        _rng: &mut R,
        deadline: &Deadline,
    ) -> Option<RoutingSolution> {
        let mut sol = solution.clone();
        let unassigned = std::mem::take(sol.unassigned_mut());
        let cx = Context {
            distances: &self.distances,
            customers: &self.customers,
            capacity: self.capacity,
            limits: &self.limits,
            kind: self.kind,
            route_cost: self.route_cost,
        };
        let mut pending = Pending::new(unassigned, sol.routes(), &cx, deadline)?;

        while !pending.is_empty() {
            if deadline.passed() {
                return None;
            }
            let mut best_regret = f64::NEG_INFINITY;
            let mut best_cust_idx = 0;
            let mut best_route = 0;
            let mut best_pos = 0;
            let mut found = false;

            for u in 0..pending.len() {
                // Insertion costs of the customer across the routes, and a route of its own as
                // one more alternative at its price; cheapest first (equal costs keep the route
                // order).
                let mut costs: Vec<(usize, usize, f64)> = pending.table[u]
                    .iter()
                    .enumerate()
                    .filter_map(|(ri, entry)| entry.map(|(pos, cost)| (ri, pos, cost)))
                    .collect();
                if let Some(cost) = pending.fresh[u] {
                    costs.push((NEW_ROUTE, 0, cost));
                }
                costs.sort_by(|a, b| {
                    a.2.partial_cmp(&b.2)
                        .expect("insertion costs should not be NaN")
                });

                if costs.is_empty() {
                    continue;
                }

                let best_cost = costs[0].2;
                // Regret = sum of differences between k-th best and best
                let regret: f64 = costs
                    .iter()
                    .skip(1)
                    .take(self.k - 1)
                    .map(|c| c.2 - best_cost)
                    .sum();

                // If fewer than k routes available, use large regret (prioritize)
                let regret = if costs.len() < self.k {
                    regret + f64::MAX / 2.0
                } else {
                    regret
                };

                if regret > best_regret || (regret == best_regret && best_cost < costs[0].2) {
                    best_regret = regret;
                    best_cust_idx = u;
                    best_route = costs[0].0;
                    best_pos = costs[0].1;
                    found = true;
                }
            }

            let changed = if !found {
                // Open a route for the first unassigned customer, or leave it
                // unassigned if even that arrives late.
                let cid = pending.remove(0);
                if cx.open_route(cid) {
                    sol.routes_mut().push(vec![cid]);
                    Some(sol.routes().len() - 1)
                } else {
                    sol.unassigned_mut().push(cid);
                    None
                }
            } else {
                let cid = pending.remove(best_cust_idx);
                place(&mut sol, best_route, best_pos, cid);
                Some(if best_route == NEW_ROUTE {
                    sol.routes().len() - 1
                } else {
                    best_route
                })
            };
            if let Some(r) = changed {
                if !pending.refresh(r, &sol.routes()[r], &cx, deadline) {
                    return None;
                }
            }
        }

        sol.recalculate(&self.distances, &self.customers);
        Some(sol)
    }
}

/// The cheapest insertion of `customer_id` across `routes` of one capacity (what a repair tracks
/// per route, as one answer).
#[cfg(test)]
fn best_insertion(
    routes: &[Vec<usize>],
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    capacity: i32,
    limits: &RouteLimits,
) -> Option<(usize, usize, f64)> {
    crate::fleet::cheapest_insertion(
        routes,
        |_| capacity,
        |_| *limits,
        customer_id,
        distances,
        customers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Vec<Customer>, DistanceMatrix) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
            Customer::new(4, 4.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn test_greedy_inserts_all() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1]], vec![2, 3, 4], &cust, &dm);
        let op = GreedyInsertion::new(dm.clone(), cust.clone(), 100);
        let mut rng = u_numflow::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
        let total: usize = repaired.routes().iter().map(|r| r.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_greedy_creates_new_route_when_full() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2]], vec![3, 4], &cust, &dm);
        let op = GreedyInsertion::new(dm.clone(), cust.clone(), 20); // cap 20, demand 10 each
        let mut rng = u_numflow::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
        assert!(repaired.num_routes() >= 2);
    }

    #[test]
    fn test_regret_inserts_all() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1]], vec![2, 3, 4], &cust, &dm);
        let op = RegretInsertion::new(dm.clone(), cust.clone(), 100);
        let mut rng = u_numflow::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
        let total: usize = repaired.routes().iter().map(|r| r.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_regret_prioritizes_constrained_customers() {
        let (cust, dm) = setup();
        // Two routes, both nearly full — regret should prioritize customers
        // with fewer insertion options
        let sol = RoutingSolution::new(vec![vec![1], vec![2]], vec![3, 4], &cust, &dm);
        let op = RegretInsertion::new(dm.clone(), cust.clone(), 20);
        let mut rng = u_numflow::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
    }

    #[test]
    fn test_best_insertion_position() {
        let (cust, dm) = setup();
        // Route [1, 3], insert 2 — best position should be between 1 and 3
        let routes = vec![vec![1, 3]];
        let result = best_insertion(&routes, 2, &dm, &cust, 100, &RouteLimits::NONE);
        assert!(result.is_some());
        let (ri, pos, _cost) = result.expect("should find insertion");
        assert_eq!(ri, 0);
        assert_eq!(pos, 1); // between 1 and 3
    }

    /// A repair whose deadline has passed gives up (the runner drops the candidate); with no
    /// deadline it completes.
    #[test]
    fn a_spent_deadline_cuts_a_repair_short() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1]], vec![2, 3, 4], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(42);
        let spent = Deadline::start(Some(0));
        let greedy = GreedyInsertion::new(dm.clone(), cust.clone(), 100);
        let regret = RegretInsertion::new(dm.clone(), cust.clone(), 100);
        assert!(greedy.repair_within(&sol, &mut rng, &spent).is_none());
        assert!(regret.repair_within(&sol, &mut rng, &spent).is_none());
        let open = Deadline::start(None);
        for repaired in [
            greedy.repair_within(&sol, &mut rng, &open),
            regret.repair_within(&sol, &mut rng, &open),
        ] {
            assert!(repaired.expect("no deadline").unassigned().is_empty());
        }
    }

    /// The repair as it was written first: after every placement it prices every pending
    /// customer against every route again. The cached repair must place exactly the same.
    fn plain_repair(
        solution: &RoutingSolution,
        distances: &DistanceMatrix,
        customers: &[Customer],
        capacity: i32,
        limits: &RouteLimits,
        regret: bool,
    ) -> RoutingSolution {
        let mut sol = solution.clone();
        let mut unassigned = std::mem::take(sol.unassigned_mut());
        let cx = Context {
            distances,
            customers,
            capacity,
            limits,
            kind: InstanceKind::of(customers, limits),
            route_cost: 0.0,
        };
        while !unassigned.is_empty() {
            let mut best = (f64::NEG_INFINITY, 0, 0, 0, false);
            let mut best_cost = f64::INFINITY;
            for (u, &cid) in unassigned.iter().enumerate() {
                let mut costs: Vec<(usize, usize, f64)> = Vec::new();
                for (ri, route) in sol.routes().iter().enumerate() {
                    if let Some((pos, cost)) = cx.in_route(route, cid) {
                        costs.push((ri, pos, cost));
                    }
                }
                if let Some(cost) = cx.new_route_cost(cid) {
                    costs.push((NEW_ROUTE, 0, cost));
                }
                if regret {
                    costs.sort_by(|a, b| a.2.partial_cmp(&b.2).expect("no NaN"));
                    if costs.is_empty() {
                        continue;
                    }
                    let first = costs[0].2;
                    let mut score: f64 = costs.iter().skip(1).take(1).map(|c| c.2 - first).sum();
                    if costs.len() < 2 {
                        score += f64::MAX / 2.0;
                    }
                    if score > best.0 {
                        best = (score, u, costs[0].0, costs[0].1, true);
                    }
                } else {
                    // Existing routes first (the first of equal cost), then a route of its own.
                    let mut existing: Option<(usize, usize, f64)> = None;
                    for &(ri, pos, cost) in costs.iter().filter(|c| c.0 != NEW_ROUTE) {
                        if existing.is_none_or(|e| cost < e.2) {
                            existing = Some((ri, pos, cost));
                        }
                    }
                    if let Some((ri, pos, cost)) = existing {
                        if cost < best_cost {
                            best_cost = cost;
                            best = (0.0, u, ri, pos, true);
                        }
                    }
                    if let Some(&(_, _, cost)) = costs.iter().find(|c| c.0 == NEW_ROUTE) {
                        if cost < best_cost {
                            best_cost = cost;
                            best = (0.0, u, NEW_ROUTE, 0, true);
                        }
                    }
                }
            }
            if !best.4 {
                let cid = unassigned.remove(0);
                if cx.open_route(cid) {
                    sol.routes_mut().push(vec![cid]);
                } else {
                    sol.unassigned_mut().push(cid);
                }
            } else {
                let cid = unassigned.remove(best.1);
                place(&mut sol, best.2, best.3, cid);
            }
        }
        sol.recalculate(distances, customers);
        sol
    }

    /// Customers on a plane with demands, and (in some instances) hard windows and a distance
    /// limit, from a fixed generator.
    fn random_instance(seed: u32, windows: bool) -> (Vec<Customer>, DistanceMatrix) {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(s >> 8) / f64::from(1u32 << 24)
        };
        let n = 8 + (next() * 24.0) as usize;
        let mut customers = vec![Customer::depot(50.0, 50.0)];
        for i in 1..=n {
            let (x, y) = (next() * 100.0, next() * 100.0);
            let demand = 1 + (next() * 6.0) as i32;
            let mut c = Customer::new(i, x, y, demand, 1.0);
            if windows {
                let open = next() * 200.0;
                c = c.with_time_window(
                    crate::models::TimeWindow::new(open, open + 20.0 + next() * 150.0)
                        .expect("a window"),
                );
            }
            customers.push(c);
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    /// Both repairs place every customer where the repair priced against every route would, on
    /// instances with tight capacity, with windows, and with a distance limit -- from nothing
    /// built and from part of a solution.
    #[test]
    fn the_cached_repairs_place_what_the_plain_scan_places() {
        let mut rng = u_numflow::random::create_rng(1);
        let (mut left_out, mut several_routes) = (false, false);
        for seed in 0..60u32 {
            let windows = seed % 3 != 0;
            let (customers, dm) = random_instance(seed, windows);
            let n = customers.len() - 1;
            let capacity = 8 + (seed % 5) as i32 * 4;
            let limits = if seed % 4 == 0 {
                RouteLimits {
                    max_distance: Some(260.0),
                    ..RouteLimits::NONE
                }
            } else {
                RouteLimits::NONE
            };
            let all: Vec<usize> = (1..=n).collect();
            let half = n / 2;
            let starts = [
                RoutingSolution::new(vec![], all.clone(), &customers, &dm),
                RoutingSolution::new(
                    vec![all[..half].to_vec()],
                    all[half..].to_vec(),
                    &customers,
                    &dm,
                ),
            ];
            for start in &starts {
                let greedy = GreedyInsertion::new(dm.clone(), customers.clone(), capacity)
                    .with_limits(limits);
                let regret = RegretInsertion::new(dm.clone(), customers.clone(), capacity)
                    .with_limits(limits);
                for (name, regret_flag, cached) in [
                    ("greedy", false, greedy.repair(start, &mut rng)),
                    ("regret", true, regret.repair(start, &mut rng)),
                ] {
                    let plain =
                        plain_repair(start, &dm, &customers, capacity, &limits, regret_flag);
                    assert_eq!(cached.routes(), plain.routes(), "{name} seed {seed}");
                    assert_eq!(
                        cached.unassigned(),
                        plain.unassigned(),
                        "{name} seed {seed}"
                    );
                    left_out |= !cached.unassigned().is_empty();
                    several_routes |= cached.num_routes() > 2;
                }
            }
        }
        // The instances do exercise a customer no route can take and more than a couple of routes.
        assert!(left_out && several_routes, "{left_out} {several_routes}");
    }
}
