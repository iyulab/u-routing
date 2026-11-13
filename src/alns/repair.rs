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

use crate::distance::DistanceMatrix;
use crate::evaluation::{effective_cost, has_soft_windows};
use crate::evaluation::{route_checks_needed, route_feasible, route_peak_load, RouteLimits};
use crate::fleet::{
    cheapest_insertion, cheapest_pair_insertion, insertion_on_time, soft_insertion_cost, units_of,
    Unit,
};
use crate::models::Customer;

use super::destroy::remove_customer;
use super::solution_repr::RoutingSolution;

/// Whether a route of this one customer reaches it on time and within the
/// vehicle's limits. Always true with no window and no limit.
fn open_route(
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
) -> bool {
    !route_checks_needed(customers, limits)
        || route_feasible(&[customer_id], 0, distances, customers, limits)
}

/// What opening a route for `customer_id` alone costs: `route_cost` (the
/// vehicle's fixed cost in distance units) plus the round trip, or `None` when
/// even that route breaks a time window or a limit.
fn new_route_cost(
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
    route_cost: f64,
) -> Option<f64> {
    if has_soft_windows(customers) {
        // A route of its own is late where the customer cannot be reached in time, and that costs.
        let alone = effective_cost(&[customer_id], 0, distances, customers, limits);
        return alone.is_finite().then_some(route_cost + alone);
    }
    open_route(customer_id, distances, customers, limits)
        .then(|| route_cost + distances.get(0, customer_id) + distances.get(customer_id, 0))
}

/// Finds the best insertion position for a customer across all routes of one
/// capacity. Returns `(route_index, position, cost_increase)`. A position that
/// would make a customer late is not a candidate.
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
    distances: DistanceMatrix,
    customers: Vec<Customer>,
    capacity: i32,
    route_cost: f64,
}

impl GreedyInsertion {
    /// Creates a new greedy insertion operator.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>, capacity: i32) -> Self {
        Self {
            distances,
            customers,
            capacity,
            limits: RouteLimits::NONE,
            route_cost: 0.0,
        }
    }

    /// Inserts only where the route keeps the vehicle's distance and duration
    /// limits (none by default).
    pub fn with_limits(mut self, limits: RouteLimits) -> Self {
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

    fn repair<R: Rng>(&self, solution: &RoutingSolution, _rng: &mut R) -> RoutingSolution {
        if has_pairs(&self.customers) {
            let world = PairWorld {
                distances: &self.distances,
                customers: &self.customers,
                capacity: self.capacity,
                limits: &self.limits,
                route_cost: self.route_cost,
            };
            return repair_with_pairs(solution, &world, None);
        }
        let mut sol = solution.clone();
        let mut unassigned = std::mem::take(sol.unassigned_mut());

        while !unassigned.is_empty() {
            // Find the unassigned customer with the cheapest insertion
            let mut best_cust_idx = 0;
            let mut best_route = 0;
            let mut best_pos = 0;
            let mut best_cost = f64::INFINITY;
            let mut found = false;

            for (ui, &cid) in unassigned.iter().enumerate() {
                if let Some((ri, pos, cost)) = best_insertion(
                    sol.routes(),
                    cid,
                    &self.distances,
                    &self.customers,
                    self.capacity,
                    &self.limits,
                ) {
                    if cost < best_cost {
                        best_cost = cost;
                        best_cust_idx = ui;
                        best_route = ri;
                        best_pos = pos;
                        found = true;
                    }
                }
                // A route of its own is a candidate at its price.
                if let Some(cost) = new_route_cost(
                    cid,
                    &self.distances,
                    &self.customers,
                    &self.limits,
                    self.route_cost,
                ) {
                    if cost < best_cost {
                        best_cost = cost;
                        best_cust_idx = ui;
                        best_route = NEW_ROUTE;
                        best_pos = 0;
                        found = true;
                    }
                }
            }

            if !found {
                // No feasible insertion — open a route for the first unassigned
                // customer, or leave it unassigned if even that arrives late.
                let cid = unassigned.remove(0);
                if open_route(cid, &self.distances, &self.customers, &self.limits) {
                    sol.routes_mut().push(vec![cid]);
                } else {
                    sol.unassigned_mut().push(cid);
                }
            } else {
                let cid = unassigned.remove(best_cust_idx);
                place(&mut sol, best_route, best_pos, cid);
            }
        }

        sol.recalculate(&self.distances, &self.customers);
        sol
    }
}

/// The route index that stands for a new route in a candidate.
const NEW_ROUTE: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Pickup-delivery pairs
// ---------------------------------------------------------------------------
//
// Either half of a pair alone is not a route, so with pairs the unit a repair places is the
// pair, the pickup before its delivery, and a half a destroy operator left in a route is taken
// out with its other half first. The same two orders -- cheapest first, largest regret first --
// choose among the units.

/// Whether any customer is part of a pickup-delivery pair.
fn has_pairs(customers: &[Customer]) -> bool {
    customers.iter().any(|c| c.delivery_index().is_some())
}

/// What a repair of an instance with pairs reads.
struct PairWorld<'a> {
    distances: &'a DistanceMatrix,
    customers: &'a [Customer],
    capacity: i32,
    limits: &'a RouteLimits,
    route_cost: f64,
}

/// What opening a route for the pair costs (`route_cost` plus its length), or `None` when even
/// that route is not allowed: the vehicle cannot carry it, or a window or limit is broken.
fn new_pair_route_cost(pickup: usize, delivery: usize, world: &PairWorld<'_>) -> Option<f64> {
    let route = [pickup, delivery];
    if route_peak_load(&route, world.customers) > world.capacity {
        return None;
    }
    let cost = effective_cost(&route, 0, world.distances, world.customers, world.limits);
    cost.is_finite().then_some(world.route_cost + cost)
}

/// Every way to place a unit, as `(route, first_position, second_position, cost_increase)`
/// sorted by cost: the cheapest position in each route that can take it, and a new route. A
/// unit alone has equal positions.
fn unit_options(
    unit: Unit,
    routes: &[Vec<usize>],
    world: &PairWorld<'_>,
) -> Vec<(usize, usize, usize, f64)> {
    let mut options = Vec::new();
    for (ri, route) in routes.iter().enumerate() {
        let single = std::slice::from_ref(route);
        let found = match unit {
            Unit::One(c) => cheapest_insertion(
                single,
                |_| world.capacity,
                |_| *world.limits,
                c,
                world.distances,
                world.customers,
            )
            .map(|(_, pos, cost)| (ri, pos, pos, cost)),
            Unit::Pair { pickup, delivery } => cheapest_pair_insertion(
                single,
                |_| world.capacity,
                |_| *world.limits,
                pickup,
                delivery,
                world.distances,
                world.customers,
            )
            .map(|(_, i, j, cost)| (ri, i, j, cost)),
        };
        options.extend(found);
    }
    let own_route = match unit {
        Unit::One(c) => new_route_cost(
            c,
            world.distances,
            world.customers,
            world.limits,
            world.route_cost,
        ),
        Unit::Pair { pickup, delivery } => new_pair_route_cost(pickup, delivery, world),
    };
    options.extend(own_route.map(|cost| (NEW_ROUTE, 0, 0, cost)));
    options.sort_by(|a, b| {
        a.3.partial_cmp(&b.3)
            .expect("insertion costs should not be NaN")
    });
    options
}

/// Repairs `solution` of an instance with pickup-delivery pairs: the pairs a destroy operator
/// split are completed (the half still in a route comes out), then the units are placed one at a
/// time -- the cheapest first (`regret` `None`) or the one with the largest regret-`k` first --
/// at the cheapest position that keeps the route within the vehicle's capacity along the way and
/// its windows and limits. A unit that fits nowhere, a new route included, stays unassigned.
fn repair_with_pairs(
    solution: &RoutingSolution,
    world: &PairWorld<'_>,
    regret: Option<usize>,
) -> RoutingSolution {
    let mut sol = solution.clone();
    let mut unassigned = std::mem::take(sol.unassigned_mut());

    // A pair is repaired whole: take the partner of a removed half out of its route.
    for c in unassigned.clone() {
        let partner = world.customers[c]
            .delivery_index()
            .or(world.customers[c].pickup_index());
        if let Some(partner) = partner {
            if !unassigned.contains(&partner) && remove_customer(&mut sol, partner) {
                unassigned.push(partner);
            }
        }
    }
    sol.remove_empty_routes();

    let mut units = units_of(&unassigned, world.customers);
    while !units.is_empty() {
        let mut chosen: Option<(usize, (usize, usize, usize, f64))> = None;
        let mut chosen_score = f64::NEG_INFINITY;
        for (ui, &unit) in units.iter().enumerate() {
            let options = unit_options(unit, sol.routes(), world);
            let Some(&best) = options.first() else {
                continue;
            };
            let score = match regret {
                // The cheapest first: the lower the cost, the higher the score.
                None => -best.3,
                Some(k) => {
                    let sum: f64 = options
                        .iter()
                        .skip(1)
                        .take(k - 1)
                        .map(|o| o.3 - best.3)
                        .sum();
                    // With fewer than k alternatives the unit has little choice: place it early.
                    if options.len() < k {
                        sum + f64::MAX / 2.0
                    } else {
                        sum
                    }
                }
            };
            if score > chosen_score {
                chosen_score = score;
                chosen = Some((ui, best));
            }
        }
        match chosen {
            Some((ui, (route, i, j, _))) => match units.remove(ui) {
                Unit::One(c) => place(&mut sol, route, i, c),
                Unit::Pair { pickup, delivery } => {
                    if route == NEW_ROUTE {
                        sol.routes_mut().push(vec![pickup, delivery]);
                    } else {
                        // The delivery first: `j` counts the route as it stands.
                        sol.routes_mut()[route].insert(j, delivery);
                        sol.routes_mut()[route].insert(i, pickup);
                    }
                }
            },
            None => {
                // Nothing fits, a route of its own included: those customers stay out.
                for unit in units.drain(..) {
                    sol.unassigned_mut().extend(unit.customers());
                }
            }
        }
    }

    sol.recalculate(world.distances, world.customers);
    sol
}

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
    distances: DistanceMatrix,
    customers: Vec<Customer>,
    capacity: i32,
    k: usize,
    route_cost: f64,
}

impl RegretInsertion {
    /// Creates a new regret-k insertion operator with k=2.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>, capacity: i32) -> Self {
        Self {
            distances,
            customers,
            capacity,
            k: 2,
            limits: RouteLimits::NONE,
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
        self.limits = limits;
        self
    }

    /// Creates a regret insertion operator with custom k.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(2);
        self
    }

    /// Computes insertion costs for a customer across all routes, sorted ascending.
    fn sorted_insertion_costs(
        &self,
        routes: &[Vec<usize>],
        customer_id: usize,
    ) -> Vec<(usize, usize, f64)> {
        let depot = 0;
        let constrained = route_checks_needed(&self.customers, &self.limits);
        let soft = has_soft_windows(&self.customers);
        let mut costs = Vec::new();

        for (ri, route) in routes.iter().enumerate() {
            let load: i32 = route.iter().map(|&c| self.customers[c].demand()).sum();
            if load + self.customers[customer_id].demand() > self.capacity {
                continue;
            }

            if soft {
                // The change in the route's distance plus lateness cost, evaluated whole.
                let value =
                    effective_cost(route, depot, &self.distances, &self.customers, &self.limits);
                if !value.is_finite() {
                    continue;
                }
                let best = (0..=route.len())
                    .filter_map(|pos| {
                        soft_insertion_cost(
                            route,
                            value,
                            pos,
                            customer_id,
                            &self.distances,
                            &self.customers,
                            &self.limits,
                        )
                        .map(|cost| (cost, pos))
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0));
                if let Some((cost, pos)) = best {
                    costs.push((ri, pos, cost));
                }
                continue;
            }

            // Find best on-time position in this route
            let mut best_pos = 0;
            let mut best_cost = f64::INFINITY;
            for pos in 0..=route.len() {
                let prev = if pos == 0 { depot } else { route[pos - 1] };
                let next = if pos == route.len() {
                    depot
                } else {
                    route[pos]
                };
                let cost = self.distances.get(prev, customer_id)
                    + self.distances.get(customer_id, next)
                    - self.distances.get(prev, next);
                if cost < best_cost
                    && insertion_on_time(
                        route,
                        pos,
                        customer_id,
                        &self.distances,
                        &self.customers,
                        &self.limits,
                        constrained,
                    )
                {
                    best_cost = cost;
                    best_pos = pos;
                }
            }
            if best_cost.is_finite() {
                costs.push((ri, best_pos, best_cost));
            }
        }
        // A route of its own is one more alternative, at its price.
        if let Some(cost) = new_route_cost(
            customer_id,
            &self.distances,
            &self.customers,
            &self.limits,
            self.route_cost,
        ) {
            costs.push((NEW_ROUTE, 0, cost));
        }

        costs.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .expect("insertion costs should not be NaN")
        });
        costs
    }
}

impl RepairOperator<RoutingSolution> for RegretInsertion {
    fn name(&self) -> &str {
        "regret_insertion"
    }

    fn repair<R: Rng>(&self, solution: &RoutingSolution, _rng: &mut R) -> RoutingSolution {
        if has_pairs(&self.customers) {
            let world = PairWorld {
                distances: &self.distances,
                customers: &self.customers,
                capacity: self.capacity,
                limits: &self.limits,
                route_cost: self.route_cost,
            };
            return repair_with_pairs(solution, &world, Some(self.k));
        }
        let mut sol = solution.clone();
        let mut unassigned = std::mem::take(sol.unassigned_mut());

        while !unassigned.is_empty() {
            let mut best_regret = f64::NEG_INFINITY;
            let mut best_cust_idx = 0;
            let mut best_route = 0;
            let mut best_pos = 0;
            let mut found = false;

            for (ui, &cid) in unassigned.iter().enumerate() {
                let costs = self.sorted_insertion_costs(sol.routes(), cid);

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
                    best_cust_idx = ui;
                    best_route = costs[0].0;
                    best_pos = costs[0].1;
                    found = true;
                }
            }

            if !found {
                // Open a route for the first unassigned customer, or leave it
                // unassigned if even that arrives late.
                let cid = unassigned.remove(0);
                if open_route(cid, &self.distances, &self.customers, &self.limits) {
                    sol.routes_mut().push(vec![cid]);
                } else {
                    sol.unassigned_mut().push(cid);
                }
            } else {
                let cid = unassigned.remove(best_cust_idx);
                place(&mut sol, best_route, best_pos, cid);
            }
        }

        sol.recalculate(&self.distances, &self.customers);
        sol
    }
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
    // -- pickup-delivery pairs ---------------------------------------------

    use crate::models::TimeWindow;

    /// Three pairs on a line (and one off it): 1 -> 2, 3 -> 4 nested inside it, 5 -> 6.
    fn pairs() -> (Vec<Customer>, DistanceMatrix) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 5, 0.0).pickup_of(2),
            Customer::new(2, 3.0, 0.0, 0, 0.0).delivery_of(1),
            Customer::new(3, 1.5, 0.0, 5, 0.0).pickup_of(4),
            Customer::new(4, 2.5, 0.0, 0, 0.0).delivery_of(3),
            Customer::new(5, 0.0, 2.0, 5, 0.0).pickup_of(6),
            Customer::new(6, 0.0, 3.0, 0, 0.0).delivery_of(5),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    /// Every route allowed, every pair whole in one route with the pickup first, and the vehicle
    /// within `capacity` along each route.
    fn assert_pairs_kept(
        sol: &RoutingSolution,
        customers: &[Customer],
        dm: &DistanceMatrix,
        capacity: i32,
    ) {
        for route in sol.routes() {
            assert!(
                route_feasible(route, 0, dm, customers, &RouteLimits::NONE),
                "{route:?}"
            );
            assert!(route_peak_load(route, customers) <= capacity, "{route:?}");
        }
        for pickup in (1..customers.len()).filter(|&c| customers[c].delivery_index().is_some()) {
            let delivery = customers[pickup].delivery_index().expect("a pickup");
            let in_route = |c: usize| sol.routes().iter().position(|r| r.contains(&c));
            let (a, b) = (in_route(pickup), in_route(delivery));
            assert_eq!(a.is_some(), b.is_some(), "a pair is served whole: {sol:?}");
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a, b, "{sol:?}");
            }
        }
    }

    #[test]
    fn pair_instances_place_a_whole_unassigned_pair() {
        let (cust, dm) = pairs();
        for greedy in [true, false] {
            let sol = RoutingSolution::new(vec![vec![1, 2]], vec![3, 4], &cust, &dm);
            let mut rng = u_numflow::random::create_rng(1);
            let repaired = if greedy {
                GreedyInsertion::new(dm.clone(), cust.clone(), 10).repair(&sol, &mut rng)
            } else {
                RegretInsertion::new(dm.clone(), cust.clone(), 10).repair(&sol, &mut rng)
            };
            assert!(repaired.unassigned().is_empty(), "{repaired:?}");
            let total: usize = repaired.routes().iter().map(Vec::len).sum();
            assert_eq!(total, 4);
            assert_pairs_kept(&repaired, &cust, &dm, 10);
        }
    }

    #[test]
    fn a_half_left_in_a_route_is_taken_out_with_its_partner_and_returns() {
        let (cust, dm) = pairs();
        // The pickup 3 was removed; its delivery 4 is still in the route.
        let sol = RoutingSolution::new(vec![vec![1, 4, 2, 5, 6]], vec![3], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(1);
        for repaired in [
            GreedyInsertion::new(dm.clone(), cust.clone(), 10).repair(&sol, &mut rng),
            RegretInsertion::new(dm.clone(), cust.clone(), 10).repair(&sol, &mut rng),
        ] {
            assert!(repaired.unassigned().is_empty(), "{repaired:?}");
            let total: usize = repaired.routes().iter().map(Vec::len).sum();
            assert_eq!(total, 6);
            assert_pairs_kept(&repaired, &cust, &dm, 10);
        }
    }

    #[test]
    fn a_pair_is_not_nested_where_the_vehicle_cannot_carry_both() {
        let (cust, dm) = pairs();
        // Nested in 1 -> 2 is the cheapest placement of 3 -> 4, but it carries 10 on a vehicle
        // of 5; the pairs have to follow one another or take another route.
        let sol = RoutingSolution::new(vec![vec![1, 2]], vec![3, 4], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(1);
        for repaired in [
            GreedyInsertion::new(dm.clone(), cust.clone(), 5).repair(&sol, &mut rng),
            RegretInsertion::new(dm.clone(), cust.clone(), 5).repair(&sol, &mut rng),
        ] {
            assert!(repaired.unassigned().is_empty(), "{repaired:?}");
            assert_pairs_kept(&repaired, &cust, &dm, 5);
        }
    }

    #[test]
    fn a_pair_no_route_can_serve_stays_unassigned_whole() {
        let (mut cust, _) = pairs();
        // The delivery of 5 -> 6 closes before the vehicle can get there.
        cust[6] = Customer::new(6, 0.0, 3.0, 0, 0.0)
            .delivery_of(5)
            .with_time_window(TimeWindow::new(0.0, 0.5).expect("window"));
        let dm = DistanceMatrix::from_customers(&cust);
        let sol = RoutingSolution::new(vec![vec![1, 2]], vec![5, 6], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(1);
        let repaired = GreedyInsertion::new(dm.clone(), cust.clone(), 10).repair(&sol, &mut rng);
        let mut left = repaired.unassigned().to_vec();
        left.sort_unstable();
        assert_eq!(left, vec![5, 6]);
        assert_eq!(repaired.routes(), &[vec![1, 2]]);
    }

    #[test]
    fn a_pair_the_vehicle_cannot_carry_even_alone_stays_unassigned() {
        let (cust, dm) = pairs();
        // The pickup loads 5 on a vehicle of 4.
        let sol = RoutingSolution::new(vec![], vec![1, 2], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(1);
        let repaired = GreedyInsertion::new(dm.clone(), cust.clone(), 4).repair(&sol, &mut rng);
        let mut left = repaired.unassigned().to_vec();
        left.sort_unstable();
        assert_eq!(left, vec![1, 2]);
        assert_eq!(repaired.num_routes(), 0);
    }

    #[test]
    fn a_new_route_is_priced_and_an_empty_one_is_no_way_round_the_price() {
        let (cust, dm) = pairs();
        // A route costs 1000 to open. Placing 1 -> 2 in the route of 5 -> 6 adds far less than
        // that, and the empty route left by a destroy operator is no cheaper way to open one.
        let sol = RoutingSolution::new(vec![vec![], vec![5, 6]], vec![1, 2], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(1);
        for repaired in [
            GreedyInsertion::new(dm.clone(), cust.clone(), 10)
                .with_route_cost(1000.0)
                .repair(&sol, &mut rng),
            RegretInsertion::new(dm.clone(), cust.clone(), 10)
                .with_route_cost(1000.0)
                .repair(&sol, &mut rng),
        ] {
            assert_eq!(repaired.num_routes(), 1, "{repaired:?}");
            assert!(repaired.unassigned().is_empty());
            assert_pairs_kept(&repaired, &cust, &dm, 10);
        }
    }

    #[test]
    fn instances_without_pairs_repair_as_before() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1]], vec![2, 3, 4], &cust, &dm);
        let mut rng = u_numflow::random::create_rng(42);
        let repaired = GreedyInsertion::new(dm.clone(), cust.clone(), 100).repair(&sol, &mut rng);
        // One route along the line, in either direction: 0 -> 4 -> 0.
        assert_eq!(repaired.num_routes(), 1);
        assert!((repaired.total_distance() - 8.0).abs() < 1e-10);
    }
}
