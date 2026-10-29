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
//! O(R² × L²) for the first scan and O(R × L²) per move taken, R routes of
//! up to L customers: a move changes two routes, and only the pairs of routes
//! with one of them in it are searched again.
//!
//! # Reference
//!
//! Or, I. (1976). "Traveling Salesman-Type Combinatorial Problems and Their
//! Relation to the Logistics of Blood Banking". PhD thesis.

use super::gain_floor;
use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, route_feasible, RouteLimits};
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
    let mut search = Relocations::new(&routes, customers, distances, vehicle);

    // Best improvement: the best move of every pair of routes is kept, and a
    // move changes two routes, so only the pairs with one of them in it are
    // searched again. The move taken is the one a full scan would take.
    while let Some(mv) = search.best() {
        let customer_id = routes[mv.from_route][mv.from_pos];
        routes[mv.from_route].remove(mv.from_pos);
        routes[mv.to_route].insert(mv.to_pos, customer_id);
        search.moved(&routes, mv.from_route, mv.to_route);
    }

    // Rebuild solution
    rebuild_solution(&routes, solution, distances, customers, vehicle)
}

/// The best improving relocation between every ordered pair of routes.
struct Relocations<'a> {
    customers: &'a [Customer],
    distances: &'a DistanceMatrix,
    vehicle: &'a Vehicle,
    limits: RouteLimits,
    constrained: bool,
    /// The least gain that counts (see [`gain_floor`]).
    floor: f64,
    loads: Vec<i32>,
    /// `pairs[from * n + to]`: the best move from route `from` into `to`.
    pairs: Vec<Option<RelocateMove>>,
}

impl<'a> Relocations<'a> {
    fn new(
        routes: &[Vec<usize>],
        customers: &'a [Customer],
        distances: &'a DistanceMatrix,
        vehicle: &'a Vehicle,
    ) -> Self {
        let limits = RouteLimits::of(vehicle);
        let n = routes.len();
        let mut search = Self {
            customers,
            distances,
            vehicle,
            constrained: route_checks_needed(customers, &limits),
            floor: gain_floor(distances),
            limits,
            loads: routes.iter().map(|r| Self::load(customers, r)).collect(),
            pairs: vec![None; n * n],
        };
        for from in 0..n {
            for to in 0..n {
                search.pairs[from * n + to] = search.best_between(routes, from, to);
            }
        }
        search
    }

    fn load(customers: &[Customer], route: &[usize]) -> i32 {
        route.iter().map(|&c| customers[c].demand()).sum()
    }

    /// The improving move a full scan in the order (from route, from
    /// position, to route, to position) would take first.
    fn best(&self) -> Option<RelocateMove> {
        self.pairs
            .iter()
            .flatten()
            .min_by(|a, b| {
                a.delta.total_cmp(&b.delta).then(
                    (a.from_route, a.from_pos, a.to_route, a.to_pos).cmp(&(
                        b.from_route,
                        b.from_pos,
                        b.to_route,
                        b.to_pos,
                    )),
                )
            })
            .cloned()
    }

    /// Searches again every pair with route `a` or `b` in it.
    fn moved(&mut self, routes: &[Vec<usize>], a: usize, b: usize) {
        let n = routes.len();
        for r in [a, b] {
            self.loads[r] = Self::load(self.customers, &routes[r]);
        }
        for other in 0..n {
            for r in [a, b] {
                self.pairs[r * n + other] = self.best_between(routes, r, other);
                self.pairs[other * n + r] = self.best_between(routes, other, r);
            }
        }
    }

    /// The best improving, feasible move of a customer of `from_r` into
    /// `to_r` (the first in position order among equals).
    fn best_between(
        &self,
        routes: &[Vec<usize>],
        from_r: usize,
        to_r: usize,
    ) -> Option<RelocateMove> {
        if from_r == to_r {
            return None;
        }
        let depot = self.vehicle.depot_id();
        let distances = self.distances;
        // A route's fixed cost, in distance units: saved when a move empties
        // it, paid when one fills an empty route.
        let fixed = self.vehicle.fixed_cost_in_distance();
        let to_route = &routes[to_r];
        // Improving moves, cheapest first: the first feasible one is the best
        // (the first in position order among equals).
        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
        let mut cheapest: Option<(f64, usize, usize)> = None;
        for from_pos in 0..routes[from_r].len() {
            let cid = routes[from_r][from_pos];
            if self.loads[to_r] + self.customers[cid].demand() > self.vehicle.capacity() {
                continue;
            }
            let removal_delta = removal_cost(&routes[from_r], from_pos, depot, distances)
                - if routes[from_r].len() == 1 {
                    fixed
                } else {
                    0.0
                };
            for to_pos in 0..=to_route.len() {
                let insertion_delta = insertion_cost(to_route, to_pos, cid, depot, distances)
                    + if to_route.is_empty() { fixed } else { 0.0 };
                let delta = removal_delta + insertion_delta;
                if delta < -self.floor {
                    if self.constrained {
                        candidates.push((delta, from_pos, to_pos));
                    } else if cheapest.is_none_or(|(d, _, _)| delta < d) {
                        // Every move is feasible: the cheapest, the first in
                        // position order among equals, without collecting
                        // and sorting them all.
                        cheapest = Some((delta, from_pos, to_pos));
                    }
                }
            }
        }
        candidates.extend(cheapest);
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best: Option<RelocateMove> = None;
        for (delta, from_pos, to_pos) in candidates {
            let cid = routes[from_r][from_pos];
            // Only the receiving route's timing changes; the giving route
            // only gets shorter.
            if self.constrained {
                let mut candidate = to_route.clone();
                candidate.insert(to_pos, cid);
                if !route_feasible(&candidate, depot, distances, self.customers, &self.limits) {
                    continue;
                }
            }
            best = Some(RelocateMove {
                from_route: from_r,
                from_pos,
                to_route: to_r,
                to_pos,
                delta,
            });
            break;
        }
        best
    }
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

    solution.price(|_| vehicle);
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
