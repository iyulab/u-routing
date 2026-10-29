//! Giving each route of a plan the vehicle that carries it.
//!
//! A fleet may mix vehicle types -- different capacities, fixed costs, costs
//! per distance and limits -- each available a limited number of times or
//! without limit. Given the routes of a plan, [`assign_types`] picks for every
//! route one vehicle type so that as many customers as possible stay on a
//! route and, among those assignments, the plan costs least.
//!
//! # Algorithm
//!
//! Route-to-vehicle assignment is a transportation problem: route `r` may take
//! type `t` when the type carries its load and keeps its limits, at the type's
//! `fixed_cost + cost_per_distance × distance`; type `t` can be taken at most
//! `count` times. It is solved exactly as a rectangular assignment (the
//! Hungarian method of Kuhn 1955 in the potentials form of Jonker & Volgenant
//! 1987, O(n²m)) over the vehicles the types stand for, plus one "left out"
//! column per route whose price is far above any real cost and grows with the
//! customers on the route -- so a route is left out only when no assignment
//! carries it, and then the lightest one goes first.
//!
//! # References
//!
//! - Kuhn, H. W. (1955). "The Hungarian method for the assignment problem",
//!   *Naval Research Logistics Quarterly* 2, 83-97.
//! - Golden, B., Assad, A., Levy, L. & Gheysens, F. (1984). "The fleet size and
//!   mix vehicle routing problem", *Computers & Operations Research* 11(1),
//!   49-66.

use crate::evaluation::RouteLimits;
use crate::models::Vehicle;

/// A kind of vehicle in a fleet and how many of it there are.
///
/// # Examples
///
/// ```
/// use u_routing::assignment::VehicleType;
/// use u_routing::models::Vehicle;
///
/// let vans = VehicleType::new(Vehicle::new(0, 50), Some(3));
/// assert_eq!(vans.count(), Some(3));
/// let unlimited = VehicleType::new(Vehicle::new(1, 200), None);
/// assert_eq!(unlimited.count(), None);
/// ```
#[derive(Debug, Clone)]
pub struct VehicleType {
    vehicle: Vehicle,
    count: Option<usize>,
}

impl VehicleType {
    /// A type of `count` vehicles shaped like `vehicle`; `None` is as many as
    /// the plan needs.
    pub fn new(vehicle: Vehicle, count: Option<usize>) -> Self {
        Self { vehicle, count }
    }

    /// The vehicle every one of this type is.
    pub fn vehicle(&self) -> &Vehicle {
        &self.vehicle
    }

    /// How many of this type there are; `None` is without limit.
    pub fn count(&self) -> Option<usize> {
        self.count
    }

    /// Whether one vehicle of this type can drive a route asking for `demand`.
    pub fn carries(&self, demand: &RouteDemand) -> bool {
        demand.load <= self.vehicle.capacity()
            && RouteLimits::of(&self.vehicle).allow(demand.distance, demand.duration)
    }
}

/// What a route asks of its vehicle: the load it carries, the distance it
/// travels and when it is back at the depot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteDemand {
    /// Total demand of the customers on the route.
    pub load: i32,
    /// Total distance, depot to depot.
    pub distance: f64,
    /// Return time to the depot on the route's own clock.
    pub duration: f64,
}

/// The cheapest of `types` that carries `demand`, ignoring how many of each
/// there are: `(index, cost)`, the first among equals. `None` when no type
/// carries it.
pub fn cheapest_type(types: &[VehicleType], demand: &RouteDemand) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, t) in types.iter().enumerate() {
        if !t.carries(demand) {
            continue;
        }
        let cost = t.vehicle.route_cost(demand.distance);
        if best.is_none_or(|(_, c)| cost < c) {
            best = Some((i, cost));
        }
    }
    best
}

/// Gives every route a vehicle type, or leaves it out.
///
/// Returns, per route, the index in `types` of the type that drives it, or
/// `None` for a route no assignment can carry. Among all assignments the one
/// that serves the most customers wins (`customers[r]` counts the customers of
/// route `r`), and among those the one with the least total cost
/// (`fixed_cost + cost_per_distance × distance` summed over routes); equal
/// costs resolve to the lower type index.
///
/// # Complexity
/// O(R² · (V + R)) for R routes and V = Σ min(count, R) vehicles.
pub fn assign_types(
    routes: &[RouteDemand],
    customers: &[usize],
    types: &[VehicleType],
) -> Vec<Option<usize>> {
    assert_eq!(routes.len(), customers.len());
    let n = routes.len();
    if n == 0 {
        return Vec::new();
    }
    // One column per vehicle a route could be given; more than `n` of a type
    // cannot be used.
    let mut column_type: Vec<usize> = Vec::new();
    for (t, ty) in types.iter().enumerate() {
        let copies = ty.count.map_or(n, |c| c.min(n));
        column_type.extend(std::iter::repeat_n(t, copies));
    }
    let vehicles = column_type.len();
    let cols = vehicles + n;

    let real = |r: usize, c: usize| -> Option<f64> {
        let ty = &types[column_type[c]];
        ty.carries(&routes[r])
            .then(|| ty.vehicle.route_cost(routes[r].distance))
    };
    // Far above what any assignment of real costs adds up to.
    let mut scale = 1.0_f64;
    for r in 0..n {
        let row_max = (0..vehicles)
            .filter_map(|c| real(r, c))
            .fold(0.0_f64, f64::max);
        scale += row_max;
    }
    let most = customers.iter().copied().max().unwrap_or(0).max(1) as f64;
    let forbidden = 2.0 * scale * (most + 1.0);
    let mut cost = vec![forbidden; n * cols];
    for r in 0..n {
        for c in 0..vehicles {
            if let Some(v) = real(r, c) {
                cost[r * cols + c] = v;
            }
        }
        cost[r * cols + vehicles + r] = scale * customers[r].max(1) as f64;
    }
    let owner = hungarian(&cost, n, cols);
    let mut out = vec![None; n];
    for (c, row) in owner.iter().enumerate() {
        if let Some(r) = *row {
            if c < vehicles {
                out[r] = Some(column_type[c]);
            }
        }
    }
    out
}

/// Minimum-cost assignment of `n` rows to distinct columns of `m ≥ n`, over a
/// row-major `n × m` matrix; `owner[c]` is the row given column `c`.
fn hungarian(cost: &[f64], n: usize, m: usize) -> Vec<Option<usize>> {
    debug_assert!(n <= m);
    // 1-based, as in the classic formulation; index 0 is a sentinel.
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; m + 1];
    let mut matched = vec![0_usize; m + 1];
    let mut way = vec![0_usize; m + 1];
    for i in 1..=n {
        matched[0] = i;
        let mut j0 = 0;
        let mut min_to = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = matched[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                if cur < min_to[j] {
                    min_to[j] = cur;
                    way[j] = j0;
                }
                if min_to[j] < delta {
                    delta = min_to[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[matched[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_to[j] -= delta;
                }
            }
            j0 = j1;
            if matched[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            matched[j0] = matched[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    (1..=m)
        .map(|j| (matched[j] != 0).then(|| matched[j] - 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demand(load: i32, distance: f64) -> RouteDemand {
        RouteDemand {
            load,
            distance,
            duration: distance,
        }
    }

    fn truck(id: usize, capacity: i32, fixed: f64, per_distance: f64) -> Vehicle {
        Vehicle::new(id, capacity)
            .with_fixed_cost(fixed)
            .with_cost_per_distance(per_distance)
    }

    #[test]
    fn one_unlimited_type_takes_every_route() {
        let types = [VehicleType::new(Vehicle::new(0, 10), None)];
        let routes = [demand(4, 3.0), demand(10, 5.0), demand(1, 1.0)];
        let given = assign_types(&routes, &[2, 3, 1], &types);
        assert_eq!(given, vec![Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn a_route_nothing_carries_is_left_out() {
        let types = [VehicleType::new(Vehicle::new(0, 10), None)];
        let routes = [demand(4, 3.0), demand(11, 5.0)];
        assert_eq!(assign_types(&routes, &[2, 3], &types), vec![Some(0), None]);
    }

    #[test]
    fn the_cheapest_carrying_type_wins_without_counts() {
        let types = [
            VehicleType::new(truck(0, 100, 50.0, 1.0), None),
            VehicleType::new(truck(1, 20, 10.0, 1.0), None),
            VehicleType::new(truck(2, 20, 10.0, 2.0), None),
        ];
        let routes = [demand(15, 8.0), demand(60, 8.0)];
        assert_eq!(
            assign_types(&routes, &[3, 5], &types),
            vec![Some(1), Some(0)]
        );
        assert_eq!(cheapest_type(&types, &routes[0]), Some((1, 18.0)));
        assert_eq!(cheapest_type(&types, &demand(101, 1.0)), None);
    }

    #[test]
    fn a_limit_of_the_type_keeps_it_off_a_long_route() {
        let types = [
            VehicleType::new(truck(0, 50, 0.0, 1.0).with_max_distance(10.0), None),
            VehicleType::new(truck(1, 50, 100.0, 1.0), None),
        ];
        let routes = [demand(5, 9.0), demand(5, 11.0)];
        assert_eq!(
            assign_types(&routes, &[1, 1], &types),
            vec![Some(0), Some(1)]
        );
    }

    #[test]
    fn counts_send_the_rest_to_the_dearer_type() {
        let types = [
            VehicleType::new(truck(0, 10, 0.0, 1.0), Some(1)),
            VehicleType::new(truck(1, 10, 0.0, 3.0), None),
        ];
        // Both routes fit either type; the cheap one is taken by the longer
        // route, where the saving is greater.
        let routes = [demand(5, 2.0), demand(5, 9.0)];
        assert_eq!(
            assign_types(&routes, &[1, 1], &types),
            vec![Some(1), Some(0)]
        );
    }

    #[test]
    fn when_the_fleet_is_short_the_lighter_route_is_left_out() {
        let types = [VehicleType::new(Vehicle::new(0, 10), Some(1))];
        let routes = [demand(5, 2.0), demand(9, 9.0)];
        // The second route has more customers: it keeps the only vehicle.
        assert_eq!(assign_types(&routes, &[2, 4], &types), vec![None, Some(0)]);
    }

    /// The assignment matches an exhaustive search over small random cases:
    /// the same number of customers served, and the same cost.
    #[test]
    fn it_matches_exhaustive_search() {
        let mut state = 20261029u64;
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        for case in 0..300 {
            let t = 1 + next(3) as usize;
            let n = 1 + next(6) as usize;
            let types: Vec<VehicleType> = (0..t)
                .map(|i| {
                    let mut v = truck(
                        i,
                        5 + next(20) as i32,
                        next(30) as f64,
                        (1 + next(4)) as f64 / 2.0,
                    );
                    if next(3) == 0 {
                        v = v.with_max_distance(5.0 + next(20) as f64);
                    }
                    let count = if next(2) == 0 {
                        None
                    } else {
                        Some(next(4) as usize)
                    };
                    VehicleType::new(v, count)
                })
                .collect();
            let routes: Vec<RouteDemand> = (0..n)
                .map(|_| demand(1 + next(24) as i32, 1.0 + next(24) as f64))
                .collect();
            let customers: Vec<usize> = (0..n).map(|_| 1 + next(5) as usize).collect();

            let got = assign_types(&routes, &customers, &types);
            let score = |plan: &[Option<usize>]| -> Option<(usize, f64)> {
                let mut used = vec![0usize; t];
                let (mut served, mut cost) = (0, 0.0);
                for (r, slot) in plan.iter().enumerate() {
                    if let Some(k) = *slot {
                        if !types[k].carries(&routes[r]) {
                            return None;
                        }
                        used[k] += 1;
                        if types[k].count().is_some_and(|c| used[k] > c) {
                            return None;
                        }
                        served += customers[r];
                        cost += types[k].vehicle().route_cost(routes[r].distance);
                    }
                }
                Some((served, cost))
            };
            let (served, cost) = score(&got).expect("the answer is itself valid");

            let mut best: (usize, f64) = (0, f64::INFINITY);
            let options = t + 1;
            for code in 0..options.pow(n as u32) {
                let mut rest = code;
                let plan: Vec<Option<usize>> = (0..n)
                    .map(|_| {
                        let pick = rest % options;
                        rest /= options;
                        (pick < t).then_some(pick)
                    })
                    .collect();
                if let Some((s, c)) = score(&plan) {
                    if s > best.0 || (s == best.0 && c < best.1) {
                        best = (s, c);
                    }
                }
            }
            assert_eq!(served, best.0, "case {case}: customers served");
            if best.1.is_finite() {
                assert!(
                    (cost - best.1).abs() < 1e-9,
                    "case {case}: {cost} vs {}",
                    best.1
                );
            }
        }
    }
}
