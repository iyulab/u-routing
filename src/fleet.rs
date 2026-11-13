//! Planning within a fixed fleet.
//!
//! The constructive and metaheuristic methods open as many routes as the
//! demand needs. When the fleet is fixed at `K` vehicles, [`limit_routes`]
//! brings such a plan down to at most `K` routes: it empties the routes whose
//! removal loses the least demand, moves their customers into the routes that
//! remain wherever capacity and time windows allow, and reports the customers
//! no remaining route can take. A pickup-delivery pair is emptied and reinserted as one unit --
//! the pickup before its delivery in a route that keeps both -- or goes unserved whole.
//!
//! # Algorithm
//!
//! Route elimination with cheapest feasible reinsertion, the route-reduction
//! step of route-minimisation heuristics (Bräysy & Gendreau 2005, §3; Nagata
//! & Bräysy 2009). Removing the route with the smallest load first keeps the
//! most demand served when not every customer fits.
//!
//! # References
//!
//! - Bräysy, O. & Gendreau, M. (2005). "Vehicle Routing Problem with Time
//!   Windows, Part I: Route Construction and Local Search Algorithms",
//!   *Transportation Science* 39(1), 104-118.
//! - Nagata, Y. & Bräysy, O. (2009). "A powerful route minimization heuristic
//!   for the vehicle routing problem with time windows", *Operations Research
//!   Letters* 37(5), 333-338.

use crate::distance::DistanceMatrix;
use crate::evaluation::{
    effective_cost, has_soft_windows, route_checks_needed, route_feasible, route_peak_load,
    RouteLimits,
};
use crate::models::{Customer, Vehicle};

/// Whether inserting `customer_id` at `pos` of `route` keeps every customer
/// of the route on time. Always true when `windowed` is false.
pub(crate) fn insertion_on_time(
    route: &[usize],
    pos: usize,
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
    constrained: bool,
) -> bool {
    if !constrained {
        return true;
    }
    let mut candidate = Vec::with_capacity(route.len() + 1);
    candidate.extend_from_slice(&route[..pos]);
    candidate.push(customer_id);
    candidate.extend_from_slice(&route[pos..]);
    route_feasible(&candidate, 0, distances, customers, limits)
}

/// What inserting `customer_id` at `pos` of `route` adds to the route's priced distance (distance
/// plus the cost of lateness at soft windows), where `route_value` is that of `route` as it stands;
/// `None` when the route would no longer be allowed. For a problem with a soft window, where
/// the change in distance of the edges alone says nothing of the lateness a detour causes.
pub(crate) fn soft_insertion_cost(
    route: &[usize],
    route_value: f64,
    pos: usize,
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
) -> Option<f64> {
    let mut candidate = Vec::with_capacity(route.len() + 1);
    candidate.extend_from_slice(&route[..pos]);
    candidate.push(customer_id);
    candidate.extend_from_slice(&route[pos..]);
    let with = effective_cost(&candidate, 0, distances, customers, limits);
    with.is_finite().then_some(with - route_value)
}

/// The cheapest position for `customer_id` across `routes`, as
/// `(route_index, position, cost_increase)`, among positions that keep the
/// route within `capacity_of(route_index)` and every customer on time.
/// `None` when no route can take the customer.
///
/// # Complexity
/// O(R·n) positions, each checked in O(n) when customers carry windows.
pub fn cheapest_insertion(
    routes: &[Vec<usize>],
    capacity_of: impl Fn(usize) -> i32,
    limits_of: impl Fn(usize) -> RouteLimits,
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> Option<(usize, usize, f64)> {
    let depot = 0;
    let demand = customers[customer_id].demand();
    let mut best: Option<(usize, usize, f64)> = None;
    let soft = has_soft_windows(customers);

    for (ri, route) in routes.iter().enumerate() {
        let load: i32 = route.iter().map(|&c| customers[c].demand()).sum();
        if load + demand > capacity_of(ri) {
            continue;
        }
        let limits = limits_of(ri);
        if soft {
            // The change in the route's distance plus lateness cost, evaluated whole.
            let value = effective_cost(route, depot, distances, customers, &limits);
            if !value.is_finite() {
                continue;
            }
            for pos in 0..=route.len() {
                let cost = soft_insertion_cost(
                    route,
                    value,
                    pos,
                    customer_id,
                    distances,
                    customers,
                    &limits,
                );
                if let Some(cost) = cost {
                    if best.as_ref().is_none_or(|b| cost < b.2) {
                        best = Some((ri, pos, cost));
                    }
                }
            }
            continue;
        }
        let constrained = route_checks_needed(customers, &limits);
        for pos in 0..=route.len() {
            let prev = if pos == 0 { depot } else { route[pos - 1] };
            let next = if pos == route.len() {
                depot
            } else {
                route[pos]
            };
            let cost = distances.get(prev, customer_id) + distances.get(customer_id, next)
                - distances.get(prev, next);
            if best.as_ref().is_none_or(|b| cost < b.2)
                && insertion_on_time(
                    route,
                    pos,
                    customer_id,
                    distances,
                    customers,
                    &limits,
                    constrained,
                )
            {
                best = Some((ri, pos, cost));
            }
        }
    }
    best
}

/// The cheapest placement of the pickup-delivery pair (`pickup`, `delivery`) across `routes`, as
/// `(route_index, pickup_position, delivery_position, cost_increase)`: the pickup is inserted
/// first and the delivery after it (positions in the route as it stands, `pickup_position <=
/// delivery_position`), so that the route stays within `capacity_of(route_index)` along the way
/// and keeps every window, limit and pair in order. The increase is that of the route's priced
/// distance. `None` when no route can take the pair.
///
/// # Complexity
/// O(R·n²) placements, each priced in O(n).
pub(crate) fn cheapest_pair_insertion(
    routes: &[Vec<usize>],
    capacity_of: impl Fn(usize) -> i32,
    limits_of: impl Fn(usize) -> RouteLimits,
    pickup: usize,
    delivery: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> Option<(usize, usize, usize, f64)> {
    let depot = 0;
    let mut best: Option<(usize, usize, usize, f64)> = None;
    for (ri, route) in routes.iter().enumerate() {
        let limits = limits_of(ri);
        let value = effective_cost(route, depot, distances, customers, &limits);
        if !value.is_finite() {
            continue;
        }
        let capacity = capacity_of(ri);
        for i in 0..=route.len() {
            for j in i..=route.len() {
                let mut candidate = Vec::with_capacity(route.len() + 2);
                candidate.extend_from_slice(&route[..i]);
                candidate.push(pickup);
                candidate.extend_from_slice(&route[i..j]);
                candidate.push(delivery);
                candidate.extend_from_slice(&route[j..]);
                if route_peak_load(&candidate, customers) > capacity {
                    continue;
                }
                // A placement the route does not allow costs infinity.
                let with = effective_cost(&candidate, depot, distances, customers, &limits);
                if with.is_finite() {
                    let cost = with - value;
                    if best.as_ref().is_none_or(|b| cost < b.3) {
                        best = Some((ri, i, j, cost));
                    }
                }
            }
        }
    }
    best
}

/// What is left to place after a route is emptied: a customer on its own, or a pickup-delivery
/// pair, which moves as one.
#[derive(Debug, Clone, Copy)]
enum Unit {
    One(usize),
    Pair { pickup: usize, delivery: usize },
}

impl Unit {
    /// The customers of the unit.
    fn customers(self) -> impl Iterator<Item = usize> {
        let (a, b) = match self {
            Unit::One(c) => (c, None),
            Unit::Pair { pickup, delivery } => (pickup, Some(delivery)),
        };
        std::iter::once(a).chain(b)
    }
}

/// The units of the customers of an emptied route: a pickup with its delivery when both are
/// there, every other customer alone.
fn units_of(pending: &[usize], customers: &[Customer]) -> Vec<Unit> {
    let has = |c: usize| pending.contains(&c);
    pending
        .iter()
        .filter_map(|&c| {
            let customer = &customers[c];
            match (customer.delivery_index(), customer.pickup_index()) {
                (Some(delivery), _) if has(delivery) => Some(Unit::Pair {
                    pickup: c,
                    delivery,
                }),
                // The delivery is placed with its pickup.
                (None, Some(pickup)) if has(pickup) => None,
                _ => Some(Unit::One(c)),
            }
        })
        .collect()
}

/// A plan brought within a route limit by [`limit_routes`].
#[derive(Debug, Clone, PartialEq)]
pub struct LimitedPlan {
    /// At most `max_routes` routes of customer indices (depot excluded).
    pub routes: Vec<Vec<usize>>,
    /// The vehicle of each route in `routes` -- its index in the fleet the
    /// plan was limited with -- in the same order.
    pub vehicles: Vec<usize>,
    /// Customers of eliminated routes that no remaining route could take,
    /// in ascending index order.
    pub unserved: Vec<usize>,
}

/// Brings `routes` down to at most `max_routes`.
///
/// While there are too many routes, the one with the smallest load (then the
/// fewest customers) is emptied and its customers are reinserted, cheapest
/// first, into the routes that remain -- never past a route's capacity and
/// never making a customer late. A customer that fits nowhere is `unserved`.
/// A plan already within the limit is returned unchanged.
///
/// `vehicles[i]` is the index in `fleet` of the vehicle that drives
/// `routes[i]`, so a fleet of mixed vehicles keeps each route's own capacity
/// and distance/duration limits.
///
/// # Panics
/// If `vehicles` and `routes` differ in length, or a vehicle index is outside
/// `fleet` (a caller bug).
///
/// # Examples
///
/// ```
/// use u_routing::models::Vehicle;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::fleet::limit_routes;
/// use u_routing::models::Customer;
///
/// // Three customers of demand 10 on three routes; two vehicles of 20.
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 0.0, 5.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let plan = limit_routes(vec![vec![1], vec![2], vec![3]], vec![0; 3], &[Vehicle::new(0, 20)], 2, &dm, &customers);
/// assert_eq!(plan.routes.len(), 2);
/// assert!(plan.unserved.is_empty()); // one route took a second customer
/// ```
pub fn limit_routes(
    routes: Vec<Vec<usize>>,
    vehicles: Vec<usize>,
    fleet: &[Vehicle],
    max_routes: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> LimitedPlan {
    assert_eq!(
        routes.len(),
        vehicles.len(),
        "one vehicle per route is required"
    );
    assert!(
        vehicles.iter().all(|&v| v < fleet.len()),
        "every route's vehicle is in the fleet"
    );
    let mut routes = routes;
    let mut vehicles = vehicles;
    let mut unserved = Vec::new();

    while routes.len() > max_routes {
        let load = |r: &Vec<usize>| -> i32 { r.iter().map(|&c| customers[c].demand()).sum() };
        let victim = (0..routes.len())
            .min_by_key(|&i| (load(&routes[i]), routes[i].len()))
            .expect("more routes than the limit, so at least one");
        let pending = routes.remove(victim);
        vehicles.remove(victim);
        let mut units = units_of(&pending, customers);

        // Cheapest insertion first -- a pair as one unit, the pickup before its delivery -- until
        // nothing pending fits anywhere.
        loop {
            let best = units
                .iter()
                .enumerate()
                .filter_map(|(ui, &unit)| {
                    let capacity_of = |ri: usize| fleet[vehicles[ri]].capacity();
                    let limits_of = |ri: usize| RouteLimits::of(&fleet[vehicles[ri]]);
                    match unit {
                        Unit::One(cid) => cheapest_insertion(
                            &routes,
                            capacity_of,
                            limits_of,
                            cid,
                            distances,
                            customers,
                        )
                        .map(|(ri, pos, cost)| (ui, ri, pos, pos, cost)),
                        Unit::Pair { pickup, delivery } => cheapest_pair_insertion(
                            &routes,
                            capacity_of,
                            limits_of,
                            pickup,
                            delivery,
                            distances,
                            customers,
                        )
                        .map(|(ri, i, j, cost)| (ui, ri, i, j, cost)),
                    }
                })
                .min_by(|a, b| a.4.total_cmp(&b.4));
            match best {
                Some((ui, ri, i, j, _)) => match units.remove(ui) {
                    Unit::One(cid) => routes[ri].insert(i, cid),
                    Unit::Pair { pickup, delivery } => {
                        // The delivery first: `j` counts the route as it stands.
                        routes[ri].insert(j, delivery);
                        routes[ri].insert(i, pickup);
                    }
                },
                None => break,
            }
        }
        unserved.extend(units.into_iter().flat_map(Unit::customers));
    }

    unserved.sort_unstable();
    LimitedPlan {
        routes,
        vehicles,
        unserved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    /// The #301 report: six customers of demand 10 on a ring, vehicles of 20.
    fn six_on_a_ring() -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for i in 1..=6 {
            let a = 2.0 * std::f64::consts::PI * i as f64 / 6.0;
            customers.push(Customer::new(i, 10.0 * a.cos(), 10.0 * a.sin(), 10, 0.0));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn a_plan_within_the_limit_is_unchanged() {
        let (customers, dm) = six_on_a_ring();
        let routes = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let plan = limit_routes(
            routes.clone(),
            vec![0; 3],
            &[Vehicle::new(0, 20)],
            3,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes, routes);
        assert!(plan.unserved.is_empty());
    }

    #[test]
    fn a_full_fleet_leaves_one_routes_customers_unserved() {
        let (customers, dm) = six_on_a_ring();
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            vec![0; 3],
            &[Vehicle::new(0, 20)],
            2,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes.len(), 2);
        assert_eq!(plan.unserved.len(), 2, "{plan:?}");
        let served: usize = plan.routes.iter().map(Vec::len).sum();
        assert_eq!(served + plan.unserved.len(), 6);
    }

    #[test]
    fn spare_capacity_absorbs_an_eliminated_route() {
        let (customers, dm) = six_on_a_ring();
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            vec![0; 3],
            &[Vehicle::new(0, 30)],
            2,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes.len(), 2);
        assert!(plan.unserved.is_empty(), "{plan:?}");
        for r in &plan.routes {
            let load: i32 = r.iter().map(|&c| customers[c].demand()).sum();
            assert!(load <= 30);
        }
    }

    #[test]
    fn the_lightest_route_goes_first() {
        let (mut customers, _) = six_on_a_ring();
        customers[6] = Customer::new(6, customers[6].x(), customers[6].y(), 1, 0.0);
        let dm = DistanceMatrix::from_customers(&customers);
        // Route [6] carries 1 unit; it is the one emptied, and it fits elsewhere.
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5], vec![6]],
            vec![0; 4],
            &[Vehicle::new(0, 21)],
            3,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes.len(), 3);
        assert!(plan.unserved.is_empty(), "{plan:?}");
    }

    /// Three pairs (pickup, delivery) on three rays from the depot; every delivery has to be
    /// reached by `latest`, which only the first pair of a route can do.
    fn three_pairs(latest: f64) -> (Vec<Customer>, DistanceMatrix) {
        let window = TimeWindow::new(0.0, latest).expect("window");
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for (k, (dx, dy)) in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
            .into_iter()
            .enumerate()
        {
            let (pickup, delivery) = (1 + 2 * k, 2 + 2 * k);
            customers.push(Customer::new(pickup, dx, dy, 5, 0.0).pickup_of(delivery));
            customers.push(
                Customer::new(delivery, 2.0 * dx, 2.0 * dy, 0, 0.0)
                    .delivery_of(pickup)
                    .with_time_window(window),
            );
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn a_pair_of_an_emptied_route_moves_whole_and_in_order() {
        let (customers, dm) = three_pairs(1e9);
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            vec![0; 3],
            &[Vehicle::new(0, 10)],
            2,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes.len(), 2);
        assert!(plan.unserved.is_empty(), "{plan:?}");
        for r in &plan.routes {
            assert!(
                route_feasible(r, 0, &dm, &customers, &RouteLimits::NONE),
                "{r:?}"
            );
            assert_eq!(r.len() % 2, 0, "a pair is never split: {r:?}");
        }
        let served: usize = plan.routes.iter().map(Vec::len).sum();
        assert_eq!(served, 6);
    }

    #[test]
    fn a_pair_that_fits_nowhere_is_unserved_whole() {
        // Each delivery has to be reached by t = 2.5: only as the first pair of a route.
        let (customers, dm) = three_pairs(2.5);
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            vec![0; 3],
            &[Vehicle::new(0, 10)],
            2,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes.len(), 2);
        assert_eq!(plan.unserved.len(), 2, "{plan:?}");
        let (a, b) = (plan.unserved[0], plan.unserved[1]);
        assert_eq!(
            customers[a]
                .delivery_index()
                .or(customers[a].pickup_index()),
            Some(b),
            "the two unserved are one pair: {plan:?}"
        );
        for r in &plan.routes {
            assert!(
                route_feasible(r, 0, &dm, &customers, &RouteLimits::NONE),
                "{r:?}"
            );
        }
    }

    #[test]
    fn the_capacity_of_a_route_bounds_the_load_along_it() {
        // Capacity 5: a pair can only be added after another, never nested.
        let (customers, dm) = three_pairs(1e9);
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            vec![0; 3],
            &[Vehicle::new(0, 5)],
            2,
            &dm,
            &customers,
        );
        assert!(plan.unserved.is_empty(), "{plan:?}");
        for r in &plan.routes {
            assert!(route_peak_load(r, &customers) <= 5, "{r:?}");
        }
    }

    #[test]
    fn a_pair_is_not_nested_in_another_when_the_vehicle_cannot_carry_both() {
        // Nested, the second pair adds the least distance (0 -> 1 -> 1.5 -> 2.5 -> 3 -> 0), but
        // it would carry 10 on a vehicle of 5: the pairs have to follow one another.
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 5, 0.0).pickup_of(2),
            Customer::new(2, 3.0, 0.0, 0, 0.0).delivery_of(1),
            Customer::new(3, 1.5, 0.0, 5, 0.0).pickup_of(4),
            Customer::new(4, 2.5, 0.0, 0, 0.0).delivery_of(3),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4]],
            vec![0; 2],
            &[Vehicle::new(0, 5)],
            1,
            &dm,
            &customers,
        );
        assert!(plan.unserved.is_empty(), "{plan:?}");
        assert_eq!(plan.routes.len(), 1);
        assert_eq!(plan.routes[0].len(), 4);
        assert!(
            route_peak_load(&plan.routes[0], &customers) <= 5,
            "{plan:?}"
        );
    }

    #[test]
    fn reinsertion_never_makes_a_customer_late() {
        let (mut customers, _) = six_on_a_ring();
        // Customer 1 must be reached by t = 11 -- only first on its route.
        let c1 = &customers[1];
        customers[1] = Customer::new(1, c1.x(), c1.y(), 10, 0.0)
            .with_time_window(TimeWindow::new(0.0, 11.0).expect("window"));
        let dm = DistanceMatrix::from_customers(&customers);
        let plan = limit_routes(
            vec![vec![1], vec![2, 3], vec![4, 5, 6]],
            vec![0; 3],
            &[Vehicle::new(0, 40)],
            2,
            &dm,
            &customers,
        );
        for r in &plan.routes {
            assert!(
                route_feasible(r, 0, &dm, &customers, &RouteLimits::NONE),
                "{r:?}"
            );
        }
        let served: usize = plan.routes.iter().map(Vec::len).sum();
        assert_eq!(served + plan.unserved.len(), 6);
    }
}
