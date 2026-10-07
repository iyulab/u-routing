//! Planning within a fixed fleet.
//!
//! The constructive and metaheuristic methods open as many routes as the
//! demand needs. When the fleet is fixed at `K` vehicles, [`limit_routes`]
//! brings such a plan down to at most `K` routes: it empties the routes whose
//! removal loses the least demand, moves their customers into the routes that
//! remain wherever capacity and time windows allow, and reports the customers
//! no remaining route can take.
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
use crate::evaluation::{has_time_windows, route_feasible, RouteLimits};
use crate::models::Customer;

/// Whether inserting `customer_id` at `pos` of `route` keeps every customer
/// of the route on time. Always true when `windowed` is false.
pub(crate) fn insertion_on_time(
    route: &[usize],
    pos: usize,
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    windowed: bool,
) -> bool {
    if !windowed {
        return true;
    }
    let mut candidate = Vec::with_capacity(route.len() + 1);
    candidate.extend_from_slice(&route[..pos]);
    candidate.push(customer_id);
    candidate.extend_from_slice(&route[pos..]);
    route_feasible(&candidate, 0, distances, customers, &RouteLimits::NONE)
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
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> Option<(usize, usize, f64)> {
    let depot = 0;
    let windowed = has_time_windows(customers);
    let demand = customers[customer_id].demand();
    let mut best: Option<(usize, usize, f64)> = None;

    for (ri, route) in routes.iter().enumerate() {
        let load: i32 = route.iter().map(|&c| customers[c].demand()).sum();
        if load + demand > capacity_of(ri) {
            continue;
        }
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
                && insertion_on_time(route, pos, customer_id, distances, customers, windowed)
            {
                best = Some((ri, pos, cost));
            }
        }
    }
    best
}

/// A plan brought within a route limit by [`limit_routes`].
#[derive(Debug, Clone, PartialEq)]
pub struct LimitedPlan {
    /// At most `max_routes` routes of customer indices (depot excluded).
    pub routes: Vec<Vec<usize>>,
    /// The capacity of each route in `routes`, in the same order.
    pub capacities: Vec<i32>,
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
/// `capacities[i]` is the capacity of `routes[i]`, so a fleet of mixed
/// vehicles keeps each route's own limit.
///
/// # Panics
/// If `capacities` and `routes` differ in length (a caller bug).
///
/// # Examples
///
/// ```
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
/// let plan = limit_routes(vec![vec![1], vec![2], vec![3]], vec![20, 20, 20], 2, &dm, &customers);
/// assert_eq!(plan.routes.len(), 2);
/// assert!(plan.unserved.is_empty()); // one route took a second customer
/// ```
pub fn limit_routes(
    routes: Vec<Vec<usize>>,
    capacities: Vec<i32>,
    max_routes: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> LimitedPlan {
    assert_eq!(
        routes.len(),
        capacities.len(),
        "one capacity per route is required"
    );
    let mut routes = routes;
    let mut capacities = capacities;
    let mut unserved = Vec::new();

    while routes.len() > max_routes {
        let load = |r: &Vec<usize>| -> i32 { r.iter().map(|&c| customers[c].demand()).sum() };
        let victim = (0..routes.len())
            .min_by_key(|&i| (load(&routes[i]), routes[i].len()))
            .expect("more routes than the limit, so at least one");
        let mut pending = routes.remove(victim);
        capacities.remove(victim);

        // Cheapest insertion first, until nothing pending fits anywhere.
        loop {
            let best = pending
                .iter()
                .enumerate()
                .filter_map(|(pi, &cid)| {
                    cheapest_insertion(&routes, |ri| capacities[ri], cid, distances, customers)
                        .map(|(ri, pos, cost)| (pi, ri, pos, cost))
                })
                .min_by(|a, b| a.3.total_cmp(&b.3));
            match best {
                Some((pi, ri, pos, _)) => {
                    let cid = pending.remove(pi);
                    routes[ri].insert(pos, cid);
                }
                None => break,
            }
        }
        unserved.extend(pending);
    }

    unserved.sort_unstable();
    LimitedPlan {
        routes,
        capacities,
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
        let plan = limit_routes(routes.clone(), vec![20; 3], 3, &dm, &customers);
        assert_eq!(plan.routes, routes);
        assert!(plan.unserved.is_empty());
    }

    #[test]
    fn a_full_fleet_leaves_one_routes_customers_unserved() {
        let (customers, dm) = six_on_a_ring();
        let plan = limit_routes(
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            vec![20; 3],
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
            vec![30; 3],
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
            vec![21, 21, 21, 21],
            3,
            &dm,
            &customers,
        );
        assert_eq!(plan.routes.len(), 3);
        assert!(plan.unserved.is_empty(), "{plan:?}");
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
            vec![40; 3],
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
