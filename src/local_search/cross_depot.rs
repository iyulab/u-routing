//! Moving customers between the routes of different depots.
//!
//! # Algorithm
//!
//! Best improvement over two moves between every pair of routes whose vehicles
//! start from different depots: relocate (a customer moves to its best position
//! in the other route) and swap (two customers change routes, each into the
//! other's place). A move counts when both routes it makes are allowed by
//! their own vehicle -- capacity, time windows, limits, break -- and the two
//! routes cost less than before, a route priced with its own vehicle (fixed
//! cost + cost per distance x distance, lateness priced as distance), so that
//! emptying a route saves its fixed cost.
//!
//! The moves inside one depot's routes are the other operators' work
//! ([`relocate_improve`](super::relocate_improve),
//! [`exchange_improve`](super::exchange_improve), ...); this one is what lets a
//! plan built depot by depot (see
//! [`construct_by_depot`](crate::constructive::construct_by_depot)) change which
//! depot serves a customer.
//!
//! # Complexity
//!
//! O(R² L³) per scan and O(R² L³) per move taken for R routes of up to L
//! customers (a route is evaluated in O(L)), and each move lowers the cost.

use super::gain_floor;
use crate::constructive::fleet_vehicle;
use crate::distance::DistanceMatrix;
use crate::evaluation::{route_eval, RouteEvaluator, RouteLimits};
use crate::models::{Customer, Solution, Vehicle};

/// A route in the search: its customers, its vehicle, what it carries and costs.
struct State<'v> {
    ids: Vec<usize>,
    vehicle: &'v Vehicle,
    load: i32,
    cost: f64,
}

/// A move between the routes `a` and `b`, and the two routes it makes.
enum Move {
    /// The customer at `pos` of `a` goes to position `at` of `b`.
    Relocate {
        a: usize,
        pos: usize,
        b: usize,
        at: usize,
    },
    /// The customer at `pa` of `a` and the one at `pb` of `b` change places.
    Swap {
        a: usize,
        pa: usize,
        b: usize,
        pb: usize,
    },
}

/// What a route of `ids` costs with `vehicle`, or `None` when the vehicle
/// does not allow it (an empty route costs nothing).
fn priced(
    ids: &[usize],
    vehicle: &Vehicle,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> Option<f64> {
    if ids.is_empty() {
        return Some(0.0);
    }
    route_eval(
        ids,
        vehicle.depot_id(),
        distances,
        customers,
        &RouteLimits::of(vehicle),
    )
    .map(|e| vehicle.route_cost(e.priced_distance()))
}

/// Moves customers between routes that start from different depots while that
/// lowers the plan's cost; routes of one depot are never touched.
///
/// `vehicles` are the fleet: a route's vehicle is the one of its
/// [`Route::vehicle_id`](crate::models::Route::vehicle_id) (by
/// [`Vehicle::id`]), and the depot it starts from is that vehicle's. The
/// plan's unassigned customers are kept as they are, and a route a move empties
/// is dropped.
///
/// # Examples
///
/// ```
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::evaluation::RouteEvaluator;
/// use u_routing::local_search::cross_depot_improve;
/// use u_routing::models::{Customer, Solution, Vehicle};
///
/// // Two depots on a line; customer 3, at 90, is nearer to the depot at 100.
/// let customers = vec![
///     Customer::depot_at(0, 0.0, 0.0),
///     Customer::depot_at(1, 100.0, 0.0),
///     Customer::new(2, 5.0, 0.0, 1, 0.0),
///     Customer::new(3, 90.0, 0.0, 1, 0.0),
///     Customer::new(4, 95.0, 0.0, 1, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 10), Vehicle::new(1, 10).with_depot(1)];
///
/// // A plan in which depot 0's vehicle drives out to customer 3 as well.
/// let mut plan = Solution::new();
/// for (vehicle, ids) in [(0, vec![2, 3]), (1, vec![4])] {
///     let route = RouteEvaluator::new(&customers, &dm, &vehicles[vehicle]).build_route(&ids).0;
///     plan.add_route(route);
/// }
/// assert_eq!(plan.total_distance(), 190.0);
///
/// // The move gives customer 3 to the route from depot 1.
/// let improved = cross_depot_improve(&plan, &customers, &dm, &vehicles);
/// assert_eq!(improved.total_distance(), 30.0);
/// ```
pub fn cross_depot_improve(
    solution: &Solution,
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicles: &[Vehicle],
) -> Solution {
    if solution.num_routes() < 2 || vehicles.is_empty() {
        return solution.clone();
    }
    let floor = gain_floor(distances);
    // A route is priced as it stands, whether or not its vehicle allows it.
    let mut routes: Vec<State> = solution
        .routes()
        .iter()
        .map(|r| {
            let vehicle = fleet_vehicle(vehicles, r);
            let ids = r.customer_ids();
            State {
                load: ids.iter().map(|&c| customers[c].demand()).sum(),
                cost: vehicle.route_cost(r.total_distance() + r.lateness_cost()),
                ids,
                vehicle,
            }
        })
        .collect();

    let mut buf_a: Vec<usize> = Vec::new();
    let mut buf_b: Vec<usize> = Vec::new();
    while let Some((mv, cost_a, cost_b)) =
        best_move(&routes, customers, distances, floor, &mut buf_a, &mut buf_b)
    {
        let (a, b) = match mv {
            Move::Relocate { a, pos, b, at } => {
                let c = routes[a].ids.remove(pos);
                routes[b].ids.insert(at, c);
                (a, b)
            }
            Move::Swap { a, pa, b, pb } => {
                let (c1, c2) = (routes[a].ids[pa], routes[b].ids[pb]);
                routes[a].ids[pa] = c2;
                routes[b].ids[pb] = c1;
                (a, b)
            }
        };
        for (r, cost) in [(a, cost_a), (b, cost_b)] {
            routes[r].cost = cost;
            routes[r].load = routes[r].ids.iter().map(|&c| customers[c].demand()).sum();
        }
        routes.retain(|r| !r.ids.is_empty());
    }

    let mut out = Solution::new();
    for r in &routes {
        let (route, _) = RouteEvaluator::new(customers, distances, r.vehicle).build_route(&r.ids);
        out.add_route(route);
    }
    for &u in solution.unassigned() {
        out.add_unassigned(u);
    }
    out.price(|r| fleet_vehicle(vehicles, r));
    out
}

/// The move with the greatest gain (the first found among equals) and the
/// costs of the two routes it makes, if any gains more than `floor`.
fn best_move(
    routes: &[State],
    customers: &[Customer],
    distances: &DistanceMatrix,
    floor: f64,
    buf_a: &mut Vec<usize>,
    buf_b: &mut Vec<usize>,
) -> Option<(Move, f64, f64)> {
    let mut best: Option<(f64, Move, f64, f64)> = None;
    let mut offer = |delta: f64, mv: Move, cost_a: f64, cost_b: f64| {
        if delta < -floor && best.as_ref().is_none_or(|(d, ..)| delta < *d) {
            best = Some((delta, mv, cost_a, cost_b));
        }
    };
    for a in 0..routes.len() {
        for b in 0..routes.len() {
            if a == b || routes[a].vehicle.depot_id() == routes[b].vehicle.depot_id() {
                continue;
            }
            let (ra, rb) = (&routes[a], &routes[b]);
            let before = ra.cost + rb.cost;
            // Relocate: out of `a`, into `b`.
            for pos in 0..ra.ids.len() {
                let c = ra.ids[pos];
                let demand = customers[c].demand();
                if rb.load + demand > rb.vehicle.capacity() {
                    continue;
                }
                buf_a.clear();
                buf_a.extend(
                    ra.ids
                        .iter()
                        .enumerate()
                        .filter(|&(p, _)| p != pos)
                        .map(|(_, &x)| x),
                );
                let Some(cost_a) = priced(buf_a, ra.vehicle, distances, customers) else {
                    continue;
                };
                for at in 0..=rb.ids.len() {
                    buf_b.clear();
                    buf_b.extend_from_slice(&rb.ids[..at]);
                    buf_b.push(c);
                    buf_b.extend_from_slice(&rb.ids[at..]);
                    if let Some(cost_b) = priced(buf_b, rb.vehicle, distances, customers) {
                        offer(
                            cost_a + cost_b - before,
                            Move::Relocate { a, pos, b, at },
                            cost_a,
                            cost_b,
                        );
                    }
                }
            }
            // Swap, each pair of routes once.
            if a < b {
                for pa in 0..ra.ids.len() {
                    for pb in 0..rb.ids.len() {
                        let (c1, c2) = (ra.ids[pa], rb.ids[pb]);
                        let (d1, d2) = (customers[c1].demand(), customers[c2].demand());
                        if ra.load - d1 + d2 > ra.vehicle.capacity()
                            || rb.load - d2 + d1 > rb.vehicle.capacity()
                        {
                            continue;
                        }
                        buf_a.clear();
                        buf_a.extend_from_slice(&ra.ids);
                        buf_a[pa] = c2;
                        buf_b.clear();
                        buf_b.extend_from_slice(&rb.ids);
                        buf_b[pb] = c1;
                        let (Some(cost_a), Some(cost_b)) = (
                            priced(buf_a, ra.vehicle, distances, customers),
                            priced(buf_b, rb.vehicle, distances, customers),
                        ) else {
                            continue;
                        };
                        offer(
                            cost_a + cost_b - before,
                            Move::Swap { a, pa, b, pb },
                            cost_a,
                            cost_b,
                        );
                    }
                }
            }
        }
    }
    best.map(|(_, mv, cost_a, cost_b)| (mv, cost_a, cost_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Depot 0 at x = 0 and depot 1 at x = 100 on a line; customers 2.. at the
    /// given x, each with the demand given.
    fn line(spec: &[(f64, i32)]) -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![
            Customer::depot_at(0, 0.0, 0.0),
            Customer::depot_at(1, 100.0, 0.0),
        ];
        for (k, &(x, demand)) in spec.iter().enumerate() {
            customers.push(Customer::new(2 + k, x, 0.0, demand, 0.0));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    /// The plan whose route `i` is the customers `routes[i].1` driven by vehicle `routes[i].0`.
    fn plan(
        customers: &[Customer],
        dm: &DistanceMatrix,
        vehicles: &[Vehicle],
        routes: &[(usize, &[usize])],
    ) -> Solution {
        let mut s = Solution::new();
        for &(v, ids) in routes {
            s.add_route(
                RouteEvaluator::new(customers, dm, &vehicles[v])
                    .build_route(ids)
                    .0,
            );
        }
        s.price(|r| fleet_vehicle(vehicles, r));
        s
    }

    fn routes_of(s: &Solution) -> Vec<(usize, Vec<usize>)> {
        s.routes()
            .iter()
            .map(|r| (r.vehicle_id(), r.customer_ids()))
            .collect()
    }

    fn two_vehicles(capacity: i32) -> Vec<Vehicle> {
        vec![
            Vehicle::new(0, capacity),
            Vehicle::new(1, capacity).with_depot(1),
        ]
    }

    #[test]
    fn a_customer_nearer_to_the_other_depot_changes_routes() {
        // Customer 3 (x = 90) rides with customer 4 (x = 5) from depot 0: 5 + 85 + 90 = 180.
        let (c, dm) = line(&[(95.0, 1), (90.0, 1), (5.0, 1)]);
        let vehicles = two_vehicles(10);
        let before = plan(&c, &dm, &vehicles, &[(0, &[4, 3]), (1, &[2])]);
        assert!((before.total_distance() - 190.0).abs() < 1e-9);
        let after = cross_depot_improve(&before, &c, &dm, &vehicles);
        // 0 -> 5 -> 0 is 10; 100 -> 90 -> 95 -> 100 is 20 (the first position among equals).
        assert_eq!(
            routes_of(&after),
            [(0, vec![4]), (1, vec![3, 2])],
            "{after:?}"
        );
        assert!((after.total_distance() - 30.0).abs() < 1e-9);
        assert!((after.total_cost() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn two_customers_in_each_others_depots_swap_when_neither_can_move_alone() {
        // Capacity 10 and demand 10: each route is full, so only a swap helps.
        let (c, dm) = line(&[(90.0, 10), (10.0, 10)]);
        let vehicles = two_vehicles(10);
        let before = plan(&c, &dm, &vehicles, &[(0, &[2]), (1, &[3])]);
        assert!((before.total_distance() - 360.0).abs() < 1e-9);
        let after = cross_depot_improve(&before, &c, &dm, &vehicles);
        assert_eq!(routes_of(&after), [(0, vec![3]), (1, vec![2])], "{after:?}");
        assert!((after.total_distance() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn a_move_a_vehicle_does_not_allow_is_not_made() {
        // Capacities 5 and 10, loads 4 and 9: the swap would put 9 on the vehicle of 5, and
        // neither customer fits the other route as an addition.
        let (c, dm) = line(&[(90.0, 4), (10.0, 9)]);
        let vehicles = [Vehicle::new(0, 5), Vehicle::new(1, 10).with_depot(1)];
        let before = plan(&c, &dm, &vehicles, &[(0, &[2]), (1, &[3])]);
        let after = cross_depot_improve(&before, &c, &dm, &vehicles);
        assert_eq!(routes_of(&after), routes_of(&before));
        // A limit does the same: depot 1's vehicle may drive 15 at most, and the
        // route 100 -> 95 -> 90 -> 100 it would make is 20.
        let limited = [
            Vehicle::new(0, 50),
            Vehicle::new(1, 50).with_depot(1).with_max_distance(15.0),
        ];
        let (c, dm) = line(&[(90.0, 1), (95.0, 1)]);
        let before = plan(&c, &dm, &limited, &[(0, &[2]), (1, &[3])]);
        let after = cross_depot_improve(&before, &c, &dm, &limited);
        assert_eq!(routes_of(&after), routes_of(&before), "{after:?}");
        // Without the limit the move is made: 180 + 10 becomes 20.
        let free = [Vehicle::new(0, 50), Vehicle::new(1, 50).with_depot(1)];
        let moved = cross_depot_improve(&before, &c, &dm, &free);
        assert_eq!(routes_of(&moved), [(1, vec![2, 3])], "{moved:?}");
    }

    #[test]
    fn routes_of_one_depot_are_left_alone() {
        // Two routes from depot 0 that one relocation would improve, and one from depot 1 that is fine.
        let (c, dm) = line(&[(10.0, 1), (20.0, 1), (15.0, 1), (95.0, 1)]);
        let vehicles = [
            Vehicle::new(0, 10),
            Vehicle::new(1, 10),
            Vehicle::new(2, 10).with_depot(1),
        ];
        let before = plan(&c, &dm, &vehicles, &[(0, &[2, 3]), (1, &[4]), (2, &[5])]);
        let after = cross_depot_improve(&before, &c, &dm, &vehicles);
        assert_eq!(routes_of(&after), routes_of(&before));
    }

    #[test]
    fn a_route_a_move_empties_gives_back_its_fixed_cost() {
        // The only customer of the route from depot 0 is at x = 70: nearer to depot 1.
        let (c, dm) = line(&[(70.0, 1), (95.0, 1)]);
        let vehicles = [
            Vehicle::new(0, 10).with_fixed_cost(50.0),
            Vehicle::new(1, 10).with_depot(1),
        ];
        let before = plan(&c, &dm, &vehicles, &[(0, &[2]), (1, &[3])]);
        // 50 + 140 for the first route, 10 for the second.
        assert!((before.total_cost() - 200.0).abs() < 1e-9);
        let after = cross_depot_improve(&before, &c, &dm, &vehicles);
        // One route from depot 1: 100 -> 70 -> 95 -> 100 = 60 (the first position among equals).
        assert_eq!(routes_of(&after), [(1, vec![2, 3])], "{after:?}");
        assert!((after.total_cost() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn a_saved_fixed_cost_decides_a_move_the_distance_alone_would_not_make() {
        // Customer 2 at x = 40 goes out and back from depot 0 for 80, with customer 3 (x = 95)
        // alone on a route of 10 from depot 1. Together from depot 1 they are 120 > 90, so the
        // distance says stay; the 50 that the emptied route's vehicle costs says move.
        let (c, dm) = line(&[(40.0, 1), (95.0, 1)]);
        let with_fixed = [
            Vehicle::new(0, 10).with_fixed_cost(50.0),
            Vehicle::new(1, 10).with_depot(1),
        ];
        let before = plan(&c, &dm, &with_fixed, &[(0, &[2]), (1, &[3])]);
        assert!((before.total_cost() - 140.0).abs() < 1e-9);
        let after = cross_depot_improve(&before, &c, &dm, &with_fixed);
        assert_eq!(routes_of(&after), [(1, vec![2, 3])], "{after:?}");
        assert!((after.total_cost() - 120.0).abs() < 1e-9);
        let free = two_vehicles(10);
        let before = plan(&c, &dm, &free, &[(0, &[2]), (1, &[3])]);
        assert_eq!(
            routes_of(&cross_depot_improve(&before, &c, &dm, &free)),
            routes_of(&before)
        );
    }

    #[test]
    fn one_depot_and_a_single_route_are_returned_as_they_are() {
        let (c, dm) = line(&[(10.0, 1), (20.0, 1)]);
        let vehicles = [Vehicle::new(0, 10), Vehicle::new(1, 10)];
        let one_depot = plan(&c, &dm, &vehicles, &[(0, &[2]), (1, &[3])]);
        let after = cross_depot_improve(&one_depot, &c, &dm, &vehicles);
        assert_eq!(routes_of(&after), routes_of(&one_depot));
        let single = plan(&c, &dm, &two_vehicles(10), &[(0, &[2, 3])]);
        assert_eq!(
            routes_of(&cross_depot_improve(&single, &c, &dm, &two_vehicles(10))),
            routes_of(&single)
        );
    }
}
