//! Construction for a problem with several depots.
//!
//! Cluster first, route second: every customer goes to the nearest depot that
//! has a vehicle, each depot's customers are routed from that depot by a
//! single-depot construction, and a customer the construction cannot serve
//! (no vehicle of the depot takes its load, or reaches it in time) is offered
//! to the next-nearest depot. The single-depot heuristics run unchanged, on
//! the depot's own sub-problem, so a problem with one depot is answered exactly
//! as the heuristic alone answers it.

use super::nearest_neighbor::fleet_vehicle;
use crate::distance::DistanceMatrix;
use crate::evaluation::RouteEvaluator;
use crate::models::{num_depots, Customer, Solution, Vehicle};

/// What one depot's planner made, in the numbering of the depot's own
/// sub-problem: its routes (position in the depot's fleet, customer ids of its
/// table) and the customers it could not serve (ids of its table).
#[derive(Default)]
pub(crate) struct Share {
    pub(crate) routes: Vec<(usize, Vec<usize>)>,
    pub(crate) unassigned: Vec<usize>,
}

/// What a depot made, in the numbering of the whole problem: its routes
/// (index into the fleet, customer ids) and the customers it left out.
#[derive(Default)]
pub(crate) struct Built {
    pub(crate) routes: Vec<(usize, Vec<usize>)>,
    pub(crate) unassigned: Vec<usize>,
}

/// Gives every customer to a depot and has `plan` make each depot's routes,
/// offering what a depot could not serve to the customer's next-nearest depot
/// (see [`construct_by_depot`], which is this with a construction heuristic as
/// `plan`).
///
/// `fleet_depots[i]` is the depot the vehicle `i` starts from (an entry that
/// is no depot is not used). `plan(depot, table, distances, fleet)` is given
/// the depot's sub-problem -- the table of the depot (entry 0) and its
/// customers, their distances, and `fleet`, the indices of the vehicles of
/// the depot -- and answers in the sub-problem's numbering, naming a vehicle
/// by its position in `fleet`.
///
/// Returns, for each depot, what it made, in the numbering of the whole
/// problem, and the customers no depot has a vehicle for.
pub(crate) fn by_depot<E>(
    customers: &[Customer],
    distances: &DistanceMatrix,
    fleet_depots: &[usize],
    mut plan: impl FnMut(usize, &[Customer], &DistanceMatrix, &[usize]) -> Result<Share, E>,
) -> Result<(Vec<Built>, Vec<usize>), E> {
    let depots = num_depots(customers);
    let n = customers.len();

    // The vehicles of each depot, as indices into the fleet.
    let mut fleet: Vec<Vec<usize>> = vec![Vec::new(); depots];
    for (i, &d) in fleet_depots.iter().enumerate() {
        if d < depots {
            fleet[d].push(i);
        }
    }

    // The depots each customer may go to, nearest first.
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, list) in candidates.iter_mut().enumerate().skip(depots) {
        *list = (0..depots).filter(|&d| !fleet[d].is_empty()).collect();
        list.sort_by(|&a, &b| {
            distances
                .get(a, c)
                .total_cmp(&distances.get(b, c))
                .then(a.cmp(&b))
        });
    }
    // A pair goes to one depot: its delivery is offered to the depots its pickup is.
    for pickup in depots..n {
        if let Some(delivery) = customers[pickup].delivery_index().filter(|&d| d < n) {
            candidates[delivery] = candidates[pickup].clone();
        }
    }
    let mut attempt = vec![0usize; n];
    let mut built: Vec<Built> = (0..depots).map(|_| Built::default()).collect();
    let mut dirty = vec![true; depots];

    loop {
        for d in 0..depots {
            if !std::mem::take(&mut dirty[d]) {
                continue;
            }
            let members: Vec<usize> = (depots..n)
                .filter(|&c| candidates[c].get(attempt[c]) == Some(&d))
                .collect();
            built[d] = if members.is_empty() {
                Built::default()
            } else {
                let mut rows = Vec::with_capacity(members.len() + 1);
                rows.push(d);
                rows.extend_from_slice(&members);
                // The table renumbers its rows: a pair's partner is named by its new row.
                let mut slot = vec![None; n];
                for (k, &r) in rows.iter().enumerate() {
                    slot[r] = Some(k);
                }
                let table: Vec<Customer> = rows
                    .iter()
                    .enumerate()
                    .map(|(k, &r)| {
                        let c = &customers[r];
                        c.placed(k, k == 0).with_partner(
                            c.delivery_index()
                                .and_then(|g| slot.get(g).copied().flatten()),
                            c.pickup_index()
                                .and_then(|g| slot.get(g).copied().flatten()),
                        )
                    })
                    .collect();
                let share = plan(d, &table, &distances.restricted(&rows), &fleet[d])?;
                Built {
                    routes: share
                        .routes
                        .into_iter()
                        .map(|(k, ids)| (fleet[d][k], ids.into_iter().map(|id| rows[id]).collect()))
                        .collect(),
                    unassigned: share.unassigned.into_iter().map(|id| rows[id]).collect(),
                }
            };
        }
        // Offer what a depot could not serve to the customer's next depot.
        let mut moved = false;
        for d in 0..depots {
            for &c in &built[d].unassigned {
                // A pair moves with its pickup.
                if customers[c].pickup_index().is_some() {
                    continue;
                }
                if attempt[c] + 1 < candidates[c].len() {
                    for moving in std::iter::once(c).chain(customers[c].delivery_index()) {
                        attempt[moving] += 1;
                    }
                    dirty[d] = true;
                    dirty[candidates[c][attempt[c]]] = true;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
    // Customers no depot has a vehicle for were never anyone's.
    let nobody: Vec<usize> = (depots..n).filter(|&c| candidates[c].is_empty()).collect();
    Ok((built, nobody))
}

/// Builds a plan for a problem with several depots by running `construct` --
/// any single-depot construction, e.g. `|c, d, v| nearest_neighbor(c, d, v)` --
/// on each depot's share of the customers.
///
/// `customers` lists the depots first ([`num_depots`]); each of `vehicles`
/// starts from its own depot ([`Vehicle::with_depot`]) and is used by that
/// depot's routes only. `construct` is given a table of the depot (entry 0)
/// and the customers given to it, and the vehicles of that depot, and returns
/// a plan for it in that table's numbering; a vehicle that names an entry
/// that is no depot is not used.
///
/// A customer goes to the nearest depot that has a vehicle (a tie to the lower
/// index). Whatever a depot's construction leaves unassigned is offered to
/// the customer's next-nearest depot, and so on; a customer no depot serves
/// is reported unassigned. With one depot `construct` is called on the
/// problem as given.
///
/// # Complexity
/// At most `D` rounds of `construct` per depot, each on the depot's
/// sub-problem, plus O(D n² ) to build the sub-matrices, for D depots and n
/// customers.
///
/// # Examples
///
/// ```
/// use u_routing::constructive::{construct_by_depot, nearest_neighbor};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::models::{Customer, Vehicle};
///
/// // Two depots at either end of a line, a customer next to each.
/// let customers = vec![
///     Customer::depot_at(0, 0.0, 0.0),
///     Customer::depot_at(1, 100.0, 0.0),
///     Customer::new(2, 1.0, 0.0, 5, 0.0),
///     Customer::new(3, 99.0, 0.0, 5, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let vehicles = vec![Vehicle::new(0, 10), Vehicle::new(1, 10).with_depot(1)];
///
/// let plan = construct_by_depot(&customers, &dm, &vehicles, |c, d, v| nearest_neighbor(c, d, v));
/// assert_eq!(plan.num_served(), 2);
/// assert_eq!(plan.total_distance(), 4.0); // 2 out and back from each depot
/// ```
pub fn construct_by_depot<F>(
    customers: &[Customer],
    distances: &DistanceMatrix,
    vehicles: &[Vehicle],
    construct: F,
) -> Solution
where
    F: Fn(&[Customer], &DistanceMatrix, &[Vehicle]) -> Solution,
{
    if num_depots(customers) == 1 {
        return construct(customers, distances, vehicles);
    }
    let fleet_depots: Vec<usize> = vehicles.iter().map(Vehicle::depot_id).collect();
    let planned = by_depot(
        customers,
        distances,
        &fleet_depots,
        |_, table, sub_distances, fleet| -> Result<Share, std::convert::Infallible> {
            // The depot's vehicles as its own table sees them: its depot is entry 0.
            let sub_vehicles: Vec<Vehicle> = fleet
                .iter()
                .enumerate()
                .map(|(k, &i)| vehicles[i].placed(k, 0))
                .collect();
            let plan = construct(table, sub_distances, &sub_vehicles);
            Ok(Share {
                routes: plan
                    .routes()
                    .iter()
                    .map(|r| (r.vehicle_id(), r.customer_ids()))
                    .collect(),
                unassigned: plan.unassigned().to_vec(),
            })
        },
    );
    let (built, nobody) = match planned {
        Ok(done) => done,
        Err(never) => match never {},
    };

    let mut plan = Solution::new();
    for b in &built {
        for (vehicle, ids) in &b.routes {
            let (route, _) =
                RouteEvaluator::new(customers, distances, &vehicles[*vehicle]).build_route(ids);
            plan.add_route(route);
        }
    }
    let mut left: Vec<usize> = built
        .iter()
        .flat_map(|b| b.unassigned.iter().copied())
        .collect();
    left.extend(nobody);
    left.sort_unstable();
    for c in left {
        plan.add_unassigned(c);
    }
    plan.price(|r| fleet_vehicle(vehicles, r));
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constructive::{nearest_neighbor, sweep};

    fn nn(c: &[Customer], d: &DistanceMatrix, v: &[Vehicle]) -> Solution {
        nearest_neighbor(c, d, v)
    }

    /// Depots at x = 0 and x = 100 on a line, customers 2.. at the given x.
    fn line(xs: &[f64], demand: i32) -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![
            Customer::depot_at(0, 0.0, 0.0),
            Customer::depot_at(1, 100.0, 0.0),
        ];
        for (k, &x) in xs.iter().enumerate() {
            customers.push(Customer::new(2 + k, x, 0.0, demand, 0.0));
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    fn routes_of(plan: &Solution) -> Vec<(usize, Vec<usize>)> {
        plan.routes()
            .iter()
            .map(|r| (r.vehicle_id(), r.customer_ids()))
            .collect()
    }

    #[test]
    fn one_depot_is_the_heuristic_alone() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 1.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20)];
        let direct = nearest_neighbor(&customers, &dm, &vehicles);
        let wrapped = construct_by_depot(&customers, &dm, &vehicles, nn);
        assert_eq!(routes_of(&direct), routes_of(&wrapped));
        assert_eq!(direct.unassigned(), wrapped.unassigned());
        assert_eq!(direct.total_distance(), wrapped.total_distance());
    }

    #[test]
    fn each_customer_goes_to_its_nearest_depot() {
        let (customers, dm) = line(&[10.0, 20.0, 90.0, 80.0], 5);
        let vehicles = vec![
            Vehicle::new(0, 20).with_fixed_cost(7.0),
            Vehicle::new(1, 20)
                .with_depot(1)
                .with_cost_per_distance(2.0),
        ];
        let plan = construct_by_depot(&customers, &dm, &vehicles, nn);
        assert_eq!(
            routes_of(&plan),
            [(0, vec![2, 3]), (1, vec![4, 5])],
            "{plan:?}"
        );
        // Priced with each route's own vehicle: 7 + 40, and 2 x 40.
        assert!((plan.total_cost() - 127.0).abs() < 1e-9, "{plan:?}");
        // 0 -> 10 -> 20 -> 0 and 100 -> 90 -> 80 -> 100: 40 each.
        assert!((plan.total_distance() - 80.0).abs() < 1e-9);
        assert_eq!(plan.num_unassigned(), 0);
    }

    #[test]
    fn a_customer_the_nearest_depot_cannot_carry_goes_to_the_next() {
        // Customer 3 is next to depot 0 but asks for 50; only depot 1's vehicle takes that.
        let customers = vec![
            Customer::depot_at(0, 0.0, 0.0),
            Customer::depot_at(1, 100.0, 0.0),
            Customer::new(2, 10.0, 0.0, 5, 0.0),
            Customer::new(3, 5.0, 0.0, 50, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 30), Vehicle::new(1, 60).with_depot(1)];
        let plan = construct_by_depot(&customers, &dm, &vehicles, nn);
        assert_eq!(routes_of(&plan), [(0, vec![2]), (1, vec![3])], "{plan:?}");
        assert_eq!(plan.num_unassigned(), 0);
    }

    #[test]
    fn a_customer_no_depot_can_carry_is_unassigned_and_depots_without_vehicles_get_none() {
        let (customers, dm) = line(&[90.0, 10.0, 20.0], 5);
        // Only depot 0 has a vehicle: customer 2 is near depot 1 but goes to 0.
        let vehicles = vec![Vehicle::new(0, 10)];
        let plan = construct_by_depot(&customers, &dm, &vehicles, nn);
        assert_eq!(plan.num_served(), 2, "{plan:?}"); // capacity 10 holds two 5s
        assert_eq!(plan.num_unassigned(), 1);
        assert!(plan.routes().iter().all(|r| r.vehicle_id() == 0));
        // No vehicle at all: everything is unassigned, in order.
        let none = construct_by_depot(&customers, &dm, &[], nn);
        assert_eq!(none.unassigned(), [2, 3, 4]);
    }

    #[test]
    fn a_route_starts_at_its_own_depot_whichever_it_is() {
        // Both customers are nearest to depot 1: from it, 95 is the nearer and 70 the farther.
        // (Started at the wrong row the route would begin at the first customer, 70.)
        let (customers, dm) = line(&[70.0, 95.0], 5);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20).with_depot(1)];
        let plan = construct_by_depot(&customers, &dm, &vehicles, nn);
        assert_eq!(routes_of(&plan), [(1, vec![3, 2])], "{plan:?}");
        assert!((plan.total_distance() - 60.0).abs() < 1e-9); // 5 + 25 + 30
    }

    #[test]
    fn a_vehicle_at_a_row_that_is_no_depot_is_not_used() {
        let (customers, dm) = line(&[10.0, 90.0], 5);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20).with_depot(3)];
        let plan = construct_by_depot(&customers, &dm, &vehicles, nn);
        assert!(plan.routes().iter().all(|r| r.vehicle_id() == 0));
        assert_eq!(plan.num_served() + plan.num_unassigned(), 2);
    }

    #[test]
    fn a_single_vehicle_heuristic_runs_per_depot_too() {
        let (customers, dm) = line(&[10.0, 20.0, 90.0, 80.0], 5);
        let vehicles = vec![Vehicle::new(0, 20), Vehicle::new(1, 20).with_depot(1)];
        let plan = construct_by_depot(&customers, &dm, &vehicles, |c, d, v| sweep(c, d, &v[0]));
        assert_eq!(plan.num_served(), 4);
        // Each route starts at its vehicle's own depot: no leg longer than 20.
        assert!((plan.total_distance() - 80.0).abs() < 1e-9, "{plan:?}");
    }
}
