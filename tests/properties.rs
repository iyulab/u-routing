//! Property tests: the invariants every routing primitive must keep on any
//! input, not just on the hand-picked instances the unit tests use.
//!
//! - A giant-tour split serves each customer exactly once, in tour order,
//!   never over capacity, and reports the distance its routes actually cover.
//! - Every local-search move returns a permutation of what it was given, never
//!   lengthens it, and reports the distance of the route it returns.
//! - Every constructive heuristic serves each customer at most once, keeps
//!   every route within capacity, and prices each route as the matrix does.
//! - Inter-route moves keep the served set, capacity and the non-worsening
//!   guarantee.

use proptest::prelude::*;
use u_routing::constructive::{clarke_wright_savings, nearest_neighbor, sweep};
use u_routing::distance::DistanceMatrix;
use u_routing::ga::split;
use u_routing::local_search::{
    exchange_improve, or_opt_improve, relocate_improve, route_distance, three_opt_improve,
    two_opt_improve,
};
use u_routing::models::{Customer, Solution, Vehicle};

const EPS: f64 = 1e-9;

/// Customers 1..=n on a bounded plane with demands in `1..=max_demand`; index 0
/// is the depot at the origin.
fn instance(
    n: impl Into<proptest::collection::SizeRange>,
    max_demand: i32,
) -> impl Strategy<Value = Vec<Customer>> {
    proptest::collection::vec((-100.0_f64..100.0, -100.0_f64..100.0, 1..=max_demand), n).prop_map(
        |points| {
            let mut customers = vec![Customer::depot(0.0, 0.0)];
            for (i, (x, y, demand)) in points.into_iter().enumerate() {
                customers.push(Customer::new(i + 1, x, y, demand, 0.0));
            }
            customers
        },
    )
}

/// A capacity that admits every single customer, so a feasible split exists.
fn capacity_for(customers: &[Customer], slack: i32) -> i32 {
    customers.iter().map(Customer::demand).max().unwrap_or(0) + slack
}

fn sorted(mut ids: Vec<usize>) -> Vec<usize> {
    ids.sort_unstable();
    ids
}

fn all_customer_ids(customers: &[Customer]) -> Vec<usize> {
    (1..customers.len()).collect()
}

/// Every route within capacity, every route priced as the matrix prices it,
/// and the served set with no duplicates. Returns the served ids, sorted.
fn check_solution(
    solution: &Solution,
    customers: &[Customer],
    dm: &DistanceMatrix,
    capacity: i32,
) -> Result<Vec<usize>, TestCaseError> {
    let mut served = Vec::new();
    for route in solution.routes() {
        let ids = route.customer_ids();
        let load: i32 = ids.iter().map(|&c| customers[c].demand()).sum();
        prop_assert!(
            load <= capacity,
            "route load {load} exceeds capacity {capacity}"
        );
        prop_assert_eq!(
            route.total_load(),
            load,
            "route reports a load it does not carry"
        );
        let priced = route_distance(&ids, 0, dm);
        prop_assert!(
            (route.total_distance() - priced).abs() < EPS,
            "route reports {} but its visits cover {priced}",
            route.total_distance()
        );
        served.extend(ids);
    }
    let served = sorted(served);
    let mut dedup = served.clone();
    dedup.dedup();
    prop_assert_eq!(&served, &dedup, "a customer is served twice");
    let total: f64 = solution.routes().iter().map(|r| r.total_distance()).sum();
    prop_assert!((solution.total_distance() - total).abs() < EPS);
    Ok(served)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ── Split ────────────────────────────────────────────────────────────

    #[test]
    fn split_partitions_the_tour_in_order_within_capacity(
        customers in instance(1..=25usize, 20),
        slack in 0..40i32,
        seed in any::<u64>(),
    ) {
        let dm = DistanceMatrix::from_customers(&customers);
        let capacity = capacity_for(&customers, slack);
        let mut tour = all_customer_ids(&customers);
        // A deterministic shuffle of the giant tour from the seed.
        let mut state = seed | 1;
        for i in (1..tour.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            tour.swap(i, (state % (i as u64 + 1)) as usize);
        }

        let result = split(&tour, &customers, &dm, capacity);

        // The routes, concatenated, are the tour: each customer once, in order.
        let concatenated: Vec<usize> = result.routes.iter().flatten().copied().collect();
        prop_assert_eq!(&concatenated, &tour);

        let mut priced = 0.0;
        for route in &result.routes {
            prop_assert!(!route.is_empty(), "split produced an empty route");
            let load: i32 = route.iter().map(|&c| customers[c].demand()).sum();
            prop_assert!(load <= capacity, "route load {load} exceeds capacity {capacity}");
            priced += route_distance(route, 0, &dm);
        }
        prop_assert!(
            (result.total_distance - priced).abs() < EPS,
            "split reports {} but its routes cover {priced}",
            result.total_distance
        );
    }

    #[test]
    fn split_is_no_worse_than_one_customer_per_route(
        customers in instance(1..=20usize, 20),
        slack in 0..40i32,
    ) {
        let dm = DistanceMatrix::from_customers(&customers);
        let capacity = capacity_for(&customers, slack);
        let tour = all_customer_ids(&customers);
        let result = split(&tour, &customers, &dm, capacity);
        // Singleton routes are always feasible here, so the optimum is bounded
        // by them; a split that costs more has not found the optimum.
        let singletons: f64 = tour.iter().map(|&c| route_distance(&[c], 0, &dm)).sum();
        prop_assert!(result.total_distance <= singletons + EPS);
    }

    // ── Intra-route local search ─────────────────────────────────────────

    #[test]
    fn intra_route_moves_permute_and_never_worsen(
        customers in instance(1..=12usize, 10),
    ) {
        let dm = DistanceMatrix::from_customers(&customers);
        let route = all_customer_ids(&customers);
        let before = route_distance(&route, 0, &dm);

        for (name, improve) in [
            ("two_opt", two_opt_improve as fn(&[usize], usize, &DistanceMatrix) -> (Vec<usize>, f64)),
            ("or_opt", or_opt_improve),
            ("three_opt", three_opt_improve),
        ] {
            let (after_route, after_dist) = improve(&route, 0, &dm);
            prop_assert_eq!(sorted(after_route.clone()), route.clone(), "{} lost or duplicated a customer", name);
            let priced = route_distance(&after_route, 0, &dm);
            prop_assert!(
                (after_dist - priced).abs() < EPS,
                "{name} reports {after_dist} but the route it returned covers {priced}"
            );
            prop_assert!(after_dist <= before + EPS, "{name} lengthened the route: {before} -> {after_dist}");
        }
    }

    // ── Constructive heuristics ──────────────────────────────────────────

    #[test]
    fn constructive_heuristics_serve_each_customer_at_most_once_within_capacity(
        customers in instance(1..=20usize, 20),
        slack in 0..40i32,
    ) {
        let dm = DistanceMatrix::from_customers(&customers);
        let capacity = capacity_for(&customers, slack);
        let vehicle = Vehicle::new(0, capacity);
        let all = all_customer_ids(&customers);

        let cw = clarke_wright_savings(&customers, &dm, &vehicle);
        let served = check_solution(&cw, &customers, &dm, capacity)?;
        prop_assert_eq!(&served, &all, "savings left a customer unserved although every one fits alone");

        let sw = sweep(&customers, &dm, &vehicle);
        let served = check_solution(&sw, &customers, &dm, capacity)?;
        prop_assert_eq!(&served, &all, "sweep left a customer unserved although every one fits alone");

        // Nearest neighbour takes a fleet: with one vehicle per customer no
        // one can be left over.
        let fleet: Vec<Vehicle> = (0..customers.len()).map(|i| Vehicle::new(i, capacity)).collect();
        let nn = nearest_neighbor(&customers, &dm, &fleet);
        let served = check_solution(&nn, &customers, &dm, capacity)?;
        let mut expected = served.clone();
        expected.extend_from_slice(nn.unassigned());
        prop_assert_eq!(sorted(expected), all, "nearest neighbour served and unassigned do not partition the customers");
    }

    // ── Inter-route local search ─────────────────────────────────────────

    #[test]
    fn inter_route_moves_keep_the_served_set_and_never_worsen(
        customers in instance(2..=16usize, 20),
        slack in 0..40i32,
    ) {
        let dm = DistanceMatrix::from_customers(&customers);
        let capacity = capacity_for(&customers, slack);
        let vehicle = Vehicle::new(0, capacity);
        let initial = clarke_wright_savings(&customers, &dm, &vehicle);
        let served_before = check_solution(&initial, &customers, &dm, capacity)?;

        for (name, improve) in [
            ("relocate", relocate_improve as fn(&Solution, &[Customer], &DistanceMatrix, &Vehicle) -> Solution),
            ("exchange", exchange_improve),
        ] {
            let improved = improve(&initial, &customers, &dm, &vehicle);
            let served_after = check_solution(&improved, &customers, &dm, capacity)?;
            prop_assert_eq!(&served_after, &served_before, "{} changed the served set", name);
            prop_assert!(
                improved.total_distance() <= initial.total_distance() + EPS,
                "{name} lengthened the solution: {} -> {}",
                initial.total_distance(),
                improved.total_distance()
            );
        }
    }
}
