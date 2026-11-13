//! Pickup-delivery pairs: the existing operators keep every pair whole and in order.
//!
//! Random problems with pairs (a pickup that loads `q`, its delivery that unloads it) and a plan
//! made of whole pairs. Every improvement the library offers -- the three intra-route moves,
//! relocation, cross-exchange, cross-depot moves, route limiting and balancing -- is applied, and
//! what comes out has to be a plan `check_solution` finds nothing wrong with. The operators know
//! nothing of pairs: what keeps them right is the one feasibility question they all ask.

use u_routing::balance::{improve as balance_improve, Balance, Quantity};
use u_routing::distance::DistanceMatrix;
use u_routing::evaluation::{
    check_solution, pairs_ordered, route_eval, route_feasible, Fleet, RouteEvaluator, RouteLimits,
};
use u_routing::fleet::limit_routes;
use u_routing::local_search::{
    cross_depot_improve, exchange_improve, or_opt_improve, relocate_improve, three_opt_improve,
    two_opt_improve,
};
use u_routing::models::{Customer, Solution, Vehicle, VehicleBreak};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n - 1)
    }
}

struct Instance {
    customers: Vec<Customer>,
    dm: DistanceMatrix,
    vehicle: Vehicle,
    /// The plan: whole pairs on each route, pickup before delivery.
    routes: Vec<Vec<usize>>,
}

/// Pickups at 1, 3, 5, ... and their deliveries at 2, 4, 6, ...; pairs dealt to `k` routes.
fn instance(seed: u64) -> Instance {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let pairs = 3 + rng.below(5);
    let mut customers = vec![Customer::depot(50.0, 50.0)];
    for p in 0..pairs {
        let (pickup, delivery) = (1 + 2 * p, 2 + 2 * p);
        let q = 1 + rng.below(5) as i32;
        customers.push(
            Customer::new(pickup, rng.next() * 100.0, rng.next() * 100.0, q, 1.0)
                .pickup_of(delivery),
        );
        customers.push(
            Customer::new(delivery, rng.next() * 100.0, rng.next() * 100.0, 0, 1.0)
                .delivery_of(pickup),
        );
    }
    let dm = DistanceMatrix::from_customers(&customers);
    let vehicle = Vehicle::new(0, 30);
    let k = 2 + rng.below(3);
    let mut routes = vec![Vec::new(); k];
    for p in 0..pairs {
        let r = rng.below(k);
        // Alternate between one pair after the other and nested pairs.
        if rng.below(2) == 0 || routes[r].is_empty() {
            routes[r].extend([1 + 2 * p, 2 + 2 * p]);
        } else {
            let first = routes[r][0];
            let at = routes[r].iter().position(|&c| c == first + 1).unwrap_or(0);
            routes[r].insert(at, 1 + 2 * p);
            routes[r].insert(at + 1, 2 + 2 * p);
        }
    }
    routes.retain(|r| !r.is_empty());
    Instance {
        customers,
        dm,
        vehicle,
        routes,
    }
}

fn plan(i: &Instance, routes: &[Vec<usize>], unassigned: &[usize]) -> Solution {
    let mut solution = Solution::new();
    for ids in routes {
        solution.add_route(
            RouteEvaluator::new(&i.customers, &i.dm, &i.vehicle)
                .build_route(ids)
                .0,
        );
    }
    for &u in unassigned {
        solution.add_unassigned(u);
    }
    solution
}

fn assert_clean(i: &Instance, solution: &Solution, what: &str, seed: u64) {
    let vehicles = [i.vehicle.clone()];
    let fleet = Fleet {
        vehicles: &vehicles,
        counts: &[],
        max_routes: None,
    };
    let found = check_solution(&i.customers, &i.dm, &fleet, solution);
    assert!(found.is_empty(), "{what} (seed {seed}): {found:?}");
}

fn served(routes: &[Vec<usize>]) -> Vec<usize> {
    let mut all: Vec<usize> = routes.iter().flatten().copied().collect();
    all.sort_unstable();
    all
}

#[test]
fn the_starting_plans_are_clean() {
    for seed in 0..60 {
        let i = instance(seed);
        assert_clean(&i, &plan(&i, &i.routes, &[]), "start", seed);
    }
}

#[test]
fn a_delivery_before_its_pickup_or_alone_is_not_a_route() {
    let i = instance(1);
    let limits = RouteLimits::NONE;
    assert!(route_feasible(&[1, 2], 0, &i.dm, &i.customers, &limits));
    assert!(!route_feasible(&[2, 1], 0, &i.dm, &i.customers, &limits));
    assert!(!route_feasible(&[1], 0, &i.dm, &i.customers, &limits));
    assert!(!route_feasible(&[2], 0, &i.dm, &i.customers, &limits));
    assert!(pairs_ordered(&[], &i.customers));
    // `route_eval` answers the same question (it is `Some` exactly when the route is feasible).
    for route in [
        &[1, 2][..],
        &[2, 1],
        &[1],
        &[2],
        &[1, 2, 3, 4],
        &[1, 3, 4, 2],
    ] {
        assert_eq!(
            route_eval(route, 0, &i.dm, &i.customers, &limits).is_some(),
            route_feasible(route, 0, &i.dm, &i.customers, &limits),
            "{route:?}"
        );
    }
    assert!(route_eval(&[2, 1], 0, &i.dm, &i.customers, &limits).is_none());

    // A driver's break takes another path through both functions: the pairs count there too.
    let with_break = RouteLimits {
        rest: Some(VehicleBreak::new(0.0, 1000.0, 5.0)),
        ..RouteLimits::NONE
    };
    assert!(route_feasible(&[1, 2], 0, &i.dm, &i.customers, &with_break));
    assert!(!route_feasible(
        &[2, 1],
        0,
        &i.dm,
        &i.customers,
        &with_break
    ));
    assert!(!route_feasible(&[1], 0, &i.dm, &i.customers, &with_break));
    assert!(route_eval(&[1, 2], 0, &i.dm, &i.customers, &with_break).is_some());
    assert!(route_eval(&[2, 1], 0, &i.dm, &i.customers, &with_break).is_none());
}

#[test]
fn the_intra_route_moves_keep_every_pair_whole_and_in_order() {
    let limits = RouteLimits::NONE;
    for seed in 0..60 {
        let i = instance(seed);
        for route in &i.routes {
            type Improve = fn(
                &[usize],
                usize,
                &DistanceMatrix,
                &[Customer],
                &RouteLimits,
            ) -> (Vec<usize>, f64);
            let moves: [(&str, Improve); 3] = [
                ("two_opt", two_opt_improve),
                ("or_opt", or_opt_improve),
                ("three_opt", three_opt_improve),
            ];
            for (name, improve) in moves {
                let (after, _) = improve(route, 0, &i.dm, &i.customers, &limits);
                assert!(
                    pairs_ordered(&after, &i.customers),
                    "{name} (seed {seed}): {after:?}"
                );
                let (mut a, mut b) = (after.clone(), route.clone());
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "{name} (seed {seed}) changed the customers");
            }
        }
    }
}

#[test]
fn the_inter_route_moves_keep_the_plan_clean() {
    for seed in 0..60 {
        let i = instance(seed);
        let start = plan(&i, &i.routes, &[]);
        let vehicles = [i.vehicle.clone()];

        let moved = relocate_improve(&start, &i.customers, &i.dm, &i.vehicle);
        assert_clean(&i, &moved, "relocate", seed);
        let moved = exchange_improve(&start, &i.customers, &i.dm, &i.vehicle);
        assert_clean(&i, &moved, "exchange", seed);
        let moved = cross_depot_improve(&start, &i.customers, &i.dm, &vehicles);
        assert_clean(&i, &moved, "cross_depot", seed);
    }
}

#[test]
fn limiting_and_balancing_keep_every_pair_together() {
    for seed in 0..60 {
        let i = instance(seed);
        let fleet = [i.vehicle.clone()];

        // Limiting: customers nobody could take go out -- a pair goes out whole or not at all.
        let kinds = vec![0; i.routes.len()];
        let limited = limit_routes(
            i.routes.clone(),
            kinds,
            &fleet,
            i.routes.len().saturating_sub(1).max(1),
            &i.dm,
            &i.customers,
        );
        let plan_after = plan(&i, &limited.routes, &limited.unserved);
        assert_clean(&i, &plan_after, "limit_routes", seed);

        // Balancing.
        for quantity in [Quantity::Load, Quantity::Distance] {
            let (routes, _) = balance_improve(
                i.routes.clone(),
                vec![0; i.routes.len()],
                &fleet,
                &i.customers,
                &i.dm,
                Balance {
                    quantity,
                    weight: 1.0,
                },
            );
            assert_eq!(
                served(&routes),
                served(&i.routes),
                "balance changed the customers (seed {seed})"
            );
            assert_clean(&i, &plan(&i, &routes, &[]), "balance", seed);
        }
    }
}
