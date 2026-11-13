//! Pickup-delivery pairs: the existing operators keep every pair whole and in order.
//!
//! Random problems with pairs (a pickup that loads `q`, its delivery that unloads it) and a plan
//! made of whole pairs. Every improvement the library offers -- the three intra-route moves,
//! relocation, cross-exchange, cross-depot moves, route limiting and balancing -- is applied, and
//! what comes out has to be a plan `check_solution` finds nothing wrong with. The operators know
//! nothing of pairs: what keeps them right is the one feasibility question they all ask.

use u_routing::balance::{improve as balance_improve, Balance, Quantity};
use u_routing::constructive::{
    clarke_wright_savings, construct_by_depot, nearest_neighbor, nearest_neighbor_tw,
};
use u_routing::distance::DistanceMatrix;
use u_routing::evaluation::{
    check_solution, pairs_ordered, route_eval, route_feasible, Fleet, RouteEvaluator, RouteLimits,
};
use u_routing::fleet::limit_routes;
use u_routing::local_search::{
    cross_depot_improve, exchange_improve, or_opt_improve, pair_relocate_improve, relocate_improve,
    three_opt_improve, two_opt_improve,
};
use u_routing::models::{Customer, Solution, TimeWindow, Vehicle, VehicleBreak};

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

/// The pairs and a few ordinary customers (they load and unload nothing but their own demand).
fn with_plain(customers: &[Customer], seed: u64) -> Vec<Customer> {
    let mut rng = Lcg(seed ^ 0x5555);
    let mut all = customers.to_vec();
    for _ in 0..(1 + rng.below(4)) {
        let id = all.len();
        all.push(Customer::new(
            id,
            rng.next() * 100.0,
            rng.next() * 100.0,
            1 + rng.below(4) as i32,
            1.0,
        ));
    }
    all
}

/// The same pairs on a map, each customer with a window of the given width around a time that
/// the trip to it takes; some pairs cannot be served at all.
fn with_windows(i: &Instance, width: f64, seed: u64) -> Vec<Customer> {
    let mut rng = Lcg(seed ^ 0xABCD);
    i.customers
        .iter()
        .map(|c| {
            if c.is_depot() {
                return c.clone();
            }
            let centre = 20.0 + rng.next() * 120.0;
            c.clone().with_time_window(
                TimeWindow::new((centre - width).max(0.0), centre + width).expect("valid"),
            )
        })
        .collect()
}

#[test]
fn the_nearest_neighbour_constructions_make_whole_routes() {
    for seed in 0..80 {
        let base = instance(seed);
        // Capacity small enough to split the pairs over several routes, and with a limit on the
        // route length now and then.
        let capacity = [6, 9, 14, 30][(seed % 4) as usize];
        let limit = seed % 3 == 0;
        let make = |id: usize| {
            let v = Vehicle::new(id, capacity);
            if limit {
                v.with_max_distance(260.0)
            } else {
                v
            }
        };
        let vehicle = make(0);
        let vehicles: Vec<Vehicle> = (0..6).map(make).collect();
        for (label, customers) in [
            ("no windows", base.customers.clone()),
            ("loose windows", with_windows(&base, 300.0, seed)),
            ("tight windows", with_windows(&base, 30.0, seed)),
        ] {
            let i = Instance {
                customers,
                dm: base.dm.clone(),
                vehicle: vehicle.clone(),
                routes: Vec::new(),
            };
            for (name, solution) in [
                ("nn", nearest_neighbor(&i.customers, &i.dm, &vehicles)),
                ("nn_tw", nearest_neighbor_tw(&i.customers, &i.dm, &vehicles)),
            ] {
                let fleet = Fleet {
                    vehicles: &vehicles,
                    counts: &[],
                    max_routes: None,
                };
                let found = check_solution(&i.customers, &i.dm, &fleet, &solution);
                // `nn` does not look at windows: only the pair and capacity rules are its.
                let found: Vec<_> = found
                    .into_iter()
                    .filter(|v| {
                        name == "nn_tw"
                            || !matches!(
                                v.kind,
                                u_routing::models::ViolationType::TimeWindowViolated { .. }
                            )
                    })
                    .collect();
                assert!(
                    found.is_empty(),
                    "{name} / {label} (seed {seed}, capacity {capacity}): {found:?}"
                );
            }
        }
    }
}

#[test]
fn a_vehicle_that_unloads_carries_the_next_pair_on_the_same_route() {
    // Two 6-unit pairs along a line, a vehicle of 6: one pair after the other is one route.
    let customers = vec![
        Customer::depot(0.0, 0.0),
        Customer::new(1, 1.0, 0.0, 6, 0.0).pickup_of(2),
        Customer::new(2, 2.0, 0.0, 0, 0.0).delivery_of(1),
        Customer::new(3, 3.0, 0.0, 6, 0.0).pickup_of(4),
        Customer::new(4, 4.0, 0.0, 0, 0.0).delivery_of(3),
    ];
    let dm = DistanceMatrix::from_customers(&customers);
    let vehicles = [Vehicle::new(0, 6), Vehicle::new(1, 6)];
    for (name, solution) in [
        ("nn", nearest_neighbor(&customers, &dm, &vehicles)),
        ("nn_tw", nearest_neighbor_tw(&customers, &dm, &vehicles)),
    ] {
        assert_eq!(solution.num_routes(), 1, "{name}");
        assert_eq!(solution.routes()[0].customer_ids(), [1, 2, 3, 4], "{name}");
    }
}

#[test]
fn the_savings_construction_merges_whole_pair_routes_forward() {
    for seed in 0..80 {
        let base = instance(seed);
        let capacity = [6, 9, 14, 30][(seed % 4) as usize];
        let mut vehicle = Vehicle::new(0, capacity);
        if seed % 3 == 0 {
            vehicle = vehicle.with_max_distance(260.0);
        }
        let vehicles = [vehicle.clone()];
        let base = if seed % 2 == 1 {
            let customers = with_plain(&base.customers, seed);
            let dm = DistanceMatrix::from_customers(&customers);
            Instance {
                customers,
                dm,
                ..base
            }
        } else {
            base
        };
        for (label, customers) in [
            ("no windows", base.customers.clone()),
            ("loose windows", with_windows(&base, 300.0, seed)),
            ("tight windows", with_windows(&base, 30.0, seed)),
        ] {
            let solution = clarke_wright_savings(&customers, &base.dm, &vehicle);
            let fleet = Fleet {
                vehicles: &vehicles,
                counts: &[],
                max_routes: None,
            };
            let found = check_solution(&customers, &base.dm, &fleet, &solution);
            assert!(
                found.is_empty(),
                "savings / {label} (seed {seed}, capacity {capacity}): {found:?}"
            );
        }
    }
}

#[test]
fn savings_chains_pairs_when_the_vehicle_unloads_between_them() {
    // Two 6-unit pairs on a line, a vehicle of 6: the pairs merge into one route (the sum 12
    // would have forbidden it; the peak is 6).
    let customers = vec![
        Customer::depot(0.0, 0.0),
        Customer::new(1, 1.0, 0.0, 6, 0.0).pickup_of(2),
        Customer::new(2, 2.0, 0.0, 0, 0.0).delivery_of(1),
        Customer::new(3, 3.0, 0.0, 6, 0.0).pickup_of(4),
        Customer::new(4, 4.0, 0.0, 0, 0.0).delivery_of(3),
    ];
    let dm = DistanceMatrix::from_customers(&customers);
    let solution = clarke_wright_savings(&customers, &dm, &Vehicle::new(0, 6));
    assert_eq!(solution.num_routes(), 1);
    assert_eq!(solution.routes()[0].customer_ids(), [1, 2, 3, 4]);
}

/// Two depots (rows 0 and 1) and pairs between them (pickups at 2, 4, ..., deliveries at 3, 5, ...).
fn depot_instance(seed: u64, depots: usize) -> (Vec<Customer>, DistanceMatrix) {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let pairs = 3 + rng.below(6);
    let mut customers: Vec<Customer> = (0..depots)
        .map(|d| Customer::depot_at(d, 15.0 + 70.0 * d as f64 / (depots - 1) as f64, 50.0))
        .collect();
    for p in 0..pairs {
        let (pickup, delivery) = (depots + 2 * p, depots + 1 + 2 * p);
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
    (customers, dm)
}

#[test]
fn several_depots_keep_a_pair_at_one_depot() {
    for seed in 0..90 {
        let depots = 2 + (seed % 3) as usize % 2 + usize::from(seed % 3 == 2);
        let (customers, dm) = depot_instance(seed, depots);
        let capacity = [6, 9, 30][(seed % 3) as usize];
        // Three vehicles at each depot; a limit now and then sends a pair to the other depot.
        let limit = seed % 2 == 0;
        let vehicles: Vec<Vehicle> = (0..3 * depots)
            .map(|v| {
                let x = Vehicle::new(v, capacity).with_depot(v / 3);
                if limit {
                    x.with_max_distance(150.0)
                } else {
                    x
                }
            })
            .collect();
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: None,
        };
        let by_nn = construct_by_depot(&customers, &dm, &vehicles, |c, d, v| {
            nearest_neighbor(c, d, v)
        });
        let found = check_solution(&customers, &dm, &fleet, &by_nn);
        assert!(
            found.is_empty(),
            "nn by depot (seed {seed}, capacity {capacity}): {found:?}"
        );
        let by_savings = construct_by_depot(&customers, &dm, &vehicles, |c, d, v| {
            clarke_wright_savings(c, d, &v[0])
        });
        let found = check_solution(&customers, &dm, &fleet, &by_savings);
        assert!(
            found.is_empty(),
            "savings by depot (seed {seed}, capacity {capacity}): {found:?}"
        );
    }
}

#[test]
fn moving_a_pair_as_a_pair_improves_a_plan_the_single_moves_cannot() {
    let (mut better, mut same) = (0, 0);
    for seed in 0..60 {
        let i = instance(seed);
        if i.routes.len() < 2 {
            continue;
        }
        let start = plan(&i, &i.routes, &[]);
        // The single-customer moves cannot touch a pair: relocation leaves the plan as it was.
        let single = relocate_improve(&start, &i.customers, &i.dm, &i.vehicle);
        assert!(
            (single.total_distance() - start.total_distance()).abs() < 1e-9,
            "seed {seed}"
        );

        let moved = pair_relocate_improve(&start, &i.customers, &i.dm, &i.vehicle);
        assert_clean(&i, &moved, "pair_relocate", seed);
        assert!(
            moved.total_distance() <= start.total_distance() + 1e-9,
            "pair_relocate made it longer (seed {seed})"
        );
        assert_eq!(
            served(
                &moved
                    .routes()
                    .iter()
                    .map(|r| r.customer_ids())
                    .collect::<Vec<_>>()
            ),
            served(&i.routes)
        );
        if moved.total_distance() < start.total_distance() - 1e-9 {
            better += 1;
        } else {
            same += 1;
        }
    }
    assert!(
        better >= 20,
        "only {better} of {} plans improved",
        better + same
    );
}

#[test]
fn pair_relocation_after_the_constructions_keeps_the_plan_clean() {
    for seed in 0..60 {
        let base = instance(seed);
        let capacity = [6, 9, 14, 30][(seed % 4) as usize];
        let vehicle = Vehicle::new(0, capacity);
        let vehicles: Vec<Vehicle> = (0..6).map(|v| Vehicle::new(v, capacity)).collect();
        let fleet = Fleet {
            vehicles: &vehicles,
            counts: &[],
            max_routes: None,
        };
        for (name, start) in [
            ("nn", nearest_neighbor(&base.customers, &base.dm, &vehicles)),
            (
                "savings",
                clarke_wright_savings(&base.customers, &base.dm, &vehicle),
            ),
        ] {
            let moved = pair_relocate_improve(&start, &base.customers, &base.dm, &vehicle);
            let found = check_solution(&base.customers, &base.dm, &fleet, &moved);
            assert!(
                found.is_empty(),
                "{name} + pair_relocate (seed {seed}): {found:?}"
            );
            assert!(
                moved.total_distance() <= start.total_distance() + 1e-9,
                "{name} (seed {seed})"
            );
        }
    }
}

#[test]
fn a_plan_with_half_a_pair_is_left_alone_by_pair_relocation() {
    let i = instance(3);
    // Pickup 1 on a route, its delivery 2 reported unassigned: not a plan the move should touch.
    let routes: Vec<Vec<usize>> = vec![vec![1], vec![3, 4]];
    let start = plan(&i, &routes, &[2]);
    let after = pair_relocate_improve(&start, &i.customers, &i.dm, &i.vehicle);
    let ids: Vec<Vec<usize>> = after.routes().iter().map(|r| r.customer_ids()).collect();
    assert_eq!(served(&ids), [1, 3, 4]);
    assert_eq!(after.unassigned(), [2]);
}
