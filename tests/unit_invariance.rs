//! Moving the same customers to a map a million times smaller (or larger) must
//! not change which routes a deterministic heuristic builds: distances scale,
//! the order of every comparison between them does not.

use u_routing::constructive::{clarke_wright_savings, nearest_neighbor};
use u_routing::distance::DistanceMatrix;
use u_routing::models::{Customer, Vehicle};

fn customers(scale: f64) -> Vec<Customer> {
    // The depot, then 24 customers on a deterministic scatter.
    let mut list = vec![Customer::new(0, 50.0 * scale, 50.0 * scale, 0, 0.0)];
    for i in 1..=24usize {
        let t = i as f64;
        let x = 50.0 + 40.0 * (t * 0.83).sin() * (1.0 + 0.2 * (t * 0.37).cos());
        let y = 50.0 + 40.0 * (t * 1.17).cos() * (1.0 + 0.2 * (t * 0.51).sin());
        list.push(Customer::new(
            i,
            x * scale,
            y * scale,
            3 + (i % 5) as i32,
            0.0,
        ));
    }
    list
}

fn routes_of(solution: &u_routing::models::Solution) -> Vec<Vec<usize>> {
    solution.routes().iter().map(|r| r.customer_ids()).collect()
}

#[test]
fn the_heuristics_build_the_same_routes_on_a_map_of_any_size() {
    let vehicles: Vec<Vehicle> = (0..6).map(|i| Vehicle::new(i, 25)).collect();
    let base = customers(1.0);
    let dm = DistanceMatrix::from_customers(&base);
    let nn_ref = routes_of(&nearest_neighbor(&base, &dm, &vehicles));
    let cw_ref = routes_of(&clarke_wright_savings(&base, &dm, &vehicles[0]));
    assert!(
        nn_ref.len() > 1 && cw_ref.len() > 1,
        "the fixture needs several routes"
    );
    for scale in [1e-6, 1e-9, 1e6] {
        let list = customers(scale);
        let dm = DistanceMatrix::from_customers(&list);
        assert_eq!(
            routes_of(&nearest_neighbor(&list, &dm, &vehicles)),
            nn_ref,
            "nearest neighbour, scale {scale}"
        );
        assert_eq!(
            routes_of(&clarke_wright_savings(&list, &dm, &vehicles[0])),
            cw_ref,
            "savings, scale {scale}"
        );
    }
}

/// The same holds for the local search: a gain is a gain on a map of any size,
/// so the moves taken -- and the routes they end in -- are the same.
#[test]
fn the_local_search_takes_the_same_moves_on_a_map_of_any_size() {
    use u_routing::evaluation::RouteLimits;
    use u_routing::local_search::{
        exchange_improve, or_opt_improve, relocate_improve, three_opt_improve, two_opt_improve,
    };

    let vehicles: Vec<Vehicle> = (0..6).map(|i| Vehicle::new(i, 25)).collect();
    // The id order is a poor tour; the nearest-neighbour routes leave work for
    // the inter-route moves.
    let scrambled: Vec<usize> = (1..=24).collect();
    let run = |scale: f64| {
        let list = customers(scale);
        let dm = DistanceMatrix::from_customers(&list);
        let start = nearest_neighbor(&list, &dm, &vehicles);
        let limits = RouteLimits::NONE;
        (
            routes_of(&relocate_improve(&start, &list, &dm, &vehicles[0])),
            routes_of(&exchange_improve(&start, &list, &dm, &vehicles[0])),
            two_opt_improve(&scrambled, 0, &dm, &list, &limits).0,
            or_opt_improve(&scrambled, 0, &dm, &list, &limits).0,
            three_opt_improve(&scrambled, 0, &dm, &list, &limits).0,
        )
    };
    let reference = run(1.0);
    assert_ne!(
        reference.2, scrambled,
        "the fixture leaves 2-opt work to do"
    );
    for scale in [1e-12, 1e-9, 1e-6, 1e6, 1e9] {
        let got = run(scale);
        assert_eq!(got.0, reference.0, "relocate, scale {scale}");
        assert_eq!(got.1, reference.1, "exchange, scale {scale}");
        assert_eq!(got.2, reference.2, "2-opt, scale {scale}");
        assert_eq!(got.3, reference.3, "or-opt, scale {scale}");
        assert_eq!(got.4, reference.4, "3-opt, scale {scale}");
    }
}

/// The checker's tolerance on a route's reported distance is a share of the
/// distance, so a plan is right -- and a wrong figure wrong -- on a map of any
/// size.
#[test]
fn the_checker_judges_a_reported_distance_on_a_map_of_any_size() {
    use u_routing::evaluation::{check_solution, Fleet, VehicleUse};
    use u_routing::models::ViolationType;

    let vehicles: Vec<Vehicle> = (0..6).map(|i| Vehicle::new(i, 25)).collect();
    let fleet = Fleet {
        vehicles: &vehicles,
        usage: VehicleUse::AnyNumber,
        max_routes: None,
    };
    for scale in [1e-12, 1e-9, 1e-6, 1.0, 1e6, 1e9] {
        let list = customers(scale);
        let dm = DistanceMatrix::from_customers(&list);
        let mut plan = clarke_wright_savings(&list, &dm, &vehicles[0]);
        assert!(
            check_solution(&list, &dm, &fleet, &plan).is_empty(),
            "a correct plan, scale {scale}"
        );
        let reported = plan.routes()[0].total_distance();
        plan.routes_mut()[0].set_total_distance(reported * 1.001);
        let violations = check_solution(&list, &dm, &fleet, &plan);
        assert!(
            violations
                .iter()
                .any(|v| matches!(v.kind, ViolationType::DistanceMismatch { .. })),
            "a figure 0.1 % off is caught, scale {scale}"
        );
    }
}
