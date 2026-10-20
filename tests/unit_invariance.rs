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
