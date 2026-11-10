//! The constructive methods on a problem with soft windows: every customer served, every route
//! allowed, the solution priced by distance plus lateness cost, and the savings and insertion
//! methods only ever taking a step that pays for itself.

use super::{clarke_wright_savings, nearest_neighbor_tw, solomon_i1, sweep};
use crate::distance::DistanceMatrix;
use crate::evaluation::{effective_cost, RouteLimits};
use crate::models::{Customer, Solution, TimeWindow, Vehicle};

/// Eight customers around the depot, every window soft, some of them tight enough to be missed.
fn soft_problem(next: &mut dyn FnMut(f64) -> f64) -> (Vec<Customer>, DistanceMatrix) {
    let mut customers = vec![Customer::depot(0.0, 0.0)];
    for id in 1..=8 {
        let ready = next(10.0);
        customers.push(
            Customer::new(id, next(40.0) - 20.0, next(40.0) - 20.0, 3, next(4.0))
                .with_time_window(TimeWindow::new(ready, ready + next(12.0)).expect("valid"))
                .with_lateness_penalty(next(8.0))
                .expect("valid"),
        );
    }
    let dm = DistanceMatrix::from_customers(&customers);
    (customers, dm)
}

fn routes_of(solution: &Solution) -> Vec<Vec<usize>> {
    solution.routes().iter().map(|r| r.customer_ids()).collect()
}

fn total(routes: &[Vec<usize>], customers: &[Customer], dm: &DistanceMatrix) -> f64 {
    routes
        .iter()
        .map(|r| effective_cost(r, 0, dm, customers, &RouteLimits::NONE))
        .sum()
}

#[test]
fn every_method_serves_every_customer_and_prices_the_lateness() {
    let mut state = 0xd1b5_4a32_d192_ed03_u64;
    let mut next = move |bound: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * bound
    };
    let mut late_somewhere = 0;
    for _ in 0..30 {
        let (customers, dm) = soft_problem(&mut next);
        let vehicle = Vehicle::new(0, 9); // three customers a route
        let vehicles: Vec<Vehicle> = (0..8).map(|id| Vehicle::new(id, 9)).collect();
        for (name, solution) in [
            ("nn_tw", nearest_neighbor_tw(&customers, &dm, &vehicles)),
            (
                "clarke_wright",
                clarke_wright_savings(&customers, &dm, &vehicle),
            ),
            ("solomon_i1", solomon_i1(&customers, &dm, &vehicle)),
            ("sweep", sweep(&customers, &dm, &vehicle)),
        ] {
            assert_eq!(solution.num_unassigned(), 0, "{name} left a customer out");
            assert_eq!(solution.num_served(), 8, "{name}");
            let priced: f64 = solution
                .routes()
                .iter()
                .map(|r| r.total_distance() + r.lateness_cost())
                .sum();
            assert!(
                (solution.total_cost() - priced).abs() < 1e-9,
                "{name} priced it wrongly"
            );
            late_somewhere += solution
                .routes()
                .iter()
                .filter(|r| r.total_lateness() > 0.0)
                .count();
        }
    }
    assert!(
        late_somewhere > 20,
        "no route was ever late, so nothing was tested: {late_somewhere}"
    );
}

#[test]
fn savings_and_insertion_never_cost_more_than_a_route_for_each_customer() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |bound: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * bound
    };
    let (mut merged, mut inserted) = (0, 0);
    for _ in 0..40 {
        let (customers, dm) = soft_problem(&mut next);
        let vehicle = Vehicle::new(0, 1000);
        let apart: f64 = (1..=8).map(|c| total(&[vec![c]], &customers, &dm)).sum();

        let routes = routes_of(&clarke_wright_savings(&customers, &dm, &vehicle));
        assert!(
            total(&routes, &customers, &dm) <= apart + 1e-9,
            "savings made it dearer"
        );
        merged += usize::from(routes.len() < 8);

        let routes = routes_of(&solomon_i1(&customers, &dm, &vehicle));
        assert!(
            total(&routes, &customers, &dm) <= apart + 1e-9,
            "insertion made it dearer"
        );
        inserted += usize::from(routes.len() < 8);
    }
    assert!(
        merged > 10 && inserted > 10,
        "nothing was ever combined: {merged} {inserted}"
    );
}

#[test]
fn a_customer_the_clock_cannot_reach_in_time_is_served_late_at_a_soft_window_and_not_at_a_hard_one()
{
    // 20 away, due at 5: late whatever is done.
    let make = |penalty: Option<f64>| {
        let mut far = Customer::new(1, 20.0, 0.0, 1, 0.0)
            .with_time_window(TimeWindow::new(0.0, 5.0).expect("valid"));
        if let Some(p) = penalty {
            far = far.with_lateness_penalty(p).expect("valid");
        }
        let customers = vec![Customer::depot(0.0, 0.0), far];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    };
    let vehicles = vec![Vehicle::new(0, 10)];

    let (customers, dm) = make(None);
    assert_eq!(
        nearest_neighbor_tw(&customers, &dm, &vehicles).num_unassigned(),
        1
    );

    let (customers, dm) = make(Some(2.0));
    let solution = nearest_neighbor_tw(&customers, &dm, &vehicles);
    assert_eq!(solution.num_unassigned(), 0);
    let route = &solution.routes()[0];
    assert_eq!(
        (route.total_lateness(), route.lateness_cost()),
        (15.0, 30.0)
    );
    assert_eq!(solution.total_cost(), 40.0 + 30.0);
}
