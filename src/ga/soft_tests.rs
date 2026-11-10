//! The genetic algorithm's split and fitness on a problem with soft windows: the split is the best
//! segmentation of the tour by distance plus lateness cost (checked against every segmentation),
//! and the fitness is that cost, not the distance.

use super::{split_tw, GiantTour, RoutingGaProblem};
use crate::distance::DistanceMatrix;
use crate::evaluation::{effective_cost, RouteLimits};
use crate::models::{Customer, TimeWindow};
use u_metaheur::ga::GaProblem;

fn soft_problem(next: &mut dyn FnMut(f64) -> f64, soft: bool) -> (Vec<Customer>, DistanceMatrix) {
    let mut customers = vec![Customer::depot(0.0, 0.0)];
    for id in 1..=7 {
        let ready = next(8.0);
        let mut c = Customer::new(id, next(30.0) - 15.0, next(30.0) - 15.0, 1, next(3.0))
            .with_time_window(TimeWindow::new(ready, ready + next(10.0)).expect("valid"));
        if soft {
            c = c.with_lateness_penalty(next(6.0)).expect("valid");
        }
        customers.push(c);
    }
    let dm = DistanceMatrix::from_customers(&customers);
    (customers, dm)
}

/// The cheapest way to cut `tour` into routes, by trying every set of cuts: each route costs
/// `route_cost` plus its distance and lateness cost, and a route that is not allowed (a late
/// arrival at a hard window) is out.
fn best_segmentation(
    tour: &[usize],
    customers: &[Customer],
    dm: &DistanceMatrix,
    route_cost: f64,
) -> f64 {
    let n = tour.len();
    let mut best = f64::INFINITY;
    for cuts in 0..1u32 << (n - 1) {
        let mut total = 0.0;
        let mut start = 0;
        for end in 1..=n {
            if end == n || cuts >> (end - 1) & 1 == 1 {
                total += route_cost
                    + effective_cost(&tour[start..end], 0, dm, customers, &RouteLimits::NONE);
                start = end;
            }
        }
        best = best.min(total);
    }
    best
}

#[test]
fn the_split_is_the_cheapest_segmentation_by_distance_plus_lateness() {
    let mut state = 0x5851_f42d_4c95_7f2d_u64;
    let mut next = move |bound: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * bound
    };
    let mut cut_somewhere = 0;
    for round in 0..40 {
        let (customers, dm) = soft_problem(&mut next, true);
        let tour: Vec<usize> = (1..=7).collect();
        let route_cost = [0.0, 3.0, 20.0][round % 3];
        let result = split_tw(&tour, &customers, &dm, 100, &RouteLimits::NONE, route_cost);
        let priced: f64 = result
            .routes
            .iter()
            .map(|r| route_cost + effective_cost(r, 0, &dm, &customers, &RouteLimits::NONE))
            .sum();
        let best = best_segmentation(&tour, &customers, &dm, route_cost);
        assert!(
            (priced - best).abs() < 1e-9,
            "split {priced} against the best {best}"
        );
        let flat: Vec<usize> = result.routes.iter().flatten().copied().collect();
        assert_eq!(flat, tour);
        cut_somewhere += usize::from(result.routes.len() > 1);
    }
    assert!(cut_somewhere > 5, "the split never cut: {cut_somewhere}");
}

#[test]
fn the_fitness_of_a_soft_problem_is_the_priced_cost_of_its_split() {
    let mut state = 0x14057b7ef767814f_u64;
    let mut next = move |bound: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * bound
    };
    for _ in 0..30 {
        let (customers, dm) = soft_problem(&mut next, true);
        let tour: Vec<usize> = (1..=7).collect();
        let best = best_segmentation(&tour, &customers, &dm, 0.0);

        // Without the 2-opt the fitness is exactly the best segmentation's cost.
        let plain =
            RoutingGaProblem::new(customers.clone(), dm.clone(), 100).without_local_search();
        let fitness = plain.evaluate(&GiantTour::new(tour.clone()));
        assert!(
            (fitness - best).abs() < 1e-9,
            "fitness {fitness} against {best}"
        );

        // With it, a route is only ever improved by the priced cost.
        let searched = RoutingGaProblem::new(customers, dm, 100);
        assert!(searched.evaluate(&GiantTour::new(tour)) <= best + 1e-9);
    }
}

#[test]
fn an_unserved_customer_outranks_the_costliest_complete_solution() {
    let mut state = 0xa076_1d64_78bd_642f_u64;
    let mut next = move |bound: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * bound
    };
    for _ in 0..30 {
        let (customers, dm) = soft_problem(&mut next, true);
        let penalty = RoutingGaProblem::new(customers.clone(), dm.clone(), 100).unserved_penalty();
        // Every customer on one route, in a random order: about the latest anyone can be.
        for _ in 0..20 {
            let mut tour: Vec<usize> = (1..=7).collect();
            for i in (1..tour.len()).rev() {
                tour.swap(i, (next(1.0) * (i + 1) as f64) as usize % (i + 1));
            }
            let cost = effective_cost(&tour, 0, &dm, &customers, &RouteLimits::NONE);
            assert!(
                cost < penalty,
                "a complete solution costs {cost}, a customer left out {penalty}"
            );
        }
    }
}
