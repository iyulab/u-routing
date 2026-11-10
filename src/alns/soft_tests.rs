//! ALNS and the fleet's insertion on a problem with soft windows: an insertion costs the change in
//! the route's distance plus lateness cost, the solution carries that cost, and a run ends no
//! worse than it began.

use super::destroy::RandomRemoval;
use super::repair::{GreedyInsertion, RegretInsertion};
use super::{RoutingAlnsProblem, RoutingSolution};
use crate::distance::DistanceMatrix;
use crate::evaluation::{effective_cost, RouteLimits};
use crate::fleet::cheapest_insertion;
use crate::models::{Customer, TimeWindow};
use u_metaheur::alns::{AlnsConfig, AlnsProblem, AlnsRunner};

fn soft_problem(next: &mut dyn FnMut(f64) -> f64) -> (Vec<Customer>, DistanceMatrix) {
    let mut customers = vec![Customer::depot(0.0, 0.0)];
    for id in 1..=8 {
        let ready = next(8.0);
        customers.push(
            Customer::new(id, next(30.0) - 15.0, next(30.0) - 15.0, 1, next(3.0))
                .with_time_window(TimeWindow::new(ready, ready + next(10.0)).expect("valid"))
                .with_lateness_penalty(next(6.0))
                .expect("valid"),
        );
    }
    let dm = DistanceMatrix::from_customers(&customers);
    (customers, dm)
}

fn rng_stream(seed: u64) -> impl FnMut(f64) -> f64 {
    let mut state = seed;
    move |bound: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * bound
    }
}

#[test]
fn the_cheapest_insertion_is_the_least_change_in_distance_plus_lateness() {
    let mut next = rng_stream(0x3c6e_f372_fe94_f82b);
    let none = RouteLimits::NONE;
    let mut differs_from_distance = 0;
    for _ in 0..40 {
        let (customers, dm) = soft_problem(&mut next);
        let routes = vec![vec![1, 2, 3], vec![4, 5]];
        let customer = 6;
        let found = cheapest_insertion(&routes, |_| 100, |_| none, customer, &dm, &customers)
            .expect("a soft window leaves every position open");

        // Every position of every route, by the whole route.
        let mut best = (usize::MAX, usize::MAX, f64::INFINITY);
        for (ri, route) in routes.iter().enumerate() {
            let base = effective_cost(route, 0, &dm, &customers, &none);
            for pos in 0..=route.len() {
                let mut with = route.clone();
                with.insert(pos, customer);
                let cost = effective_cost(&with, 0, &dm, &customers, &none) - base;
                if cost < best.2 {
                    best = (ri, pos, cost);
                }
            }
        }
        assert_eq!((found.0, found.1), (best.0, best.1));
        assert!((found.2 - best.2).abs() < 1e-9);

        // The same customer by the change in distance alone: the lateness can change the choice.
        let mut by_distance = (usize::MAX, usize::MAX, f64::INFINITY);
        for (ri, route) in routes.iter().enumerate() {
            for pos in 0..=route.len() {
                let prev = if pos == 0 { 0 } else { route[pos - 1] };
                let next_stop = route.get(pos).copied().unwrap_or(0);
                let d =
                    dm.get(prev, customer) + dm.get(customer, next_stop) - dm.get(prev, next_stop);
                if d < by_distance.2 {
                    by_distance = (ri, pos, d);
                }
            }
        }
        differs_from_distance += usize::from((by_distance.0, by_distance.1) != (found.0, found.1));
    }
    assert!(
        differs_from_distance > 3,
        "lateness never changed a choice: {differs_from_distance}"
    );
}

#[test]
fn a_solution_carries_the_cost_of_its_lateness() {
    let mut next = rng_stream(0xbb67_ae85_84ca_a73b);
    let none = RouteLimits::NONE;
    for _ in 0..20 {
        let (customers, dm) = soft_problem(&mut next);
        let routes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let mut solution = RoutingSolution::new(routes.clone(), vec![], &customers, &dm);
        let priced: f64 = routes
            .iter()
            .map(|r| effective_cost(r, 0, &dm, &customers, &none))
            .sum();
        assert!((solution.total_distance() + solution.lateness_cost() - priced).abs() < 1e-9);

        // Moving a customer and recalculating keeps the two in step.
        let moved = solution.routes_mut()[0].pop().expect("a customer");
        solution.routes_mut()[1].insert(0, moved);
        solution.recalculate(&dm, &customers);
        let priced: f64 = solution
            .routes()
            .iter()
            .map(|r| effective_cost(r, 0, &dm, &customers, &none))
            .sum();
        assert!((solution.total_distance() + solution.lateness_cost() - priced).abs() < 1e-9);
    }
}

#[test]
fn a_run_serves_everyone_and_ends_no_worse_than_it_began() {
    let mut next = rng_stream(0x3c6e_f372_fe94_f82b ^ 0x9e37);
    for round in 0..6 {
        let (customers, dm) = soft_problem(&mut next);
        let problem = RoutingAlnsProblem::new(customers.clone(), dm.clone(), 4);
        let mut rng = u_numflow::random::create_rng(round);
        let start = problem.cost(&problem.initial_solution(&mut rng));

        let destroy = vec![RandomRemoval];
        let repair = vec![
            GreedyInsertion::new(dm.clone(), customers.clone(), 4),
            GreedyInsertion::new(dm.clone(), customers.clone(), 4),
        ];
        let config = AlnsConfig::default()
            .with_max_iterations(150)
            .with_seed(round);
        let result = AlnsRunner::run(&problem, &destroy, &repair, &config).expect("runs");
        assert!(result.best.unassigned().is_empty(), "round {round}");
        assert!(
            result.best_cost <= start + 1e-9,
            "{} > {start}",
            result.best_cost
        );

        // The cost it reports is the routes' distance plus their lateness cost.
        let priced: f64 = result
            .best
            .routes()
            .iter()
            .map(|r| effective_cost(r, 0, &dm, &customers, &RouteLimits::NONE))
            .sum();
        assert!(
            (result.best_cost - priced).abs() < 1e-9,
            "{} against {priced}",
            result.best_cost
        );

        // Regret insertion runs on the same problem.
        let repair = vec![RegretInsertion::new(dm.clone(), customers.clone(), 4)];
        let result = AlnsRunner::run(&problem, &destroy, &repair, &config).expect("runs");
        assert!(result.best.unassigned().is_empty());
    }
}
