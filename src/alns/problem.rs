//! ALNS problem definition for vehicle routing.
//!
//! Implements the [`AlnsProblem`](u_metaheur::alns::AlnsProblem) trait
//! for capacitated VRP, enabling the ALNS runner to optimize routing solutions.

use rand::Rng;
use u_metaheur::alns::AlnsProblem;

use crate::constructive::{nearest_neighbor, nearest_neighbor_tw};
use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, RouteLimits};
use crate::models::{Customer, Vehicle};

use super::solution_repr::RoutingSolution;

/// ALNS problem for capacitated vehicle routing.
///
/// # Examples
///
/// ```
/// use u_routing::models::{Customer, Vehicle};
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::alns::{RoutingAlnsProblem, RoutingSolution};
/// use u_routing::alns::destroy::RandomRemoval;
/// use u_routing::alns::repair::GreedyInsertion;
/// use u_metaheur::alns::{AlnsConfig, AlnsRunner, AlnsProblem};
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 3.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// let capacity = 30;
///
/// let problem = RoutingAlnsProblem::new(customers.clone(), dm.clone(), capacity);
/// let destroy_ops = vec![RandomRemoval];
/// let repair_ops = vec![GreedyInsertion::new(dm, customers, capacity)];
/// let config = AlnsConfig::default()
///     .with_max_iterations(100)
///     .with_seed(42);
///
/// let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config).unwrap();
/// assert!(result.best_cost < f64::INFINITY);
/// ```
pub struct RoutingAlnsProblem {
    customers: Vec<Customer>,
    distances: DistanceMatrix,
    capacity: i32,
    limits: RouteLimits,
}

impl RoutingAlnsProblem {
    /// Creates a new routing ALNS problem.
    pub fn new(customers: Vec<Customer>, distances: DistanceMatrix, capacity: i32) -> Self {
        Self {
            customers,
            distances,
            capacity,
            limits: RouteLimits::NONE,
        }
    }

    /// Plans every route within these distance and duration limits (none by
    /// default). Give the repair operators the same limits.
    pub fn with_limits(mut self, limits: RouteLimits) -> Self {
        self.limits = limits;
        self
    }
}

impl AlnsProblem for RoutingAlnsProblem {
    type Solution = RoutingSolution;

    fn initial_solution<R: Rng>(&self, _rng: &mut R) -> RoutingSolution {
        // Use nearest neighbor heuristic for initial solution
        let vehicles: Vec<Vehicle> = (0..self.customers.len())
            .map(|i| {
                let v = Vehicle::new(i, self.capacity);
                let v = match self.limits.max_distance {
                    Some(d) => v.with_max_distance(d),
                    None => v,
                };
                match self.limits.max_duration {
                    Some(t) => v.with_max_duration(t),
                    None => v,
                }
            })
            .collect();

        // With time windows or limits the seed must already respect them: the
        // repair operators only ever insert feasibly, so a late or too-long
        // seed would stay that way.
        let nn_sol = if route_checks_needed(&self.customers, &self.limits) {
            nearest_neighbor_tw(&self.customers, &self.distances, &vehicles)
        } else {
            nearest_neighbor(&self.customers, &self.distances, &vehicles)
        };

        let routes: Vec<Vec<usize>> = nn_sol.routes().iter().map(|r| r.customer_ids()).collect();
        let unassigned: Vec<usize> = nn_sol.unassigned().to_vec();

        RoutingSolution::new(routes, unassigned, &self.customers, &self.distances)
    }

    fn cost(&self, solution: &RoutingSolution) -> f64 {
        // Penalize unassigned customers heavily
        let unassigned_penalty = solution.unassigned().len() as f64 * 10_000.0;
        solution.total_distance() + unassigned_penalty
    }
}

// RoutingAlnsProblem contains only owned data
unsafe impl Send for RoutingAlnsProblem {}
unsafe impl Sync for RoutingAlnsProblem {}

#[cfg(test)]
mod tests {
    use super::*;
    use u_metaheur::alns::{AlnsConfig, AlnsRunner};

    use super::super::destroy::{RandomRemoval, ShawRemoval, WorstRemoval};
    use super::super::repair::{GreedyInsertion, RegretInsertion};

    fn setup() -> (Vec<Customer>, DistanceMatrix) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn test_initial_solution() {
        let (cust, dm) = setup();
        let problem = RoutingAlnsProblem::new(cust, dm, 30);
        let mut rng = u_numflow::random::create_rng(42);
        let sol = problem.initial_solution(&mut rng);
        let total: usize = sol.routes().iter().map(|r| r.len()).sum();
        assert_eq!(total, 3);
        assert!(sol.unassigned().is_empty());
    }

    #[test]
    fn test_cost_penalizes_unassigned() {
        let (cust, dm) = setup();
        let problem = RoutingAlnsProblem::new(cust.clone(), dm.clone(), 30);
        let sol_full = RoutingSolution::new(vec![vec![1, 2, 3]], vec![], &cust, &dm);
        let sol_partial = RoutingSolution::new(vec![vec![1, 2]], vec![3], &cust, &dm);
        assert!(problem.cost(&sol_partial) > problem.cost(&sol_full));
    }

    #[test]
    fn test_alns_runner_basic() {
        let (cust, dm) = setup();
        let capacity = 30;
        let problem = RoutingAlnsProblem::new(cust.clone(), dm.clone(), capacity);
        let destroy_ops = vec![RandomRemoval];
        let repair_ops = vec![GreedyInsertion::new(dm, cust, capacity)];
        let config = AlnsConfig::default().with_max_iterations(200).with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config).unwrap();
        assert!(result.best_cost < f64::INFINITY);
        assert!(result.best.unassigned().is_empty());
    }

    #[test]
    fn test_alns_runner_worst_removal() {
        let (cust, dm) = setup();
        let capacity = 30;
        let problem = RoutingAlnsProblem::new(cust.clone(), dm.clone(), capacity);
        let destroy_ops = vec![WorstRemoval::new(dm.clone())];
        let repair_ops = vec![GreedyInsertion::new(dm, cust, capacity)];
        let config = AlnsConfig::default().with_max_iterations(200).with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config).unwrap();
        assert!(result.best_cost < f64::INFINITY);
        assert!(result.best.unassigned().is_empty());
    }

    #[test]
    fn test_alns_runner_shaw_regret() {
        let (cust, dm) = setup();
        let capacity = 30;
        let problem = RoutingAlnsProblem::new(cust.clone(), dm.clone(), capacity);
        let destroy_ops = vec![ShawRemoval::new(dm.clone(), cust.clone())];
        let repair_ops = vec![RegretInsertion::new(dm, cust, capacity)];
        let config = AlnsConfig::default().with_max_iterations(200).with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config).unwrap();
        assert!(result.best_cost < f64::INFINITY);
        assert!(result.best.unassigned().is_empty());
    }
}
