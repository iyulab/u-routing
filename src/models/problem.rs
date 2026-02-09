//! Routing problem trait.

use super::{Customer, Solution, Vehicle, Violation};

/// Defines a vehicle routing problem instance.
///
/// This trait provides the interface that solvers use to access problem
/// data and evaluate solutions. Implementations supply customers, vehicles,
/// distance calculations, and feasibility checking.
///
/// # Examples
///
/// ```
/// use u_routing::models::{RoutingProblem, Customer, Vehicle, Solution, Violation};
/// use u_routing::distance::DistanceMatrix;
///
/// struct MyProblem {
///     customers: Vec<Customer>,
///     vehicles: Vec<Vehicle>,
///     distances: DistanceMatrix,
/// }
///
/// impl RoutingProblem for MyProblem {
///     fn customers(&self) -> &[Customer] { &self.customers }
///     fn vehicles(&self) -> &[Vehicle] { &self.vehicles }
///     fn num_customers(&self) -> usize { self.customers.len() - 1 }
///     fn distance(&self, from: usize, to: usize) -> f64 {
///         self.distances.get(from, to)
///     }
///     fn travel_time(&self, from: usize, to: usize) -> f64 {
///         self.distances.get(from, to) // distance == travel time
///     }
///     fn evaluate(&self, _solution: &Solution) -> (f64, Vec<Violation>) {
///         (0.0, vec![])
///     }
/// }
/// ```
pub trait RoutingProblem: Send + Sync {
    /// Returns all locations (index 0 = depot, 1..=N = customers).
    fn customers(&self) -> &[Customer];

    /// Returns the available vehicles.
    fn vehicles(&self) -> &[Vehicle];

    /// Number of customers (excluding depot).
    fn num_customers(&self) -> usize;

    /// Travel distance from location `from` to location `to`.
    fn distance(&self, from: usize, to: usize) -> f64;

    /// Travel time from location `from` to location `to`.
    ///
    /// Defaults to `distance(from, to)` (speed = 1).
    fn travel_time(&self, from: usize, to: usize) -> f64 {
        self.distance(from, to)
    }

    /// Evaluates a solution, returning `(cost, violations)`.
    ///
    /// A feasible solution has an empty violations list.
    fn evaluate(&self, solution: &Solution) -> (f64, Vec<Violation>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMatrix;

    struct SimpleProblem {
        customers: Vec<Customer>,
        vehicles: Vec<Vehicle>,
        distances: DistanceMatrix,
    }

    impl RoutingProblem for SimpleProblem {
        fn customers(&self) -> &[Customer] {
            &self.customers
        }
        fn vehicles(&self) -> &[Vehicle] {
            &self.vehicles
        }
        fn num_customers(&self) -> usize {
            self.customers.len() - 1
        }
        fn distance(&self, from: usize, to: usize) -> f64 {
            self.distances.get(from, to)
        }
        fn travel_time(&self, from: usize, to: usize) -> f64 {
            self.distances.get(from, to)
        }
        fn evaluate(&self, solution: &Solution) -> (f64, Vec<Violation>) {
            let cost = solution.total_distance();
            (cost, vec![])
        }
    }

    #[test]
    fn test_simple_problem() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 4.0, 10, 5.0),
            Customer::new(2, 6.0, 8.0, 20, 5.0),
        ];
        let distances = DistanceMatrix::from_customers(&customers);
        let vehicles = vec![Vehicle::new(0, 100)];

        let problem = SimpleProblem {
            customers,
            vehicles,
            distances,
        };

        assert_eq!(problem.num_customers(), 2);
        assert!((problem.distance(0, 1) - 5.0).abs() < 1e-10);

        let sol = Solution::new();
        let (cost, violations) = problem.evaluate(&sol);
        assert_eq!(cost, 0.0);
        assert!(violations.is_empty());
    }
}
