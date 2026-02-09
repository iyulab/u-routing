//! Lightweight solution representation for ALNS operators.
//!
//! Routes are stored as `Vec<Vec<usize>>` (customer ID sequences) with a
//! separate list of unassigned customers. This allows efficient insertion
//! and removal without rebuilding full `Route` objects.

use crate::distance::DistanceMatrix;
use crate::models::Customer;

/// Lightweight VRP solution for ALNS manipulation.
///
/// # Examples
///
/// ```
/// use u_routing::alns::RoutingSolution;
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// let sol = RoutingSolution::new(vec![vec![1, 2]], vec![], &customers, &dm);
/// assert_eq!(sol.num_routes(), 1);
/// assert!(sol.unassigned().is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct RoutingSolution {
    routes: Vec<Vec<usize>>,
    unassigned: Vec<usize>,
    total_distance: f64,
}

impl RoutingSolution {
    /// Creates a new solution from route sequences.
    pub fn new(
        routes: Vec<Vec<usize>>,
        unassigned: Vec<usize>,
        _customers: &[Customer],
        distances: &DistanceMatrix,
    ) -> Self {
        let total_distance = compute_total_distance(&routes, distances);
        Self {
            routes,
            unassigned,
            total_distance,
        }
    }

    /// Returns the routes as customer ID sequences.
    pub fn routes(&self) -> &[Vec<usize>] {
        &self.routes
    }

    /// Returns mutable routes.
    pub fn routes_mut(&mut self) -> &mut Vec<Vec<usize>> {
        &mut self.routes
    }

    /// Returns unassigned customers.
    pub fn unassigned(&self) -> &[usize] {
        &self.unassigned
    }

    /// Returns mutable unassigned list.
    pub fn unassigned_mut(&mut self) -> &mut Vec<usize> {
        &mut self.unassigned
    }

    /// Total distance across all routes.
    pub fn total_distance(&self) -> f64 {
        self.total_distance
    }

    /// Number of routes.
    pub fn num_routes(&self) -> usize {
        self.routes.len()
    }

    /// Recalculates total distance from current routes.
    pub fn recalculate_distance(&mut self, distances: &DistanceMatrix) {
        self.total_distance = compute_total_distance(&self.routes, distances);
    }

    /// Removes empty routes.
    pub fn remove_empty_routes(&mut self) {
        self.routes.retain(|r| !r.is_empty());
    }
}

/// Computes total distance for all routes (depot=0).
fn compute_total_distance(routes: &[Vec<usize>], distances: &DistanceMatrix) -> f64 {
    let depot = 0;
    let mut total = 0.0;
    for route in routes {
        if route.is_empty() {
            continue;
        }
        total += distances.get(depot, route[0]);
        for w in route.windows(2) {
            total += distances.get(w[0], w[1]);
        }
        total += distances.get(route[route.len() - 1], depot);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Customer;

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
    fn test_solution_distance() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2, 3]], vec![], &cust, &dm);
        // 0→1→2→3→0 = 1+1+1+3 = 6
        assert!((sol.total_distance() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_solution_two_routes() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1], vec![2, 3]], vec![], &cust, &dm);
        // (0→1→0)+(0→2→3→0) = 2+6 = 8
        assert!((sol.total_distance() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_solution_with_unassigned() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2]], vec![3], &cust, &dm);
        assert_eq!(sol.unassigned(), &[3]);
        // Only counts assigned: 0→1→2→0 = 4
        assert!((sol.total_distance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_empty_routes() {
        let (cust, dm) = setup();
        let mut sol = RoutingSolution::new(vec![vec![1], vec![], vec![2]], vec![], &cust, &dm);
        sol.remove_empty_routes();
        assert_eq!(sol.num_routes(), 2);
    }
}
