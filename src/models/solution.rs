//! Solution and violation types.

use super::Route;

/// A type of constraint violation in a route or solution.
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    /// Vehicle capacity exceeded.
    CapacityExceeded {
        /// Route index in the solution.
        route_index: usize,
        /// Load that exceeded capacity.
        load: i32,
        /// Vehicle capacity.
        capacity: i32,
    },
    /// Arrival after the customer's time window closes.
    TimeWindowViolated {
        /// Customer ID where violation occurred.
        customer_id: usize,
        /// Actual arrival time.
        arrival: f64,
        /// Time window due date.
        due: f64,
    },
    /// Route distance exceeds vehicle's maximum.
    MaxDistanceExceeded {
        /// Route index.
        route_index: usize,
        /// Actual distance.
        distance: f64,
        /// Maximum allowed distance.
        max_distance: f64,
    },
    /// Route duration exceeds vehicle's maximum.
    MaxDurationExceeded {
        /// Route index.
        route_index: usize,
        /// Actual duration.
        duration: f64,
        /// Maximum allowed duration.
        max_duration: f64,
    },
}

/// A constraint violation in a solution.
#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    /// The type of violation.
    pub kind: ViolationType,
}

impl Violation {
    /// Creates a new violation.
    pub fn new(kind: ViolationType) -> Self {
        Self { kind }
    }
}

/// A complete solution to a routing problem.
///
/// Contains a set of routes and optionally unassigned customers.
///
/// # Examples
///
/// ```
/// use u_routing::models::{Solution, Route};
///
/// let mut sol = Solution::new();
/// sol.add_route(Route::new(0));
/// assert_eq!(sol.num_routes(), 1);
/// assert_eq!(sol.num_unassigned(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct Solution {
    routes: Vec<Route>,
    unassigned: Vec<usize>,
    total_cost: f64,
}

impl Solution {
    /// Creates an empty solution.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            unassigned: Vec::new(),
            total_cost: 0.0,
        }
    }

    /// Adds a route to this solution.
    pub fn add_route(&mut self, route: Route) {
        self.routes.push(route);
    }

    /// Marks a customer as unassigned.
    pub fn add_unassigned(&mut self, customer_id: usize) {
        self.unassigned.push(customer_id);
    }

    /// Returns the routes in this solution.
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Returns a mutable reference to the routes.
    pub fn routes_mut(&mut self) -> &mut Vec<Route> {
        &mut self.routes
    }

    /// Returns the number of routes (vehicles used).
    pub fn num_routes(&self) -> usize {
        self.routes.len()
    }

    /// Returns the IDs of unassigned customers.
    pub fn unassigned(&self) -> &[usize] {
        &self.unassigned
    }

    /// Returns the number of unassigned customers.
    pub fn num_unassigned(&self) -> usize {
        self.unassigned.len()
    }

    /// Returns the total cost of this solution.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Sets the total cost.
    pub fn set_total_cost(&mut self, cost: f64) {
        self.total_cost = cost;
    }

    /// Total distance across all routes.
    pub fn total_distance(&self) -> f64 {
        self.routes.iter().map(|r| r.total_distance()).sum()
    }

    /// Total number of customers served (across all routes).
    pub fn num_served(&self) -> usize {
        self.routes.iter().map(|r| r.len()).sum()
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Visit;

    #[test]
    fn test_solution_empty() {
        let sol = Solution::new();
        assert_eq!(sol.num_routes(), 0);
        assert_eq!(sol.num_unassigned(), 0);
        assert_eq!(sol.total_cost(), 0.0);
        assert_eq!(sol.num_served(), 0);
    }

    #[test]
    fn test_solution_with_routes() {
        let mut sol = Solution::new();

        let mut r1 = Route::new(0);
        r1.push_visit(Visit {
            customer_id: 1,
            arrival_time: 0.0,
            departure_time: 0.0,
            load_after: 10,
        });
        r1.set_total_distance(50.0);

        let mut r2 = Route::new(1);
        r2.push_visit(Visit {
            customer_id: 2,
            arrival_time: 0.0,
            departure_time: 0.0,
            load_after: 5,
        });
        r2.push_visit(Visit {
            customer_id: 3,
            arrival_time: 0.0,
            departure_time: 0.0,
            load_after: 15,
        });
        r2.set_total_distance(80.0);

        sol.add_route(r1);
        sol.add_route(r2);
        sol.add_unassigned(4);

        assert_eq!(sol.num_routes(), 2);
        assert_eq!(sol.num_served(), 3);
        assert_eq!(sol.num_unassigned(), 1);
        assert!((sol.total_distance() - 130.0).abs() < 1e-10);
    }

    #[test]
    fn test_violation_types() {
        let v = Violation::new(ViolationType::CapacityExceeded {
            route_index: 0,
            load: 250,
            capacity: 200,
        });
        assert_eq!(
            v.kind,
            ViolationType::CapacityExceeded {
                route_index: 0,
                load: 250,
                capacity: 200,
            }
        );
    }

    #[test]
    fn test_solution_default() {
        let sol = Solution::default();
        assert_eq!(sol.num_routes(), 0);
    }
}
