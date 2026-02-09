//! Route and visit types.

/// A single visit to a customer within a route.
///
/// Tracks the customer ID along with computed timing and load state.
#[derive(Debug, Clone, PartialEq)]
pub struct Visit {
    /// Customer ID being visited.
    pub customer_id: usize,
    /// Arrival time at this customer.
    pub arrival_time: f64,
    /// Departure time (arrival + waiting + service duration).
    pub departure_time: f64,
    /// Cumulative load after this visit.
    pub load_after: i32,
}

/// An ordered sequence of customer visits assigned to a single vehicle.
///
/// A route starts and ends at the vehicle's depot (not stored in `visits`).
///
/// # Examples
///
/// ```
/// use u_routing::models::{Route, Visit};
///
/// let mut route = Route::new(0);
/// route.push_visit(Visit {
///     customer_id: 1,
///     arrival_time: 10.0,
///     departure_time: 20.0,
///     load_after: 10,
/// });
/// assert_eq!(route.len(), 1);
/// assert_eq!(route.vehicle_id(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct Route {
    vehicle_id: usize,
    visits: Vec<Visit>,
    total_distance: f64,
    total_duration: f64,
    total_load: i32,
}

impl Route {
    /// Creates an empty route for the given vehicle.
    pub fn new(vehicle_id: usize) -> Self {
        Self {
            vehicle_id,
            visits: Vec::new(),
            total_distance: 0.0,
            total_duration: 0.0,
            total_load: 0,
        }
    }

    /// Appends a visit to the end of this route.
    pub fn push_visit(&mut self, visit: Visit) {
        self.total_load = visit.load_after;
        self.visits.push(visit);
    }

    /// Returns the vehicle assigned to this route.
    pub fn vehicle_id(&self) -> usize {
        self.vehicle_id
    }

    /// Returns the ordered sequence of visits.
    pub fn visits(&self) -> &[Visit] {
        &self.visits
    }

    /// Returns the number of customer visits (excluding depot).
    pub fn len(&self) -> usize {
        self.visits.len()
    }

    /// Returns `true` if this route has no customer visits.
    pub fn is_empty(&self) -> bool {
        self.visits.is_empty()
    }

    /// Returns the customer IDs in visit order.
    pub fn customer_ids(&self) -> Vec<usize> {
        self.visits.iter().map(|v| v.customer_id).collect()
    }

    /// Total distance of this route (set by evaluator).
    pub fn total_distance(&self) -> f64 {
        self.total_distance
    }

    /// Total duration of this route (set by evaluator).
    pub fn total_duration(&self) -> f64 {
        self.total_duration
    }

    /// Total load served by this route.
    pub fn total_load(&self) -> i32 {
        self.total_load
    }

    /// Sets the total distance (used by evaluator).
    pub fn set_total_distance(&mut self, d: f64) {
        self.total_distance = d;
    }

    /// Sets the total duration (used by evaluator).
    pub fn set_total_duration(&mut self, d: f64) {
        self.total_duration = d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_empty() {
        let r = Route::new(0);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.vehicle_id(), 0);
        assert_eq!(r.total_distance(), 0.0);
        assert_eq!(r.total_load(), 0);
    }

    #[test]
    fn test_route_push_visit() {
        let mut r = Route::new(1);
        r.push_visit(Visit {
            customer_id: 5,
            arrival_time: 10.0,
            departure_time: 15.0,
            load_after: 20,
        });
        r.push_visit(Visit {
            customer_id: 3,
            arrival_time: 20.0,
            departure_time: 25.0,
            load_after: 35,
        });
        assert_eq!(r.len(), 2);
        assert_eq!(r.customer_ids(), vec![5, 3]);
        assert_eq!(r.total_load(), 35);
    }

    #[test]
    fn test_visit_equality() {
        let a = Visit {
            customer_id: 1,
            arrival_time: 10.0,
            departure_time: 20.0,
            load_after: 5,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
