//! Vehicle type with capacity and cost parameters.

/// A vehicle that services routes in a routing problem.
///
/// # Examples
///
/// ```
/// use u_routing::models::Vehicle;
///
/// let v = Vehicle::new(0, 200);
/// assert_eq!(v.id(), 0);
/// assert_eq!(v.capacity(), 200);
/// ```
#[derive(Debug, Clone)]
pub struct Vehicle {
    id: usize,
    capacity: i32,
    depot_id: usize,
    cost_per_distance: f64,
    fixed_cost: f64,
    max_distance: Option<f64>,
    max_duration: Option<f64>,
}

impl Vehicle {
    /// Creates a vehicle with the given ID and capacity.
    ///
    /// Default: depot 0, cost_per_distance = 1.0, no fixed cost,
    /// no distance/duration limits.
    pub fn new(id: usize, capacity: i32) -> Self {
        Self {
            id,
            capacity,
            depot_id: 0,
            cost_per_distance: 1.0,
            fixed_cost: 0.0,
            max_distance: None,
            max_duration: None,
        }
    }

    /// Sets the depot for this vehicle.
    pub fn with_depot(mut self, depot_id: usize) -> Self {
        self.depot_id = depot_id;
        self
    }

    /// Sets cost per unit distance.
    pub fn with_cost_per_distance(mut self, cost: f64) -> Self {
        self.cost_per_distance = cost;
        self
    }

    /// Sets fixed cost for using this vehicle.
    pub fn with_fixed_cost(mut self, cost: f64) -> Self {
        self.fixed_cost = cost;
        self
    }

    /// Sets maximum route distance.
    pub fn with_max_distance(mut self, max: f64) -> Self {
        self.max_distance = Some(max);
        self
    }

    /// Sets maximum route duration.
    pub fn with_max_duration(mut self, max: f64) -> Self {
        self.max_duration = Some(max);
        self
    }

    /// Vehicle ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Maximum load capacity.
    pub fn capacity(&self) -> i32 {
        self.capacity
    }

    /// Depot location ID (start and end of route).
    pub fn depot_id(&self) -> usize {
        self.depot_id
    }

    /// Cost per unit distance traveled.
    pub fn cost_per_distance(&self) -> f64 {
        self.cost_per_distance
    }

    /// Fixed cost for using this vehicle (independent of distance).
    pub fn fixed_cost(&self) -> f64 {
        self.fixed_cost
    }

    /// Maximum distance limit, if any.
    pub fn max_distance(&self) -> Option<f64> {
        self.max_distance
    }

    /// Maximum duration limit, if any.
    pub fn max_duration(&self) -> Option<f64> {
        self.max_duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_new() {
        let v = Vehicle::new(0, 200);
        assert_eq!(v.id(), 0);
        assert_eq!(v.capacity(), 200);
        assert_eq!(v.depot_id(), 0);
        assert_eq!(v.cost_per_distance(), 1.0);
        assert_eq!(v.fixed_cost(), 0.0);
        assert!(v.max_distance().is_none());
        assert!(v.max_duration().is_none());
    }

    #[test]
    fn test_vehicle_builder() {
        let v = Vehicle::new(1, 100)
            .with_depot(2)
            .with_cost_per_distance(1.5)
            .with_fixed_cost(50.0)
            .with_max_distance(500.0)
            .with_max_duration(480.0);
        assert_eq!(v.id(), 1);
        assert_eq!(v.capacity(), 100);
        assert_eq!(v.depot_id(), 2);
        assert_eq!(v.cost_per_distance(), 1.5);
        assert_eq!(v.fixed_cost(), 50.0);
        assert_eq!(v.max_distance(), Some(500.0));
        assert_eq!(v.max_duration(), Some(480.0));
    }
}
