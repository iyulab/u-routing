//! Vehicle type with capacity and cost parameters.

/// A vehicle that services routes in a routing problem.
///
/// Using it for a route costs `fixed_cost + cost_per_distance × distance`
/// ([`Vehicle::route_cost`]); every method minimizes the sum over routes.
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
    rest: Option<VehicleBreak>,
}

/// A rest a driver has to take during a route: `duration` long, starting at some time in
/// `[earliest, latest]`, at the stop the vehicle has just reached (before the stop's service, so
/// the rest can overlap the wait for a time window to open).
///
/// A route has to contain the break unless the vehicle is back at the depot by `earliest`
/// (a shift that ends before the break is due). The break is placed where the route finishes
/// earliest; it is not a stop, and it is not a cost -- it makes the route longer, which a
/// `max_duration` and the time windows after it feel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleBreak {
    earliest: f64,
    latest: f64,
    duration: f64,
}

impl VehicleBreak {
    /// A break of `duration` that starts no earlier than `earliest` and no later than `latest`.
    pub fn new(earliest: f64, latest: f64, duration: f64) -> Self {
        Self {
            earliest,
            latest,
            duration,
        }
    }

    /// The first time the break may start.
    pub fn earliest(&self) -> f64 {
        self.earliest
    }

    /// The last time the break may start.
    pub fn latest(&self) -> f64 {
        self.latest
    }

    /// How long the break takes.
    pub fn duration(&self) -> f64 {
        self.duration
    }
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
            rest: None,
        }
    }

    /// Sets the depot for this vehicle: the table index of the depot its routes
    /// start from and end at, one of the first [`num_depots`](super::num_depots)
    /// entries of the problem's locations (default 0).
    pub fn with_depot(mut self, depot_id: usize) -> Self {
        self.depot_id = depot_id;
        self
    }

    /// This vehicle as entry `id` of a vehicle list, starting from `depot_id`.
    pub(crate) fn placed(&self, id: usize, depot_id: usize) -> Vehicle {
        Vehicle {
            id,
            depot_id,
            ..self.clone()
        }
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

    /// Gives the driver a break ([`VehicleBreak`]) during every route.
    pub fn with_break(mut self, rest: VehicleBreak) -> Self {
        self.rest = Some(rest);
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

    /// What a route of `distance` costs with this vehicle:
    /// `fixed_cost + cost_per_distance × distance`.
    pub fn route_cost(&self, distance: f64) -> f64 {
        self.fixed_cost + self.cost_per_distance * distance
    }

    /// The fixed cost in units of distance, `fixed_cost / cost_per_distance`:
    /// for a fleet of one vehicle type, minimizing total cost is minimizing
    /// total distance plus this much per route.
    pub fn fixed_cost_in_distance(&self) -> f64 {
        if self.cost_per_distance > 0.0 {
            self.fixed_cost / self.cost_per_distance
        } else {
            0.0
        }
    }

    /// Maximum distance limit, if any.
    pub fn max_distance(&self) -> Option<f64> {
        self.max_distance
    }

    /// Maximum duration limit, if any.
    pub fn max_duration(&self) -> Option<f64> {
        self.max_duration
    }

    /// The driver's break, if any.
    pub fn rest(&self) -> Option<VehicleBreak> {
        self.rest
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
        assert_eq!(v.route_cost(10.0), v.fixed_cost() + 15.0);
        assert_eq!(v.fixed_cost(), 50.0);
        assert_eq!(v.max_distance(), Some(500.0));
        assert_eq!(v.max_duration(), Some(480.0));
    }
}
