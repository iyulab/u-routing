//! Customer and time window types.

/// A time window constraint for service at a customer location.
///
/// The vehicle must arrive no later than `due` and may arrive as early as
/// `ready` (waiting is allowed if early).
///
/// # Examples
///
/// ```
/// use u_routing::models::TimeWindow;
///
/// let tw = TimeWindow::new(100.0, 200.0).unwrap();
/// assert!(tw.ready() <= tw.due());
/// assert!(tw.contains(150.0));
/// assert!(!tw.contains(250.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeWindow {
    ready: f64,
    due: f64,
}

impl TimeWindow {
    /// Creates a new time window.
    ///
    /// Returns `None` if `ready > due` or either value is non-finite.
    pub fn new(ready: f64, due: f64) -> Option<Self> {
        if !ready.is_finite() || !due.is_finite() || ready > due {
            return None;
        }
        Some(Self { ready, due })
    }

    /// Earliest allowable arrival time.
    pub fn ready(&self) -> f64 {
        self.ready
    }

    /// Latest allowable arrival time.
    pub fn due(&self) -> f64 {
        self.due
    }

    /// Returns `true` if the given time falls within this window.
    pub fn contains(&self, time: f64) -> bool {
        time >= self.ready && time <= self.due
    }

    /// Returns the waiting time if arriving at the given time.
    ///
    /// Zero if the vehicle arrives within or after the window.
    pub fn waiting_time(&self, arrival: f64) -> f64 {
        if arrival < self.ready {
            self.ready - arrival
        } else {
            0.0
        }
    }

    /// Returns `true` if arriving at the given time violates this window.
    pub fn is_violated(&self, arrival: f64) -> bool {
        arrival > self.due
    }
}

/// A customer (or depot) in a routing problem.
///
/// Customer 0 is conventionally the depot. Customers have a location
/// (coordinates), a demand, an optional time window, and a service duration.
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
///
/// let depot = Customer::depot(35.0, 35.0);
/// assert_eq!(depot.id(), 0);
/// assert_eq!(depot.demand(), 0);
///
/// let c = Customer::new(1, 41.0, 49.0, 10, 10.0);
/// assert_eq!(c.id(), 1);
/// assert_eq!(c.demand(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct Customer {
    id: usize,
    x: f64,
    y: f64,
    demand: i32,
    service_duration: f64,
    time_window: Option<TimeWindow>,
}

impl Customer {
    /// Creates a new customer.
    pub fn new(id: usize, x: f64, y: f64, demand: i32, service_duration: f64) -> Self {
        Self {
            id,
            x,
            y,
            demand,
            service_duration,
            time_window: None,
        }
    }

    /// Creates a depot at the given coordinates (id=0, demand=0).
    pub fn depot(x: f64, y: f64) -> Self {
        Self::new(0, x, y, 0, 0.0)
    }

    /// Sets a time window for this customer.
    pub fn with_time_window(mut self, tw: TimeWindow) -> Self {
        self.time_window = Some(tw);
        self
    }

    /// Customer ID (0 = depot).
    pub fn id(&self) -> usize {
        self.id
    }

    /// X-coordinate.
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Y-coordinate.
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Demand at this customer (units to deliver or pick up).
    pub fn demand(&self) -> i32 {
        self.demand
    }

    /// Service duration at this customer.
    pub fn service_duration(&self) -> f64 {
        self.service_duration
    }

    /// Time window constraint, if any.
    pub fn time_window(&self) -> Option<&TimeWindow> {
        self.time_window.as_ref()
    }

    /// Euclidean distance to another customer.
    pub fn distance_to(&self, other: &Customer) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_window_valid() {
        let tw = TimeWindow::new(10.0, 20.0).expect("valid");
        assert_eq!(tw.ready(), 10.0);
        assert_eq!(tw.due(), 20.0);
    }

    #[test]
    fn test_time_window_invalid() {
        assert!(TimeWindow::new(20.0, 10.0).is_none());
        assert!(TimeWindow::new(f64::NAN, 10.0).is_none());
        assert!(TimeWindow::new(10.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_time_window_contains() {
        let tw = TimeWindow::new(10.0, 20.0).expect("valid");
        assert!(tw.contains(10.0));
        assert!(tw.contains(15.0));
        assert!(tw.contains(20.0));
        assert!(!tw.contains(9.9));
        assert!(!tw.contains(20.1));
    }

    #[test]
    fn test_time_window_waiting() {
        let tw = TimeWindow::new(10.0, 20.0).expect("valid");
        assert!((tw.waiting_time(5.0) - 5.0).abs() < 1e-10);
        assert!((tw.waiting_time(10.0)).abs() < 1e-10);
        assert!((tw.waiting_time(15.0)).abs() < 1e-10);
    }

    #[test]
    fn test_time_window_violated() {
        let tw = TimeWindow::new(10.0, 20.0).expect("valid");
        assert!(!tw.is_violated(10.0));
        assert!(!tw.is_violated(20.0));
        assert!(tw.is_violated(20.1));
    }

    #[test]
    fn test_customer_new() {
        let c = Customer::new(1, 10.0, 20.0, 5, 3.0);
        assert_eq!(c.id(), 1);
        assert_eq!(c.x(), 10.0);
        assert_eq!(c.y(), 20.0);
        assert_eq!(c.demand(), 5);
        assert_eq!(c.service_duration(), 3.0);
        assert!(c.time_window().is_none());
    }

    #[test]
    fn test_customer_depot() {
        let d = Customer::depot(35.0, 35.0);
        assert_eq!(d.id(), 0);
        assert_eq!(d.demand(), 0);
        assert_eq!(d.service_duration(), 0.0);
    }

    #[test]
    fn test_customer_with_time_window() {
        let tw = TimeWindow::new(100.0, 200.0).expect("valid");
        let c = Customer::new(1, 10.0, 20.0, 5, 3.0).with_time_window(tw);
        assert!(c.time_window().is_some());
        assert_eq!(c.time_window().expect("has tw").ready(), 100.0);
    }

    #[test]
    fn test_customer_distance() {
        let a = Customer::new(0, 0.0, 0.0, 0, 0.0);
        let b = Customer::new(1, 3.0, 4.0, 0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_customer_distance_symmetric() {
        let a = Customer::new(0, 1.0, 2.0, 0, 0.0);
        let b = Customer::new(1, 4.0, 6.0, 0, 0.0);
        assert!((a.distance_to(&b) - b.distance_to(&a)).abs() < 1e-10);
    }
}
