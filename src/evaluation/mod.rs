//! Route feasibility checking and cost evaluation.

mod evaluator;

pub use evaluator::RouteEvaluator;

use crate::distance::DistanceMatrix;
use crate::models::Customer;

/// Whether any customer carries a time window.
pub fn has_time_windows(customers: &[Customer]) -> bool {
    customers.iter().any(|c| c.time_window().is_some())
}

/// Whether a route `depot → route[0] → … → depot` reaches every customer by
/// its due time, with the same clock as [`RouteEvaluator::build_route`]:
/// departure at 0, travel time equal to distance, waiting for a window that
/// has not opened, and service before moving on. A route with no windowed
/// customer is always respected. O(n) and allocation-free, so a search can
/// ask it about every candidate move.
pub fn time_windows_respected(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> bool {
    let mut time = 0.0;
    let mut prev = depot;
    for &cid in route {
        let arrival = time + distances.get(prev, cid);
        let customer = &customers[cid];
        let start = match customer.time_window() {
            Some(tw) => {
                if tw.is_violated(arrival) {
                    return false;
                }
                arrival + tw.waiting_time(arrival)
            }
            None => arrival,
        };
        time = start + customer.service_duration();
        prev = cid;
    }
    true
}
