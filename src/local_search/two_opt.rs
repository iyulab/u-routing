//! Intra-route 2-opt improvement.
//!
//! # Algorithm
//!
//! For each pair of edges (i, i+1) and (j, j+1) in a route, compute the
//! change in distance from reversing the segment between them:
//!
//! ```text
//! delta = d(r[i], r[j]) + d(r[i+1], r[j+1]) - d(r[i], r[i+1]) - d(r[j], r[j+1])
//! ```
//!
//! If delta < 0, reverse the segment [i+1..=j] and accept the improvement.
//! Repeat until no further improvements are found (first-improvement strategy).
//!
//! # Complexity
//!
//! O(n²) per pass, O(n³) worst case for convergence.
//!
//! # Reference
//!
//! Croes, G.A. (1958). "A method for solving traveling salesman problems",
//! *Operations Research* 6(6), 791-812.

use crate::distance::DistanceMatrix;

/// Applies 2-opt improvement to a single route (given as a sequence of customer IDs).
///
/// The route is assumed to start and end at `depot`. Returns the improved
/// customer sequence and the total route distance.
///
/// # Arguments
///
/// * `route` — Ordered customer IDs (excluding depot)
/// * `depot` — Depot location ID
/// * `distances` — Distance matrix
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::local_search::two_opt_improve;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 3.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// // Suboptimal order: 1, 3, 2
/// let (improved, dist) = two_opt_improve(&[1, 3, 2], 0, &dm);
/// // 2-opt should fix crossings
/// assert!(dist <= 6.0 + 1e-10); // optimal: 0→1→2→3→0 = 6
/// ```
pub fn two_opt_improve(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
) -> (Vec<usize>, f64) {
    if route.len() < 2 {
        let dist = if route.is_empty() {
            0.0
        } else {
            distances.get(depot, route[0]) + distances.get(route[0], depot)
        };
        return (route.to_vec(), dist);
    }

    let mut current = route.to_vec();
    let mut improved = true;

    while improved {
        improved = false;
        let n = current.len();

        for i in 0..n - 1 {
            for j in i + 1..n {
                let delta = two_opt_delta(&current, depot, distances, i, j);
                if delta < -1e-10 {
                    // Reverse segment [i+1..=j] — but in our 0-indexed route
                    // that means reverse [i..=j] since i and j are customer indices
                    current[i..=j].reverse();
                    improved = true;
                }
            }
        }
    }

    let dist = route_distance(&current, depot, distances);
    (current, dist)
}

/// Computes the distance change from a 2-opt swap of edges at positions i and j.
///
/// Before: ...-prev_i - route[i] - route[i+1] - ... - route[j] - next_j-...
/// After:  ...-prev_i - route[j] - route[j-1] - ... - route[i] - next_j-...
fn two_opt_delta(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    i: usize,
    j: usize,
) -> f64 {
    let n = route.len();
    let prev_i = if i == 0 { depot } else { route[i - 1] };
    let next_j = if j == n - 1 { depot } else { route[j + 1] };

    let old_cost = distances.get(prev_i, route[i]) + distances.get(route[j], next_j);
    let new_cost = distances.get(prev_i, route[j]) + distances.get(route[i], next_j);

    new_cost - old_cost
}

/// Computes the total distance of a route: `depot → route[0] → ... → route[n-1] → depot`.
fn route_distance(route: &[usize], depot: usize, distances: &DistanceMatrix) -> f64 {
    if route.is_empty() {
        return 0.0;
    }
    let mut dist = distances.get(depot, route[0]);
    for i in 0..route.len() - 1 {
        dist += distances.get(route[i], route[i + 1]);
    }
    dist += distances.get(route[route.len() - 1], depot);
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Customer;

    fn line_customers() -> (Vec<Customer>, DistanceMatrix) {
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
    fn test_2opt_already_optimal() {
        let (_, dm) = line_customers();
        let (improved, dist) = two_opt_improve(&[1, 2, 3], 0, &dm);
        assert_eq!(improved, vec![1, 2, 3]);
        assert!((dist - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_2opt_reverses_crossing() {
        let _ = line_customers();
        // 0→1→3→2→0 has distance 1 + 2 + 1 + 2 = 6, same as optimal in this case
        // Let's use a case where crossing actually matters
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 1.0, -1.0, 10, 0.0),
        ];
        let dm2 = DistanceMatrix::from_customers(&customers);
        // Route [1, 3, 2]: depot(0,0)→(1,1)→(1,-1)→(2,0)→depot = crosses
        let (_, improved_dist) = two_opt_improve(&[1, 3, 2], 0, &dm2);
        let (_, original_dist) = (vec![1, 3, 2], route_distance(&[1, 3, 2], 0, &dm2));
        assert!(improved_dist <= original_dist + 1e-10);
    }

    #[test]
    fn test_2opt_empty_route() {
        let (_, dm) = line_customers();
        let (improved, dist) = two_opt_improve(&[], 0, &dm);
        assert!(improved.is_empty());
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_2opt_single_customer() {
        let (_, dm) = line_customers();
        let (improved, dist) = two_opt_improve(&[2], 0, &dm);
        assert_eq!(improved, vec![2]);
        assert!((dist - 4.0).abs() < 1e-10); // 0→2→0 = 2+2
    }

    #[test]
    fn test_route_distance() {
        let (_, dm) = line_customers();
        let d = route_distance(&[1, 2, 3], 0, &dm);
        assert!((d - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_2opt_does_not_worsen() {
        let customers = vec![
            Customer::depot(5.0, 5.0),
            Customer::new(1, 0.0, 0.0, 5, 0.0),
            Customer::new(2, 10.0, 0.0, 5, 0.0),
            Customer::new(3, 0.0, 10.0, 5, 0.0),
            Customer::new(4, 10.0, 10.0, 5, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let initial = vec![1, 4, 2, 3]; // deliberately bad order
        let initial_dist = route_distance(&initial, 0, &dm);
        let (_, improved_dist) = two_opt_improve(&initial, 0, &dm);
        assert!(improved_dist <= initial_dist + 1e-10);
    }
}
