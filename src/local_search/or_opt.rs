//! Intra-route Or-opt improvement.
//!
//! # Algorithm
//!
//! Tries moving segments of 1, 2, or 3 consecutive customers to a different
//! position within the same route. Accepts moves that reduce total distance.
//!
//! For each segment size k ∈ {1, 2, 3} and each starting position, computes
//! the cost change from removing the segment and reinserting it at every
//! other position.
//!
//! # Complexity
//!
//! O(n²) per pass, O(n³) worst case for convergence.
//!
//! # Reference
//!
//! Or, I. (1976). "Traveling Salesman-Type Combinatorial Problems and Their
//! Relation to the Logistics of Blood Banking". PhD thesis.

use crate::distance::DistanceMatrix;

/// Applies Or-opt improvement to a single route.
///
/// Tries relocating segments of 1, 2, and 3 customers to better positions.
/// Returns the improved customer sequence and total distance.
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
/// use u_routing::local_search::{or_opt_improve, route_distance};
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 1.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 1.0, -1.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// // Try a suboptimal order
/// let (improved, dist) = or_opt_improve(&[1, 3, 2], 0, &dm);
/// let orig_dist = route_distance(&[1, 3, 2], 0, &dm);
/// assert!(dist <= orig_dist + 1e-10);
/// ```
pub fn or_opt_improve(
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

        // Try segment sizes 1, 2, 3
        for seg_len in 1..=3.min(current.len()) {
            if try_or_opt_pass(&mut current, depot, distances, seg_len) {
                improved = true;
            }
        }
    }

    let dist = route_distance(&current, depot, distances);
    (current, dist)
}

/// Computes the total distance: depot → route[0] → ... → route[n-1] → depot.
pub fn route_distance(route: &[usize], depot: usize, distances: &DistanceMatrix) -> f64 {
    if route.is_empty() {
        return 0.0;
    }
    let mut dist = distances.get(depot, route[0]);
    for w in route.windows(2) {
        dist += distances.get(w[0], w[1]);
    }
    dist += distances.get(route[route.len() - 1], depot);
    dist
}

/// One pass of Or-opt for a given segment length. Returns true if improved.
fn try_or_opt_pass(
    route: &mut Vec<usize>,
    depot: usize,
    distances: &DistanceMatrix,
    seg_len: usize,
) -> bool {
    let n = route.len();
    if n < seg_len + 1 {
        return false;
    }

    let mut best_delta = -1e-10;
    let mut best_from = 0;
    let mut best_to = 0;

    for from in 0..=(n - seg_len) {
        // Cost of removing segment [from..from+seg_len]
        let prev = if from == 0 { depot } else { route[from - 1] };
        let after = if from + seg_len >= n {
            depot
        } else {
            route[from + seg_len]
        };
        let seg_first = route[from];
        let seg_last = route[from + seg_len - 1];

        // Old edges: prev→seg_first + seg_last→after
        // New edges (after removal): prev→after
        let removal_gain = distances.get(prev, seg_first) + distances.get(seg_last, after)
            - distances.get(prev, after);

        // Try inserting the segment at each other position
        for to in 0..=n - seg_len {
            // Adjust 'to' for positions that don't overlap with [from..from+seg_len]
            if to >= from && to <= from + seg_len {
                continue;
            }

            // Position in the route *after* removing the segment
            // We need the insertion edges
            let (ins_prev, ins_next) = if to < from {
                let p = if to == 0 { depot } else { route[to - 1] };
                let nx = route[to];
                (p, nx)
            } else {
                // to > from + seg_len
                let actual_to = to; // index in original route
                let p = route[actual_to - 1];
                let nx = if actual_to >= n { depot } else { route[actual_to] };
                (p, nx)
            };

            // Insertion cost: ins_prev→seg_first + seg_last→ins_next - ins_prev→ins_next
            let insertion_cost = distances.get(ins_prev, seg_first)
                + distances.get(seg_last, ins_next)
                - distances.get(ins_prev, ins_next);

            let delta = insertion_cost - removal_gain;

            if delta < best_delta {
                best_delta = delta;
                best_from = from;
                best_to = to;
            }
        }
    }

    if best_delta < -1e-10 {
        // Execute the move: remove segment, then insert at new position
        let segment: Vec<usize> = route.drain(best_from..best_from + seg_len).collect();
        let insert_pos = if best_to > best_from {
            best_to - seg_len
        } else {
            best_to
        };
        for (i, &cid) in segment.iter().enumerate() {
            route.insert(insert_pos + i, cid);
        }
        true
    } else {
        false
    }
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
    fn test_or_opt_already_optimal() {
        let (_, dm) = line_customers();
        let (improved, dist) = or_opt_improve(&[1, 2, 3], 0, &dm);
        assert_eq!(improved, vec![1, 2, 3]);
        assert!((dist - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_or_opt_empty() {
        let (_, dm) = line_customers();
        let (improved, dist) = or_opt_improve(&[], 0, &dm);
        assert!(improved.is_empty());
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_or_opt_single() {
        let (_, dm) = line_customers();
        let (improved, dist) = or_opt_improve(&[2], 0, &dm);
        assert_eq!(improved, vec![2]);
        assert!((dist - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_or_opt_does_not_worsen() {
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
        let (_, improved_dist) = or_opt_improve(&initial, 0, &dm);
        assert!(improved_dist <= initial_dist + 1e-10);
    }

    #[test]
    fn test_or_opt_two_customers() {
        let (_, dm) = line_customers();
        let (improved, dist) = or_opt_improve(&[2, 1], 0, &dm);
        // 0→2→1→0 = 2+1+1 = 4 vs 0→1→2→0 = 1+1+2 = 4 (same distance on line)
        assert_eq!(improved.len(), 2);
        assert!(dist <= 4.0 + 1e-10);
    }

    #[test]
    fn test_or_opt_with_crossing() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 1.0, -1.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let initial = vec![1, 3, 2]; // crosses
        let initial_dist = route_distance(&initial, 0, &dm);
        let (_, improved_dist) = or_opt_improve(&initial, 0, &dm);
        assert!(improved_dist <= initial_dist + 1e-10);
    }

    #[test]
    fn test_route_distance() {
        let (_, dm) = line_customers();
        let d = route_distance(&[1, 2, 3], 0, &dm);
        assert!((d - 6.0).abs() < 1e-10);
    }
}
