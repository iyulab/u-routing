//! Intra-route 3-opt improvement.
//!
//! # Algorithm
//!
//! Examines all triples of edges in a route and evaluates reconnection
//! options. For three cut edges, there are 8 possible reconnection patterns;
//! the identity is excluded, leaving 7 candidates (including 2-opt moves
//! as special cases).
//!
//! Uses first-improvement strategy: applies the first improving move found
//! and restarts the search.
//!
//! # Complexity
//!
//! O(n³) per pass, O(n⁴) worst case for convergence.
//!
//! # Reference
//!
//! Lin, S. (1965). "Computer Solutions of the Traveling Salesman Problem",
//! *Bell System Technical Journal* 44(10), 2245-2269.

use super::or_opt::route_distance;
use crate::distance::DistanceMatrix;
use crate::evaluation::{has_time_windows, time_windows_respected};
use crate::models::Customer;

/// Applies 3-opt improvement to a single route.
///
/// Tries all possible 3-edge reconnection patterns and accepts the first
/// improvement found. Returns the improved customer sequence and total distance.
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
/// use u_routing::local_search::{three_opt_improve, route_distance};
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 2.0, 0.0, 10, 0.0),
///     Customer::new(2, 3.0, 1.0, 10, 0.0),
///     Customer::new(3, 1.0, 1.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// let (improved, dist) = three_opt_improve(&[1, 3, 2], 0, &dm, &customers);
/// let orig_dist = route_distance(&[1, 3, 2], 0, &dm);
/// assert!(dist <= orig_dist + 1e-10);
/// ```
pub fn three_opt_improve(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
) -> (Vec<usize>, f64) {
    if route.len() < 4 {
        // 3-opt needs at least 4 customers to have 3 non-adjacent edges
        let dist = route_distance(route, depot, distances);
        return (route.to_vec(), dist);
    }

    let windowed = has_time_windows(customers);
    let accepts = |candidate: &[usize]| {
        !windowed || time_windows_respected(candidate, depot, distances, customers)
    };
    let mut current = route.to_vec();
    let mut improved = true;

    while improved {
        improved = false;
        let n = current.len();

        'outer: for i in 0..n - 2 {
            for j in (i + 1)..n - 1 {
                for k in (j + 1)..n {
                    if let Some(new_route) =
                        try_three_opt_move(&current, depot, distances, i, j, k, &accepts)
                    {
                        current = new_route;
                        improved = true;
                        break 'outer;
                    }
                }
            }
        }
    }

    let dist = route_distance(&current, depot, distances);
    (current, dist)
}

/// Tries all 3-opt reconnection patterns for cut positions (i, j, k).
///
/// We cut the route into 4 segments:
///   A = depot..route[i], B = route[i+1..=j], C = route[j+1..=k], D = route[k+1..]..depot
///
/// Edge cuts at: (prev_i → route[i+1]), (route[j] → route[j+1]), (route[k] → next_k)
///
/// Returns Some(new_route) if an improving reconnection is found.
fn try_three_opt_move(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    i: usize,
    j: usize,
    k: usize,
    accepts: &dyn Fn(&[usize]) -> bool,
) -> Option<Vec<usize>> {
    let n = route.len();

    // Segment endpoints for cost calculation
    let a_end = route[i];
    let b_start = route[i + 1];
    let b_end = route[j];
    let c_start = route[j + 1];
    let c_end = route[k];
    let d_start = if k + 1 < n { route[k + 1] } else { depot };

    // Current cost of the three edges being replaced
    let old_cost = distances.get(a_end, b_start)
        + distances.get(b_end, c_start)
        + distances.get(c_end, d_start);

    // Segments (as slices)
    let seg_a = &route[..=i];
    let seg_b = &route[i + 1..=j];
    let seg_c = &route[j + 1..=k];
    let seg_d = &route[k + 1..];

    // Every improving pattern, best first; the first one the gate accepts is
    // taken, so a window-violating reconnection yields to the next best.
    let mut improving: Vec<(f64, u8)> = Vec::with_capacity(7);

    // Pattern 1: A - B - C' - D (reverse C only, = 2-opt on (j, k))
    let cost1 = distances.get(a_end, b_start)
        + distances.get(b_end, c_end)
        + distances.get(c_start, d_start);
    let delta1 = cost1 - old_cost;
    if delta1 < -1e-10 {
        improving.push((delta1, 1));
    }

    // Pattern 2: A - B' - C - D (reverse B only, = 2-opt on (i, j))
    let cost2 = distances.get(a_end, b_end)
        + distances.get(b_start, c_start)
        + distances.get(c_end, d_start);
    let delta2 = cost2 - old_cost;
    if delta2 < -1e-10 {
        improving.push((delta2, 2));
    }

    // Pattern 3: A - B' - C' - D (reverse both B and C)
    let cost3 = distances.get(a_end, b_end)
        + distances.get(b_start, c_end)
        + distances.get(c_start, d_start);
    let delta3 = cost3 - old_cost;
    if delta3 < -1e-10 {
        improving.push((delta3, 3));
    }

    // Pattern 4: A - C - B - D (swap B and C)
    let cost4 = distances.get(a_end, c_start)
        + distances.get(c_end, b_start)
        + distances.get(b_end, d_start);
    let delta4 = cost4 - old_cost;
    if delta4 < -1e-10 {
        improving.push((delta4, 4));
    }

    // Pattern 5: A - C - B' - D (swap, reverse B)
    let cost5 = distances.get(a_end, c_start)
        + distances.get(c_end, b_end)
        + distances.get(b_start, d_start);
    let delta5 = cost5 - old_cost;
    if delta5 < -1e-10 {
        improving.push((delta5, 5));
    }

    // Pattern 6: A - C' - B - D (swap, reverse C)
    let cost6 = distances.get(a_end, c_end)
        + distances.get(c_start, b_start)
        + distances.get(b_end, d_start);
    let delta6 = cost6 - old_cost;
    if delta6 < -1e-10 {
        improving.push((delta6, 6));
    }

    // Pattern 7: A - C' - B' - D (swap, reverse both)
    let cost7 = distances.get(a_end, c_end)
        + distances.get(c_start, b_end)
        + distances.get(b_start, d_start);
    let delta7 = cost7 - old_cost;
    if delta7 < -1e-10 {
        improving.push((delta7, 7));
    }

    improving.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite deltas"));
    improving
        .into_iter()
        .map(|(_, pattern)| reconnect(seg_a, seg_b, seg_c, seg_d, pattern))
        .find(|candidate| accepts(candidate))
}

/// The route `A · (B, C in the order and orientation of `pattern`) · D`.
fn reconnect(
    seg_a: &[usize],
    seg_b: &[usize],
    seg_c: &[usize],
    seg_d: &[usize],
    pattern: u8,
) -> Vec<usize> {
    let mut new_route = Vec::with_capacity(seg_a.len() + seg_b.len() + seg_c.len() + seg_d.len());
    new_route.extend_from_slice(seg_a);

    match pattern {
        1 => {
            // A - B - C' - D
            new_route.extend_from_slice(seg_b);
            new_route.extend(seg_c.iter().rev());
        }
        2 => {
            // A - B' - C - D
            new_route.extend(seg_b.iter().rev());
            new_route.extend_from_slice(seg_c);
        }
        3 => {
            // A - B' - C' - D
            new_route.extend(seg_b.iter().rev());
            new_route.extend(seg_c.iter().rev());
        }
        4 => {
            // A - C - B - D
            new_route.extend_from_slice(seg_c);
            new_route.extend_from_slice(seg_b);
        }
        5 => {
            // A - C - B' - D
            new_route.extend_from_slice(seg_c);
            new_route.extend(seg_b.iter().rev());
        }
        6 => {
            // A - C' - B - D
            new_route.extend(seg_c.iter().rev());
            new_route.extend_from_slice(seg_b);
        }
        7 => {
            // A - C' - B' - D
            new_route.extend(seg_c.iter().rev());
            new_route.extend(seg_b.iter().rev());
        }
        _ => unreachable!(),
    }

    new_route.extend_from_slice(seg_d);
    new_route
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Customer;

    fn square_customers() -> (Vec<Customer>, DistanceMatrix) {
        // Depot at center, 4 customers at square corners
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 1.0, 10, 0.0),
            Customer::new(2, 1.0, -1.0, 10, 0.0),
            Customer::new(3, -1.0, -1.0, 10, 0.0),
            Customer::new(4, -1.0, 1.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn test_3opt_already_optimal() {
        let (customers, dm) = square_customers();
        // Optimal tour around the square: 1→2→3→4
        let (improved, dist) = three_opt_improve(&[1, 2, 3, 4], 0, &dm, &customers);
        let orig_dist = route_distance(&[1, 2, 3, 4], 0, &dm);
        assert!((dist - orig_dist).abs() < 1e-10);
        assert_eq!(improved.len(), 4);
    }

    #[test]
    fn test_3opt_does_not_worsen() {
        let (customers, dm) = square_customers();
        let initial = vec![1, 3, 2, 4]; // deliberately bad
        let initial_dist = route_distance(&initial, 0, &dm);
        let (_, improved_dist) = three_opt_improve(&initial, 0, &dm, &customers);
        assert!(improved_dist <= initial_dist + 1e-10);
    }

    #[test]
    fn test_3opt_larger_instance() {
        let customers = vec![
            Customer::depot(5.0, 5.0),
            Customer::new(1, 0.0, 0.0, 5, 0.0),
            Customer::new(2, 10.0, 0.0, 5, 0.0),
            Customer::new(3, 10.0, 10.0, 5, 0.0),
            Customer::new(4, 0.0, 10.0, 5, 0.0),
            Customer::new(5, 5.0, 0.0, 5, 0.0),
            Customer::new(6, 5.0, 10.0, 5, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let initial = vec![1, 3, 5, 2, 6, 4]; // scrambled
        let initial_dist = route_distance(&initial, 0, &dm);
        let (_, improved_dist) = three_opt_improve(&initial, 0, &dm, &customers);
        assert!(improved_dist <= initial_dist + 1e-10);
    }

    #[test]
    fn test_3opt_small_routes_passthrough() {
        let (customers, dm) = square_customers();
        // Routes with < 4 customers should pass through unchanged
        let (r1, d1) = three_opt_improve(&[1], 0, &dm, &customers);
        assert_eq!(r1, vec![1]);
        assert!(d1 > 0.0);

        let (r2, d2) = three_opt_improve(&[1, 2], 0, &dm, &customers);
        assert_eq!(r2.len(), 2);
        assert!(d2 > 0.0);

        let (r3, d3) = three_opt_improve(&[1, 2, 3], 0, &dm, &customers);
        assert_eq!(r3.len(), 3);
        assert!(d3 > 0.0);
    }

    #[test]
    fn test_3opt_empty() {
        let (customers, dm) = square_customers();
        let (improved, dist) = three_opt_improve(&[], 0, &dm, &customers);
        assert!(improved.is_empty());
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_3opt_preserves_all_customers() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 2.0, 3.0, 5, 0.0),
            Customer::new(2, 4.0, 1.0, 5, 0.0),
            Customer::new(3, 6.0, 4.0, 5, 0.0),
            Customer::new(4, 3.0, 5.0, 5, 0.0),
            Customer::new(5, 1.0, 4.0, 5, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let initial = vec![1, 4, 2, 5, 3];
        let (improved, _) = three_opt_improve(&initial, 0, &dm, &customers);
        let mut sorted = improved.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_3opt_improves_crossed_route() {
        // Create a route with obvious crossings that 3-opt can fix
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 1.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
            Customer::new(4, 2.0, -1.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let initial = vec![1, 3, 2, 4]; // crosses edges
        let initial_dist = route_distance(&initial, 0, &dm);
        let (_, improved_dist) = three_opt_improve(&initial, 0, &dm, &customers);
        assert!(improved_dist <= initial_dist + 1e-10);
    }
}
