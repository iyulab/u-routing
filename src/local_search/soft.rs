//! The intra-route searches when a customer has a soft window.
//!
//! [`two_opt_improve`](super::two_opt_improve), [`or_opt_improve`](super::or_opt_improve) and
//! [`three_opt_improve`](super::three_opt_improve) choose a move by the change in *distance*, which
//! they read off the edges the move replaces in O(1), and ask the feasibility check only about the
//! move they have chosen. With a soft window that picture is wrong in both directions: a move that
//! shortens the route can make a vehicle late, and one that lengthens it can save the lateness it
//! costs. When any customer has a soft window each of the three is this search instead: it generates
//! the neighbourhood of the move, evaluates every neighbour whole ([`route_eval`], O(n)) and takes
//! the one of least *effective cost* -- the distance plus the cost of the lateness -- as long as it
//! beats the route by the gain floor (steepest descent). Routes with no soft window never come here,
//! so every result without a penalty is what it was.

use super::gain_floor;
use super::or_opt::route_distance;
use crate::distance::DistanceMatrix;
use crate::evaluation::{route_eval, RouteLimits};
use crate::models::Customer;

/// Which neighbourhood a search moves in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Neighbourhood {
    /// Reverse a stretch of the route.
    Reverse,
    /// Move a stretch of one to three customers elsewhere, in the same direction.
    Relocate,
    /// Cut at three edges and reconnect the four pieces in every other way.
    Reconnect,
}

/// What the searches minimise on one route: its distance plus the cost of its lateness, infinite
/// if the route is not allowed.
pub(crate) fn effective_cost(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
) -> f64 {
    route_eval(route, depot, distances, customers, limits)
        .map_or(f64::INFINITY, |e| e.distance + e.lateness_cost)
}

/// Steepest descent in `which` over the effective cost. Returns the route and its distance -- the
/// distance, as the searches it stands in for return, not the effective cost.
pub(crate) fn improve(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    limits: &RouteLimits,
    which: Neighbourhood,
) -> (Vec<usize>, f64) {
    let floor = gain_floor(distances);
    let mut current = route.to_vec();
    let mut cost = effective_cost(&current, depot, distances, customers, limits);
    loop {
        let mut best_cost = cost - floor;
        let mut best: Option<Vec<usize>> = None;
        for_each_neighbour(&current, which, &mut |candidate| {
            let c = effective_cost(candidate, depot, distances, customers, limits);
            if c < best_cost {
                best_cost = c;
                best = Some(candidate.to_vec());
            }
        });
        match best {
            Some(next) => {
                current = next;
                cost = best_cost;
            }
            None => break,
        }
    }
    let distance = route_distance(&current, depot, distances);
    (current, distance)
}

/// Calls `visit` with every route one move of `which` away from `route`.
pub(crate) fn for_each_neighbour(
    route: &[usize],
    which: Neighbourhood,
    visit: &mut dyn FnMut(&[usize]),
) {
    let n = route.len();
    let mut buffer = Vec::with_capacity(n);
    match which {
        Neighbourhood::Reverse => {
            for i in 0..n.saturating_sub(1) {
                for j in i + 1..n {
                    buffer.clear();
                    buffer.extend_from_slice(&route[..i]);
                    buffer.extend(route[i..=j].iter().rev());
                    buffer.extend_from_slice(&route[j + 1..]);
                    visit(&buffer);
                }
            }
        }
        Neighbourhood::Relocate => {
            for len in 1..=3.min(n) {
                if n < len + 1 {
                    continue;
                }
                for from in 0..=n - len {
                    let segment = &route[from..from + len];
                    let rest: Vec<usize> = route[..from]
                        .iter()
                        .chain(&route[from + len..])
                        .copied()
                        .collect();
                    for at in 0..=rest.len() {
                        if at == from {
                            continue;
                        }
                        buffer.clear();
                        buffer.extend_from_slice(&rest[..at]);
                        buffer.extend_from_slice(segment);
                        buffer.extend_from_slice(&rest[at..]);
                        visit(&buffer);
                    }
                }
            }
        }
        Neighbourhood::Reconnect => {
            if n < 4 {
                return;
            }
            for i in 0..n - 2 {
                for j in i + 1..n - 1 {
                    for k in j + 1..n {
                        let (a, b, c, d) = (
                            &route[..=i],
                            &route[i + 1..=j],
                            &route[j + 1..=k],
                            &route[k + 1..],
                        );
                        // Each pattern as (first piece, reversed?) of the middle two.
                        for (first, first_rev, second, second_rev) in [
                            (b, false, c, true),
                            (b, true, c, false),
                            (b, true, c, true),
                            (c, false, b, false),
                            (c, false, b, true),
                            (c, true, b, false),
                            (c, true, b, true),
                        ] {
                            buffer.clear();
                            buffer.extend_from_slice(a);
                            push_piece(&mut buffer, first, first_rev);
                            push_piece(&mut buffer, second, second_rev);
                            buffer.extend_from_slice(d);
                            visit(&buffer);
                        }
                    }
                }
            }
        }
    }
}

fn push_piece(buffer: &mut Vec<usize>, piece: &[usize], reversed: bool) {
    if reversed {
        buffer.extend(piece.iter().rev());
    } else {
        buffer.extend_from_slice(piece);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    /// Customer 1 far out at 10 with a soft window closing at 10; customers 2 and 3 close to the
    /// depot; 5 units of service each.
    fn far_one_is_due() -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 10.0, 0.0, 1, 5.0),
            Customer::new(2, 1.0, 0.0, 1, 5.0),
            Customer::new(3, 2.0, 0.0, 1, 5.0),
        ];
        let window = TimeWindow::new(0.0, 10.0).expect("valid");
        customers[1] = customers[1]
            .clone()
            .with_time_window(window)
            .with_lateness_penalty(5.0)
            .expect("valid");
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn a_shorter_route_that_is_later_costs_more_and_the_search_ends_on_time() {
        let (customers, dm) = far_one_is_due();
        let none = RouteLimits::NONE;
        // Visiting 1 first: on time, 22 long. Visiting it last: 20 long, 10 late -- 50 of penalty.
        assert_eq!(route_distance(&[1, 2, 3], 0, &dm), 22.0);
        assert_eq!(route_distance(&[2, 3, 1], 0, &dm), 20.0);
        assert_eq!(effective_cost(&[1, 2, 3], 0, &dm, &customers, &none), 22.0);
        assert_eq!(effective_cost(&[2, 3, 1], 0, &dm, &customers, &none), 70.0);

        // From the late route the search moves 1 to the front (70 to 22) and then to the shorter
        // on-time order (20); from the on-time one it only takes the shorter on-time order.
        for start in [[2, 3, 1], [1, 2, 3]] {
            let (end, distance) =
                improve(&start, 0, &dm, &customers, &none, Neighbourhood::Relocate);
            assert_eq!(
                (end.clone(), distance),
                (vec![1, 3, 2], 20.0),
                "from {start:?}"
            );
            let eval = route_eval(&end, 0, &dm, &customers, &none).expect("allowed");
            assert_eq!(eval.lateness, 0.0);
        }
        // The same through the public search.
        let (end, _) = super::super::or_opt_improve(&[2, 3, 1], 0, &dm, &customers, &none);
        assert_eq!(end, vec![1, 3, 2]);
    }

    #[test]
    fn the_neighbourhoods_have_the_size_they_should() {
        let count = |n: usize, which| {
            let route: Vec<usize> = (1..=n).collect();
            let mut seen = 0;
            for_each_neighbour(&route, which, &mut |_| seen += 1);
            seen
        };
        // Reversals of every stretch of two or more.
        assert_eq!(count(5, Neighbourhood::Reverse), 10);
        // Moves of a stretch of 1, 2 or 3 to each of the other n - len positions.
        assert_eq!(count(5, Neighbourhood::Relocate), 5 * 4 + 4 * 3 + 3 * 2);
        // Seven reconnections for each choice of three positions out of n: C(5, 3) = 10.
        assert_eq!(count(5, Neighbourhood::Reconnect), 7 * 10);
        assert_eq!(count(3, Neighbourhood::Reconnect), 0);
    }

    #[test]
    fn every_neighbour_is_a_permutation_of_the_route() {
        let route = [4, 2, 7, 1, 5, 3];
        for which in [
            Neighbourhood::Reverse,
            Neighbourhood::Relocate,
            Neighbourhood::Reconnect,
        ] {
            for_each_neighbour(&route, which, &mut |candidate| {
                let mut sorted = candidate.to_vec();
                sorted.sort_unstable();
                assert_eq!(sorted, vec![1, 2, 3, 4, 5, 7], "{which:?}");
                // A reversed piece of one customer is the piece, so a 3-opt pattern can be the route;
                // that is harmless (no gain) and not worth a test of its own.
                if which != Neighbourhood::Reconnect {
                    assert_ne!(candidate, route, "{which:?} offered the route itself");
                }
            });
        }
    }

    #[test]
    fn the_search_never_ends_worse_and_ends_where_no_neighbour_is_better() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |bound: f64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * bound
        };
        let mut improved_some = 0;
        for _ in 0..40 {
            let mut customers = vec![Customer::depot(0.0, 0.0)];
            for id in 1..=6 {
                let mut c = Customer::new(id, next(30.0), next(30.0), 1, next(4.0));
                if next(1.0) < 0.7 {
                    let ready = next(20.0);
                    c = c.with_time_window(
                        TimeWindow::new(ready, ready + next(25.0)).expect("valid"),
                    );
                    if next(1.0) < 0.7 {
                        c = c.with_lateness_penalty(next(6.0)).expect("valid");
                    }
                }
                customers.push(c);
            }
            // A route that is allowed: soft windows only, so make every window soft.
            for c in customers.iter_mut().skip(1) {
                if c.time_window().is_some() && c.lateness_penalty().is_none() {
                    *c = c.clone().with_lateness_penalty(1.0).expect("valid");
                }
            }
            let dm = DistanceMatrix::from_customers(&customers);
            let none = RouteLimits::NONE;
            let start = [1, 2, 3, 4, 5, 6];
            let start_cost = effective_cost(&start, 0, &dm, &customers, &none);
            for which in [
                Neighbourhood::Reverse,
                Neighbourhood::Relocate,
                Neighbourhood::Reconnect,
            ] {
                let (end, _) = improve(&start, 0, &dm, &customers, &none, which);
                let end_cost = effective_cost(&end, 0, &dm, &customers, &none);
                assert!(end_cost <= start_cost + 1e-9, "{which:?} ended worse");
                improved_some += usize::from(end_cost < start_cost - 1e-9);
                for_each_neighbour(&end, which, &mut |candidate| {
                    let c = effective_cost(candidate, 0, &dm, &customers, &none);
                    assert!(
                        c >= end_cost - gain_floor(&dm) - 1e-9,
                        "{which:?} stopped early"
                    );
                });
            }
        }
        assert!(
            improved_some > 20,
            "the searches never moved: {improved_some}"
        );
    }
}
