//! The clock of a route that carries a driver's break ([`VehicleBreak`]).
//!
//! Without a break a route's clock is one pass (see [`route_feasible`]). With one, the break can
//! go at any stop, and the best stop depends on the windows after it, so the clock keeps two
//! labels per position -- the earliest the vehicle can have served the stops so far with the
//! break still to take, and with it taken -- and ends on whichever the break rule asks for.
//! Arriving earlier is never worse (the vehicle can wait anywhere), which is what makes the
//! earliest time per label enough: O(n) and allocation-free like the plain pass.
//!
//! [`route_feasible`]: super::route_feasible

use crate::distance::DistanceMatrix;
use crate::models::{Customer, VehicleBreak};

/// What the clock reports of a route that can carry its break.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scheduled {
    /// Total distance, depot to depot.
    pub distance: f64,
    /// Return to the depot, on the clock that starts at 0.
    pub duration: f64,
    /// Time late at soft windows.
    pub lateness: f64,
    /// That lateness times each customer's penalty.
    pub lateness_cost: f64,
    /// Where the break goes: the index in the route of the stop it comes before the service of,
    /// and when it starts. `None` when the vehicle is back at the depot by the break's `earliest`.
    pub rest: Option<(usize, f64)>,
}

/// The vehicle at a stop with its service done.
#[derive(Debug, Clone, Copy)]
struct Label {
    time: f64,
    lateness: f64,
    lateness_cost: f64,
    /// `(stop index, start)` of the break, if taken.
    rest: Option<(usize, f64)>,
}

impl Label {
    const START: Label = Label {
        time: 0.0,
        lateness: 0.0,
        lateness_cost: 0.0,
        rest: None,
    };

    /// Whichever of the two is done earlier, the cheaper in lateness on a tie.
    fn earlier(a: Option<Label>, b: Option<Label>) -> Option<Label> {
        match (a, b) {
            (Some(x), Some(y)) => {
                if (y.time, y.lateness_cost) < (x.time, x.lateness_cost) {
                    Some(y)
                } else {
                    Some(x)
                }
            }
            (x, None) => x,
            (None, y) => y,
        }
    }
}

/// One stop of `label`'s route: travel to `cid`, the break first if `rest` gives its start
/// (`index` is the stop's index in the route), the wait for the window, the service. `None` when a
/// hard window closes before the service can start, or the break cannot start by its `latest`.
fn serve(
    label: &Label,
    travel: f64,
    cid: usize,
    index: usize,
    rest: Option<&VehicleBreak>,
    customers: &[Customer],
) -> Option<Label> {
    let arrival = label.time + travel;
    let (ready, taken) = match rest {
        Some(b) => {
            let start = arrival.max(b.earliest());
            if start > b.latest() {
                return None;
            }
            (start + b.duration(), Some((index, start)))
        }
        None => (arrival, label.rest),
    };
    let customer = &customers[cid];
    let mut lateness = label.lateness;
    let mut lateness_cost = label.lateness_cost;
    let service_start = match customer.time_window() {
        Some(tw) => {
            if tw.is_violated(ready) {
                let penalty = customer.lateness_penalty()?;
                let late = tw.lateness(ready);
                lateness += late;
                lateness_cost += penalty * late;
            }
            ready + tw.waiting_time(ready)
        }
        None => ready,
    };
    Some(Label {
        time: service_start + customer.service_duration(),
        lateness,
        lateness_cost,
        rest: taken,
    })
}

/// The clock of a route being extended one stop at a time: the two labels of the module
/// documentation. A search that grows a route (the split of a giant tour) keeps one and asks
/// [`BreakClock::finish`] at every end it considers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BreakClock<'a> {
    rest: &'a VehicleBreak,
    /// The break is still to take.
    before: Option<Label>,
    /// The break is taken.
    after: Option<Label>,
    prev: usize,
    index: usize,
}

impl<'a> BreakClock<'a> {
    /// The vehicle at `depot` at time 0.
    pub(crate) fn new(rest: &'a VehicleBreak, depot: usize) -> Self {
        Self {
            rest,
            before: Some(Label::START),
            after: None,
            prev: depot,
            index: 0,
        }
    }

    /// Drives on to `cid`. `false` when no way of placing the break lets the vehicle serve the
    /// stops so far -- a hard window closes, or the break cannot start by its `latest` -- so no
    /// longer route does either.
    pub(crate) fn push(
        &mut self,
        cid: usize,
        distances: &DistanceMatrix,
        customers: &[Customer],
    ) -> bool {
        let (index, rest) = (self.index, self.rest);
        let travel = distances.travel_time(self.prev, cid);
        let skipped = self
            .before
            .as_ref()
            .and_then(|l| serve(l, travel, cid, index, None, customers));
        let here = self
            .before
            .as_ref()
            .and_then(|l| serve(l, travel, cid, index, Some(rest), customers));
        let carried = self
            .after
            .as_ref()
            .and_then(|l| serve(l, travel, cid, index, None, customers));
        self.before = skipped;
        self.after = Label::earlier(carried, here);
        self.prev = cid;
        self.index += 1;
        self.before.is_some() || self.after.is_some()
    }

    /// The route ended by the return to `depot` from the last stop, with `distance` travelled in
    /// all (the clock does not track it): `None` if the break cannot be placed.
    pub(crate) fn finish(
        &self,
        distance: f64,
        depot: usize,
        distances: &DistanceMatrix,
    ) -> Option<Scheduled> {
        let back = distances.travel_time(self.prev, depot);
        // Home before the break is due: no break needed.
        if let Some(l) = self.before {
            if l.time + back <= self.rest.earliest() {
                return Some(Scheduled {
                    distance,
                    duration: l.time + back,
                    lateness: l.lateness,
                    lateness_cost: l.lateness_cost,
                    rest: None,
                });
            }
        }
        self.after.map(|l| Scheduled {
            distance,
            duration: l.time + back,
            lateness: l.lateness,
            lateness_cost: l.lateness_cost,
            rest: l.rest,
        })
    }
}

/// The clock of `route` with `rest`: `None` if a hard window cannot be kept, or the break cannot
/// be placed (the vehicle is still out at `earliest` and no stop lets it start by `latest`).
/// The limits of the vehicle are the caller's to apply to the distance and duration returned.
pub(crate) fn schedule(
    route: &[usize],
    depot: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    rest: &VehicleBreak,
) -> Option<Scheduled> {
    let mut clock = BreakClock::new(rest, depot);
    let mut distance = 0.0;
    let mut prev = depot;
    for &cid in route {
        distance += distances.get(prev, cid);
        if !clock.push(cid, distances, customers) {
            return None;
        }
        prev = cid;
    }
    distance += distances.get(prev, depot);
    clock.finish(distance, depot, distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TimeWindow;

    /// A line: depot at 0, customer `i` at `10 i`; travel time = distance; service 2.
    fn line(windows: &[Option<(f64, f64, Option<f64>)>]) -> (Vec<Customer>, DistanceMatrix) {
        let mut customers = vec![Customer::depot(0.0, 0.0)];
        for (i, w) in windows.iter().enumerate() {
            let mut c = Customer::new(i + 1, 10.0 * (i + 1) as f64, 0.0, 1, 2.0);
            if let Some((ready, due, penalty)) = w {
                c = c.with_time_window(TimeWindow::new(*ready, *due).expect("window"));
                if let Some(p) = penalty {
                    c = c.with_lateness_penalty(*p).expect("penalty");
                }
            }
            customers.push(c);
        }
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    /// A plain simulation of `route` with the break before stop `at` (none: no break):
    /// `(return time, lateness cost)`, `None` if a hard window or the break's `latest` is missed.
    fn simulate(
        route: &[usize],
        customers: &[Customer],
        dm: &DistanceMatrix,
        b: &VehicleBreak,
        at: Option<usize>,
    ) -> Option<(f64, f64)> {
        let (mut time, mut cost, mut prev) = (0.0, 0.0, 0);
        for (k, &cid) in route.iter().enumerate() {
            let mut t = time + dm.travel_time(prev, cid);
            if at == Some(k) {
                let start = t.max(b.earliest());
                if start > b.latest() {
                    return None;
                }
                t = start + b.duration();
            }
            let c = &customers[cid];
            if let Some(tw) = c.time_window() {
                if t > tw.due() {
                    cost += c.lateness_penalty()? * (t - tw.due());
                }
                t = t.max(tw.ready());
            }
            time = t + c.service_duration();
            prev = cid;
        }
        Some((time + dm.travel_time(prev, 0), cost))
    }

    /// The definition, by trying every place the break can go: the break before stop `k` (or none,
    /// if the vehicle is home by `earliest`), the best of them as `(duration, lateness cost)`.
    fn oracle(
        route: &[usize],
        customers: &[Customer],
        dm: &DistanceMatrix,
        b: &VehicleBreak,
    ) -> Option<(f64, f64)> {
        if let Some((end, cost)) = simulate(route, customers, dm, b, None) {
            if end <= b.earliest() {
                return Some((end, cost));
            }
        }
        (0..route.len())
            .filter_map(|k| simulate(route, customers, dm, b, Some(k)))
            .min_by(|a, b| a.partial_cmp(b).expect("finite"))
    }

    #[test]
    fn a_route_home_before_the_break_is_due_takes_none() {
        let (customers, dm) = line(&[None, None]);
        let b = VehicleBreak::new(100.0, 120.0, 10.0);
        // 10 + 2 + 10 + 2 + 20 = 44.
        let s = schedule(&[1, 2], 0, &dm, &customers, &b).expect("feasible");
        assert_eq!((s.duration, s.rest), (44.0, None));
    }

    #[test]
    fn the_break_is_taken_at_the_first_stop_once_it_is_due_and_lengthens_the_route() {
        let (customers, dm) = line(&[None, None, None]);
        // Stops are reached at 10, 22, 34; the break is due from 20.
        let b = VehicleBreak::new(20.0, 40.0, 5.0);
        let s = schedule(&[1, 2, 3], 0, &dm, &customers, &b).expect("feasible");
        // Break before the second stop (arrival 22 >= 20): 22 + 5 + 2 -> leaves 29, 3rd at 39, +2, +30.
        assert_eq!(s.rest, Some((1, 22.0)));
        assert_eq!(
            s.duration,
            10.0 + 2.0 + 10.0 + 5.0 + 2.0 + 10.0 + 2.0 + 30.0
        );
    }

    #[test]
    fn the_break_overlaps_the_wait_for_a_window_to_open() {
        // The first stop opens at 30; the vehicle is there at 10 and waits 20.
        let (customers, dm) = line(&[Some((30.0, 90.0, None))]);
        let b = VehicleBreak::new(0.0, 50.0, 15.0);
        let s = schedule(&[1], 0, &dm, &customers, &b).expect("feasible");
        // Break 10..25, still before the window: service starts at 30 as without a break.
        assert_eq!(s.rest, Some((0, 10.0)));
        assert_eq!(s.duration, 30.0 + 2.0 + 10.0);
    }

    #[test]
    fn a_break_that_cannot_start_by_its_latest_time_is_not_placeable() {
        let (customers, dm) = line(&[None, None]);
        // Out at 44 with the break due from 12 until 13: the stops are reached at 10 and 22.
        let b = VehicleBreak::new(12.0, 13.0, 5.0);
        // At the first stop (10 < 12) wait to 12 and break: start 12 <= 13 -- placeable there.
        let s = schedule(&[1, 2], 0, &dm, &customers, &b).expect("feasible");
        assert_eq!(s.rest, Some((0, 12.0)));
        // With the first stop reached after `latest` and nothing earlier: not placeable.
        let b = VehicleBreak::new(1.0, 5.0, 5.0);
        assert!(schedule(&[1, 2], 0, &dm, &customers, &b).is_none());
    }

    #[test]
    fn a_hard_window_the_break_pushes_past_its_due_time_forbids_that_place() {
        // Stop 1 closes at 14: a break there (arrival 10, 10 + 8) would start service at 18.
        let (customers, dm) = line(&[Some((0.0, 14.0, None)), None]);
        let b = VehicleBreak::new(5.0, 40.0, 8.0);
        let s = schedule(&[1, 2], 0, &dm, &customers, &b).expect("feasible");
        assert_eq!(s.rest, Some((1, 22.0)));
    }

    #[test]
    fn the_clock_equals_the_definition_on_random_routes() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move |n: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % n
        };
        let mut checked = 0;
        let mut with_break = 0;
        for _ in 0..4000 {
            let m = 1 + next(6) as usize;
            let windows: Vec<_> = (0..m)
                .map(|_| match next(3) {
                    0 => None,
                    _ => {
                        let ready = next(60) as f64;
                        let due = ready + next(50) as f64;
                        let penalty = (next(2) == 0).then(|| 1.0 + next(3) as f64);
                        Some((ready, due, penalty))
                    }
                })
                .collect();
            let (customers, dm) = line(&windows);
            let earliest = next(80) as f64;
            let b = VehicleBreak::new(earliest, earliest + next(40) as f64, 1.0 + next(12) as f64);
            let mut route: Vec<usize> = (1..=m).collect();
            // A shuffle: the order the clock sees is what matters.
            for i in (1..m).rev() {
                route.swap(i, next(i as u64 + 1) as usize);
            }
            let got = schedule(&route, 0, &dm, &customers, &b);
            let want = oracle(&route, &customers, &dm, &b);
            match (got, want) {
                (None, None) => {}
                (Some(g), Some((end, cost))) => {
                    assert!(
                        (g.duration - end).abs() < 1e-9 && (g.lateness_cost - cost).abs() < 1e-9,
                        "{route:?} {b:?}: got {g:?}, oracle {end} {cost}"
                    );
                    // The place reported is one the definition accepts at that value (equal
                    // places are a tie, not an error).
                    let at = g.rest.map(|r| r.0);
                    let (e, c) = simulate(&route, &customers, &dm, &b, at).expect("a place");
                    assert!(
                        (e - end).abs() < 1e-9 && (c - cost).abs() < 1e-9,
                        "{route:?} {b:?}"
                    );
                    with_break += usize::from(at.is_some());
                }
                (g, w) => panic!("{route:?} {b:?}: got {g:?}, oracle {w:?}"),
            }
            checked += 1;
        }
        assert_eq!(checked, 4000);
        assert!(with_break > 400, "{with_break} routes took the break");
    }
}
