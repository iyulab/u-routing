//! Workload balance: a plan's objective with a term for its busiest route.
//!
//! The cost of a plan (fixed cost plus cost per distance, over its routes) is
//! all a plan minimises by default, so a plan fills its vehicles to capacity.
//! A balanced plan adds `weight × E`, where `E` is the largest workload of any
//! route — the min-max measure. Monotone measures such as the maximum do not
//! reward lengthening a short route to look even, which the range (max − min)
//! and the variance do (Matl, Hartl & Vidal 2018, "Workload equity in vehicle
//! routing problems: a survey and analysis", *Transportation Science* 52(2)).
//!
//! [`improve`] moves customers between the routes of any plan — relocating one
//! or exchanging two — while that lowers `cost + weight × E` and keeps every
//! route within its vehicle's capacity, time windows and limits. Every method
//! ends with it, so every method answers the same objective.

use crate::distance::DistanceMatrix;
use crate::evaluation::{route_checks_needed, route_feasible, RouteLimits};
use crate::models::{Customer, Vehicle};

/// What a route's workload counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantity {
    /// The load as a share of the route's vehicle's capacity, so that a small
    /// and a large vehicle are equally busy when equally full.
    Load,
    /// The route's length, depot to depot.
    Distance,
    /// When the route is back at the depot, leaving it at time 0.
    Duration,
}

impl Quantity {
    /// The wire names, in the order of the variants.
    pub const NAMES: [&'static str; 3] = ["load", "distance", "duration"];

    /// The variant a wire name stands for.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "load" => Some(Quantity::Load),
            "distance" => Some(Quantity::Distance),
            "duration" => Some(Quantity::Duration),
            _ => None,
        }
    }
}

/// The balance term of the objective: what a workload counts, and how many
/// units of cost one unit of the largest workload is worth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Balance {
    pub quantity: Quantity,
    /// `>= 0`; `0` leaves the objective the cost alone.
    pub weight: f64,
}

/// A route's distance and duration (the clock of [`route_feasible`]), and its
/// load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteFigures {
    pub distance: f64,
    pub duration: f64,
    pub load: i64,
    /// What the route's lateness at soft windows costs, in units of distance (0 when every window
    /// is hard); the route is priced at [`RouteFigures::priced_distance`].
    pub lateness_cost: f64,
}

impl RouteFigures {
    pub fn of(route: &[usize], dm: &DistanceMatrix, customers: &[Customer]) -> Self {
        let (mut time, mut distance, mut load, mut prev) = (0.0, 0.0, 0_i64, 0);
        let mut lateness_cost = 0.0;
        for &c in route {
            distance += dm.get(prev, c);
            let arrival = time + dm.travel_time(prev, c);
            let customer = &customers[c];
            let start = match customer.time_window() {
                Some(tw) => {
                    if let Some(penalty) = customer.lateness_penalty() {
                        lateness_cost += penalty * tw.lateness(arrival);
                    }
                    arrival + tw.waiting_time(arrival)
                }
                None => arrival,
            };
            time = start + customer.service_duration();
            load += i64::from(customer.demand());
            prev = c;
        }
        RouteFigures {
            distance: distance + dm.get(prev, 0),
            duration: time + dm.travel_time(prev, 0),
            load,
            lateness_cost,
        }
    }

    /// The distance the route is priced at: its distance plus the cost of its lateness.
    pub fn priced_distance(&self) -> f64 {
        self.distance + self.lateness_cost
    }

    /// The workload `quantity` counts, on a route driven by `vehicle`.
    pub fn workload(&self, quantity: Quantity, vehicle: &Vehicle) -> f64 {
        match quantity {
            Quantity::Load => self.load as f64 / f64::from(vehicle.capacity()),
            Quantity::Distance => self.distance,
            Quantity::Duration => self.duration,
        }
    }
}

/// The routes of a plan as [`improve`] keeps them.
struct State<'a> {
    routes: Vec<Vec<usize>>,
    kinds: Vec<usize>,
    vehicles: &'a [Vehicle],
    customers: &'a [Customer],
    dm: &'a DistanceMatrix,
    balance: Balance,
    cost: Vec<f64>,
    work: Vec<f64>,
    load: Vec<i64>,
}

impl State<'_> {
    fn objective(&self) -> f64 {
        self.cost.iter().sum::<f64>() + self.balance.weight * self.largest_except(&[])
    }

    /// The largest workload among the routes not in `skip`.
    fn largest_except(&self, skip: &[usize]) -> f64 {
        self.work
            .iter()
            .enumerate()
            .filter(|(r, _)| !skip.contains(r))
            .map(|(_, &w)| w)
            .fold(0.0, f64::max)
    }

    /// The cost and workload of `route` driven by `kind`, or `None` when the
    /// vehicle cannot drive it (capacity, time windows, limits). An empty
    /// route is no route: it costs nothing and does no work.
    fn figures(&self, route: &[usize], kind: usize) -> Option<(f64, f64, i64)> {
        if route.is_empty() {
            return Some((0.0, 0.0, 0));
        }
        let vehicle = &self.vehicles[kind];
        let fig = RouteFigures::of(route, self.dm, self.customers);
        if fig.load > i64::from(vehicle.capacity()) {
            return None;
        }
        let limits = RouteLimits::of(vehicle);
        if route_checks_needed(self.customers, &limits)
            && !route_feasible(route, 0, self.dm, self.customers, &limits)
        {
            return None;
        }
        Some((
            vehicle.route_cost(fig.priced_distance()),
            fig.workload(self.balance.quantity, vehicle),
            fig.load,
        ))
    }

    /// Takes the change of routes `a` and `b` to `ra` and `rb` when it lowers
    /// the objective; returns whether it did.
    fn try_move(&mut self, a: usize, b: usize, ra: Vec<usize>, rb: Vec<usize>) -> bool {
        let Some((ca, wa, la)) = self.figures(&ra, self.kinds[a]) else {
            return false;
        };
        let Some((cb, wb, lb)) = self.figures(&rb, self.kinds[b]) else {
            return false;
        };
        let before = self.objective();
        let largest = self.largest_except(&[a, b]).max(wa).max(wb);
        let total: f64 = self
            .cost
            .iter()
            .enumerate()
            .filter(|(r, _)| *r != a && *r != b)
            .map(|(_, &c)| c)
            .sum::<f64>()
            + ca
            + cb;
        let after = total + self.balance.weight * largest;
        if after < before - 1e-9 * before.abs().max(1.0) {
            self.routes[a] = ra;
            self.routes[b] = rb;
            (self.cost[a], self.work[a], self.load[a]) = (ca, wa, la);
            (self.cost[b], self.work[b], self.load[b]) = (cb, wb, lb);
            true
        } else {
            false
        }
    }

    /// One pass of relocate and exchange moves out of the busiest route;
    /// whether any move was taken.
    fn pass(&mut self) -> bool {
        let busiest = (0..self.routes.len())
            .max_by(|&x, &y| self.work[x].total_cmp(&self.work[y]))
            .expect("improve runs on two routes or more");
        let mut moved = false;
        // Relocate: a customer of the busiest route to any place of another.
        let mut i = 0;
        while i < self.routes[busiest].len() {
            let c = self.routes[busiest][i];
            let mut taken = false;
            'targets: for b in 0..self.routes.len() {
                if b == busiest || self.routes[b].is_empty() {
                    continue;
                }
                if self.load[b] + i64::from(self.customers[c].demand())
                    > i64::from(self.vehicles[self.kinds[b]].capacity())
                {
                    continue;
                }
                for pos in 0..=self.routes[b].len() {
                    let mut ra = self.routes[busiest].clone();
                    ra.remove(i);
                    let mut rb = self.routes[b].clone();
                    rb.insert(pos, c);
                    if self.try_move(busiest, b, ra, rb) {
                        taken = true;
                        break 'targets;
                    }
                }
            }
            if taken {
                moved = true;
            } else {
                i += 1;
            }
        }
        // Exchange: a customer of the busiest route with one of another.
        for i in 0..self.routes[busiest].len() {
            for b in 0..self.routes.len() {
                if b == busiest {
                    continue;
                }
                for j in 0..self.routes[b].len() {
                    let (Some(&u), Some(&v)) = (self.routes[busiest].get(i), self.routes[b].get(j))
                    else {
                        continue;
                    };
                    let mut ra = self.routes[busiest].clone();
                    let mut rb = self.routes[b].clone();
                    ra[i] = v;
                    rb[j] = u;
                    if self.try_move(busiest, b, ra, rb) {
                        moved = true;
                    }
                }
            }
        }
        moved
    }
}

/// The routes and their vehicles after balance moves: relocations and
/// exchanges between routes, each taken when it lowers `cost + weight × E`
/// and keeps both routes feasible for their vehicles, until none does (or
/// `max_passes` passes). Routes left empty are dropped. Never more routes than
/// given, the same customers.
///
/// A move that lowers the objective must touch the busiest route or lower the
/// cost; the passes look for it among moves out of the busiest route, which
/// is the route that sets `E`.
pub fn improve(
    routes: Vec<Vec<usize>>,
    kinds: Vec<usize>,
    vehicles: &[Vehicle],
    customers: &[Customer],
    dm: &DistanceMatrix,
    balance: Balance,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    if balance.weight <= 0.0 || routes.len() < 2 {
        return (routes, kinds);
    }
    let mut state = State {
        cost: Vec::with_capacity(routes.len()),
        work: Vec::with_capacity(routes.len()),
        load: Vec::with_capacity(routes.len()),
        routes,
        kinds,
        vehicles,
        customers,
        dm,
        balance,
    };
    for r in 0..state.routes.len() {
        let fig = RouteFigures::of(&state.routes[r], dm, customers);
        let vehicle = &vehicles[state.kinds[r]];
        state.cost.push(vehicle.route_cost(fig.priced_distance()));
        state.work.push(fig.workload(balance.quantity, vehicle));
        state.load.push(fig.load);
    }
    let max_passes = 4 * state.routes.iter().map(Vec::len).sum::<usize>().max(1);
    for _ in 0..max_passes {
        if !state.pass() {
            break;
        }
    }
    let mut routes = Vec::with_capacity(state.routes.len());
    let mut kinds = Vec::with_capacity(state.routes.len());
    for (route, kind) in state.routes.into_iter().zip(state.kinds) {
        if !route.is_empty() {
            routes.push(route);
            kinds.push(kind);
        }
    }
    (routes, kinds)
}
