//! Several depots: a route runs out of its vehicle's depot and back to it.
//!
//! Random problems (one to three depots, customers with hard and soft windows,
//! vehicles with limits, each at a depot of its own) against a simulator
//! written from the definition -- coordinates, a clock that waits for a window
//! and serves, a sum of legs -- for every way the library measures a route:
//! `RouteEvaluator::build_route`, `route_eval`, `route_feasible`, and
//! `check_solution` on a plan that reports each route's own figures.

use u_routing::distance::DistanceMatrix;
use u_routing::evaluation::RouteLimits;
use u_routing::evaluation::{check_solution, route_eval, route_feasible, Fleet, RouteEvaluator};
use u_routing::models::{
    num_depots, Customer, Route, Solution, TimeWindow, Vehicle, Violation, ViolationType, Visit,
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n - 1)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

struct Problem {
    customers: Vec<Customer>,
    vehicles: Vec<Vehicle>,
}

fn problem(r: &mut Lcg) -> Problem {
    let depots = 1 + r.below(3);
    let n = 4 + r.below(6);
    let mut customers = Vec::new();
    for d in 0..depots {
        customers.push(Customer::depot_at(
            d,
            r.range(0.0, 100.0),
            r.range(0.0, 100.0),
        ));
    }
    for i in depots..depots + n {
        let mut c = Customer::new(
            i,
            r.range(0.0, 100.0),
            r.range(0.0, 100.0),
            1 + r.below(6) as i32,
            r.range(0.0, 4.0),
        );
        if r.next() < 0.6 {
            let ready = r.range(0.0, 120.0);
            let due = ready + r.range(20.0, 90.0);
            c = c.with_time_window(TimeWindow::new(ready, due).expect("a window"));
            if r.next() < 0.4 {
                c = c
                    .with_lateness_penalty(r.range(0.1, 2.0))
                    .expect("a penalty");
            }
        }
        customers.push(c);
    }
    let vehicles = (0..3)
        .map(|v| {
            let mut vehicle = Vehicle::new(v, 6 + r.below(24) as i32).with_depot(r.below(depots));
            if r.next() < 0.4 {
                vehicle = vehicle.with_max_distance(r.range(120.0, 500.0));
            }
            if r.next() < 0.4 {
                vehicle = vehicle.with_max_duration(r.range(150.0, 600.0));
            }
            vehicle
        })
        .collect();
    Problem {
        customers,
        vehicles,
    }
}

/// What the definition says of a route out of `depot`: distance, return time,
/// the soft lateness and its cost, the customers a hard window closes on
/// before the vehicle arrives, and the load.
#[derive(Debug)]
struct Truth {
    distance: f64,
    duration: f64,
    lateness: f64,
    lateness_cost: f64,
    hard_late: Vec<usize>,
    load: i32,
}

fn simulate(customers: &[Customer], depot: usize, ids: &[usize]) -> Truth {
    let leg = |a: usize, b: usize| {
        ((customers[a].x() - customers[b].x()).powi(2)
            + (customers[a].y() - customers[b].y()).powi(2))
        .sqrt()
    };
    let mut t = Truth {
        distance: 0.0,
        duration: 0.0,
        lateness: 0.0,
        lateness_cost: 0.0,
        hard_late: Vec::new(),
        load: 0,
    };
    let (mut clock, mut at) = (0.0, depot);
    for &id in ids {
        t.distance += leg(at, id);
        let arrive = clock + leg(at, id);
        let c = &customers[id];
        let mut start = arrive;
        if let Some(w) = c.time_window() {
            if arrive > w.due() {
                match c.lateness_penalty() {
                    Some(p) => {
                        t.lateness += arrive - w.due();
                        t.lateness_cost += p * (arrive - w.due());
                    }
                    None => t.hard_late.push(id),
                }
            }
            start = arrive.max(w.ready());
        }
        clock = start + c.service_duration();
        t.load += c.demand();
        at = id;
    }
    t.distance += leg(at, depot);
    t.duration = clock + leg(at, depot);
    t
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

fn name(v: &Violation) -> &'static str {
    match v.kind {
        ViolationType::CapacityExceeded { .. } => "capacity",
        ViolationType::TimeWindowViolated { .. } => "window",
        ViolationType::MaxDistanceExceeded { .. } => "distance",
        ViolationType::MaxDurationExceeded { .. } => "duration",
        ViolationType::DistanceMismatch { .. } => "distance_mismatch",
        ViolationType::LatenessMismatch { .. } => "lateness_mismatch",
        _ => "structure",
    }
}

/// The violations of one route the definition predicts, by name.
fn expected(truth: &Truth, vehicle: &Vehicle) -> Vec<&'static str> {
    let mut v = Vec::new();
    if truth.load > vehicle.capacity() {
        v.push("capacity");
    }
    v.extend(truth.hard_late.iter().map(|_| "window"));
    if vehicle.max_distance().is_some_and(|m| truth.distance > m) {
        v.push("distance");
    }
    if vehicle.max_duration().is_some_and(|m| truth.duration > m) {
        v.push("duration");
    }
    v
}

#[test]
fn every_way_to_measure_a_route_agrees_with_the_definition_from_any_depot() {
    let mut r = Lcg(20261111);
    let (mut routes_seen, mut elsewhere, mut late, mut limited) = (0, 0, 0, 0);
    for case in 0..600 {
        let p = problem(&mut r);
        let dm = DistanceMatrix::from_customers(&p.customers);
        let depots = num_depots(&p.customers);
        let ids: Vec<usize> = (depots..p.customers.len()).collect();

        // Deal the customers onto the vehicles in a random order; a few are left unassigned.
        let mut order = ids.clone();
        for i in (1..order.len()).rev() {
            order.swap(i, r.below(i + 1));
        }
        let mut sequences: Vec<Vec<usize>> = vec![Vec::new(); p.vehicles.len()];
        let mut unassigned = Vec::new();
        for id in order {
            if r.next() < 0.1 {
                unassigned.push(id);
            } else {
                sequences[r.below(p.vehicles.len())].push(id);
            }
        }

        let mut plan = Solution::new();
        let mut predicted: Vec<&str> = Vec::new();
        for (v, seq) in sequences.iter().enumerate() {
            let vehicle = &p.vehicles[v];
            let depot = vehicle.depot_id();
            let truth = simulate(&p.customers, depot, seq);
            routes_seen += 1;
            elsewhere += (depot != 0) as usize;
            late += (!truth.hard_late.is_empty() || truth.lateness > 0.0) as usize;
            limited +=
                (vehicle.max_distance().is_some() || vehicle.max_duration().is_some()) as usize;
            let want = expected(&truth, vehicle);

            // The evaluator, with the vehicle that knows its depot.
            let (built, violations) =
                RouteEvaluator::new(&p.customers, &dm, vehicle).build_route(seq);
            assert!(
                close(built.total_distance(), truth.distance),
                "case {case}: distance"
            );
            assert!(
                close(built.total_duration(), truth.duration),
                "case {case}: duration"
            );
            assert!(
                close(built.total_lateness(), truth.lateness),
                "case {case}: lateness"
            );
            assert!(
                close(built.lateness_cost(), truth.lateness_cost),
                "case {case}: cost"
            );
            let mut got: Vec<&str> = violations.iter().map(name).collect();
            let mut want_sorted = want.clone();
            got.sort_unstable();
            want_sorted.sort_unstable();
            assert_eq!(
                got, want_sorted,
                "case {case}: violations of {seq:?} from {depot}"
            );

            // The searches' fast measures, with the depot named.
            let limits = RouteLimits::of(vehicle);
            let allowed = truth.hard_late.is_empty()
                && vehicle.max_distance().is_none_or(|m| truth.distance <= m)
                && vehicle.max_duration().is_none_or(|m| truth.duration <= m);
            assert_eq!(
                route_feasible(seq, depot, &dm, &p.customers, &limits),
                allowed,
                "case {case}: feasible"
            );
            match route_eval(seq, depot, &dm, &p.customers, &limits) {
                Some(e) => {
                    assert!(
                        allowed,
                        "case {case}: eval allowed a route the definition forbids"
                    );
                    assert!(close(e.distance, truth.distance));
                    assert!(close(e.duration, truth.duration));
                    assert!(close(e.lateness, truth.lateness));
                    assert!(close(e.lateness_cost, truth.lateness_cost));
                }
                None => assert!(!allowed, "case {case}: eval refused an allowed route"),
            }

            // A route that reports its own figures.
            let mut route = Route::new(v);
            for &id in seq {
                route.push_visit(Visit {
                    customer_id: id,
                    arrival_time: 0.0,
                    departure_time: 0.0,
                    load_after: 0,
                });
            }
            route.set_total_distance(truth.distance);
            route.set_lateness(truth.lateness, truth.lateness_cost);
            plan.add_route(route);
            predicted.extend(want);
        }
        for &u in &unassigned {
            plan.add_unassigned(u);
        }

        let counts: Vec<Option<usize>> = Vec::new();
        let fleet = Fleet {
            vehicles: &p.vehicles,
            counts: &counts,
            max_routes: None,
        };
        let mut got: Vec<&str> = check_solution(&p.customers, &dm, &fleet, &plan)
            .iter()
            .map(name)
            .collect();
        got.sort_unstable();
        predicted.sort_unstable();
        assert_eq!(got, predicted, "case {case}: the plan check");
    }
    // The battery reaches what it claims to.
    assert!(
        routes_seen > 1500 && elsewhere > 600,
        "routes {routes_seen} elsewhere {elsewhere}"
    );
    assert!(late > 200 && limited > 600, "late {late} limited {limited}");
}

#[test]
fn a_route_out_of_another_depot_is_another_route() {
    // One customer at x = 90 between depot 0 (x = 0) and depot 1 (x = 100),
    // its window open from 30 to 60 with 5 of service.
    let customers = vec![
        Customer::depot_at(0, 0.0, 0.0),
        Customer::depot_at(1, 100.0, 0.0),
        Customer::new(2, 90.0, 0.0, 5, 5.0)
            .with_time_window(TimeWindow::new(30.0, 60.0).expect("window")),
    ];
    let dm = DistanceMatrix::from_customers(&customers);
    let near = Vehicle::new(1, 10).with_depot(1);
    let far = Vehicle::new(0, 10);

    // From depot 1: 10 out, a wait of 20 for the window, 5 of service, 10 back.
    let (route, violations) = RouteEvaluator::new(&customers, &dm, &near).build_route(&[2]);
    assert!(violations.is_empty());
    assert_eq!(route.total_distance(), 20.0);
    assert_eq!(route.total_duration(), 45.0);

    // From depot 0: 90 out -- after the window has closed.
    let (route, violations) = RouteEvaluator::new(&customers, &dm, &far).build_route(&[2]);
    assert_eq!(route.total_distance(), 180.0);
    assert!(matches!(
        violations[..],
        [Violation {
            kind: ViolationType::TimeWindowViolated { customer_id: 2, .. }
        }]
    ));

    let limits = RouteLimits::NONE;
    assert!(route_feasible(&[2], 1, &dm, &customers, &limits));
    assert!(!route_feasible(&[2], 0, &dm, &customers, &limits));
}
