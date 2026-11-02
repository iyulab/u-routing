//! Every method, on the problems where geometry, time or load leaves nothing
//! to choose between: every customer on the depot, on one point, on a line,
//! with no demand, with zero-width or identical time windows, one customer.
//! The answer is a plan or a refusal -- never `internal` (a panic, or a
//! non-finite number the test build's `finite_out` guard rejects).

use super::request_keys::call;
use serde_json::{json, Value};

const METHODS: [&str; 4] = ["nn", "savings", "ga", "alns"];

fn customer(id: i64, x: f64, y: f64, demand: f64, window: [f64; 2]) -> Value {
    json!({ "id": id, "x": x, "y": y, "demand": demand, "service_time": 0.0, "time_window": window })
}

fn problems() -> Vec<(&'static str, Vec<Value>, Value)> {
    let open = [0.0, 1000.0];
    let vehicle = json!([{ "capacity": 10.0 }]);
    let fleet = json!([{ "capacity": 2.0, "count": 2 }, { "capacity": 10.0 }]);
    let ids = 1..=6;
    vec![
        (
            "on the depot",
            ids.clone()
                .map(|i| customer(i, 0.0, 0.0, 1.0, open))
                .collect(),
            vehicle.clone(),
        ),
        (
            "on one point",
            ids.clone()
                .map(|i| customer(i, 5.0, 5.0, 1.0, open))
                .collect(),
            vehicle.clone(),
        ),
        (
            "on a line through the depot",
            ids.clone()
                .map(|i| customer(i, i as f64, 0.0, 1.0, open))
                .collect(),
            vehicle.clone(),
        ),
        (
            "no demand",
            ids.clone()
                .map(|i| customer(i, i as f64, (i * i) as f64, 0.0, open))
                .collect(),
            vehicle.clone(),
        ),
        (
            "zero-width windows at 0",
            ids.clone()
                .map(|i| customer(i, i as f64, 1.0, 1.0, [0.0, 0.0]))
                .collect(),
            vehicle.clone(),
        ),
        (
            "every customer exactly fills a vehicle",
            ids.clone()
                .map(|i| customer(i, i as f64, 2.0, 10.0, open))
                .collect(),
            vehicle.clone(),
        ),
        (
            "one customer",
            vec![customer(1, 3.0, 4.0, 1.0, open)],
            vehicle.clone(),
        ),
        (
            "on the depot, mixed fleet",
            ids.map(|i| customer(i, 0.0, 0.0, 1.0, open)).collect(),
            fleet,
        ),
    ]
}

#[test]
fn every_method_answers_degenerate_problems_without_internal() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for method in METHODS {
        for (what, customers, vehicles) in problems() {
            let request = json!({
                "customers": customers,
                "vehicles": vehicles,
                "depot_x": 0.0,
                "depot_y": 0.0,
                "method": method,
            });
            let (_, body) = call(super::urouting_solve_vrp, &request);
            if body.get("code").and_then(Value::as_str) == Some("internal") {
                failures.push(format!("{method} / {what}: {body}"));
                continue;
            }
            // A plan comes back checkable: the checker accepts what the solver
            // wrote.
            if body.get("routes").is_some() {
                let mut solution = serde_json::Map::new();
                for key in [
                    "routes",
                    "route_vehicles",
                    "unassigned",
                    "total_distance",
                    "total_cost",
                ] {
                    if let Some(v) = body.get(key) {
                        solution.insert(key.to_string(), v.clone());
                    }
                }
                let check = json!({ "problem": request, "solution": solution });
                let (_, verdict) = call(super::urouting_check_solution, &check);
                checked += 1;
                if verdict.get("feasible") != Some(&Value::Bool(true)) {
                    failures.push(format!("{method} / {what}: check {verdict}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // Every method plans every problem: a customer it cannot reach is
    // `unassigned`, not a refusal.
    assert_eq!(checked, METHODS.len() * problems().len());
}
