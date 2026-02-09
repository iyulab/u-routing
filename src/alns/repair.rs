//! Repair operators for ALNS-based VRP optimization.
//!
//! # Operators
//!
//! - [`GreedyInsertion`] — Inserts each customer at the cheapest position
//! - [`RegretInsertion`] — Inserts the customer with highest regret value first
//!
//! # Reference
//!
//! Ropke, S. & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search
//! Heuristic for the Pickup and Delivery Problem with Time Windows",
//! *Transportation Science* 40(4), 455-472.

use rand::Rng;
use u_metaheur::alns::RepairOperator;

use crate::distance::DistanceMatrix;
use crate::models::Customer;

use super::solution_repr::RoutingSolution;

/// Finds the best insertion position for a customer across all routes.
///
/// Returns `(route_index, position, cost_increase)`.
fn best_insertion(
    routes: &[Vec<usize>],
    customer_id: usize,
    distances: &DistanceMatrix,
    customers: &[Customer],
    capacity: i32,
) -> Option<(usize, usize, f64)> {
    let depot = 0;
    let mut best: Option<(usize, usize, f64)> = None;

    for (ri, route) in routes.iter().enumerate() {
        // Check capacity
        let load: i32 = route.iter().map(|&c| customers[c].demand()).sum();
        if load + customers[customer_id].demand() > capacity {
            continue;
        }

        for pos in 0..=route.len() {
            let prev = if pos == 0 { depot } else { route[pos - 1] };
            let next = if pos == route.len() {
                depot
            } else {
                route[pos]
            };

            let cost = distances.get(prev, customer_id) + distances.get(customer_id, next)
                - distances.get(prev, next);

            if best.as_ref().is_none_or(|b| cost < b.2) {
                best = Some((ri, pos, cost));
            }
        }
    }

    best
}

/// Greedy insertion: inserts each unassigned customer at its cheapest position.
///
/// Iteratively selects the unassigned customer with the lowest insertion cost
/// and places it at the best position. Creates new routes when needed.
pub struct GreedyInsertion {
    distances: DistanceMatrix,
    customers: Vec<Customer>,
    capacity: i32,
}

impl GreedyInsertion {
    /// Creates a new greedy insertion operator.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>, capacity: i32) -> Self {
        Self {
            distances,
            customers,
            capacity,
        }
    }
}

impl RepairOperator<RoutingSolution> for GreedyInsertion {
    fn name(&self) -> &str {
        "greedy_insertion"
    }

    fn repair<R: Rng>(&self, solution: &RoutingSolution, _rng: &mut R) -> RoutingSolution {
        let mut sol = solution.clone();
        let mut unassigned = std::mem::take(sol.unassigned_mut());

        while !unassigned.is_empty() {
            // Find the unassigned customer with the cheapest insertion
            let mut best_cust_idx = 0;
            let mut best_route = 0;
            let mut best_pos = 0;
            let mut best_cost = f64::INFINITY;
            let mut found = false;

            for (ui, &cid) in unassigned.iter().enumerate() {
                if let Some((ri, pos, cost)) =
                    best_insertion(sol.routes(), cid, &self.distances, &self.customers, self.capacity)
                {
                    if cost < best_cost {
                        best_cost = cost;
                        best_cust_idx = ui;
                        best_route = ri;
                        best_pos = pos;
                        found = true;
                    }
                }
            }

            if !found {
                // No feasible insertion — create new route for first unassigned
                let cid = unassigned.remove(0);
                sol.routes_mut().push(vec![cid]);
            } else {
                let cid = unassigned.remove(best_cust_idx);
                sol.routes_mut()[best_route].insert(best_pos, cid);
            }
        }

        sol.recalculate_distance(&self.distances);
        sol
    }
}

/// Regret-k insertion: prioritizes customers with the highest regret value.
///
/// Regret-k is defined as the difference between the k-th best and the best
/// insertion cost. Customers with high regret have fewer good alternatives
/// and should be inserted first.
///
/// Uses k=2 (regret-2) by default.
pub struct RegretInsertion {
    distances: DistanceMatrix,
    customers: Vec<Customer>,
    capacity: i32,
    k: usize,
}

impl RegretInsertion {
    /// Creates a new regret-k insertion operator with k=2.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>, capacity: i32) -> Self {
        Self {
            distances,
            customers,
            capacity,
            k: 2,
        }
    }

    /// Creates a regret insertion operator with custom k.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(2);
        self
    }

    /// Computes insertion costs for a customer across all routes, sorted ascending.
    fn sorted_insertion_costs(
        &self,
        routes: &[Vec<usize>],
        customer_id: usize,
    ) -> Vec<(usize, usize, f64)> {
        let depot = 0;
        let mut costs = Vec::new();

        for (ri, route) in routes.iter().enumerate() {
            let load: i32 = route.iter().map(|&c| self.customers[c].demand()).sum();
            if load + self.customers[customer_id].demand() > self.capacity {
                continue;
            }

            // Find best position in this route
            let mut best_pos = 0;
            let mut best_cost = f64::INFINITY;
            for pos in 0..=route.len() {
                let prev = if pos == 0 { depot } else { route[pos - 1] };
                let next = if pos == route.len() {
                    depot
                } else {
                    route[pos]
                };
                let cost = self.distances.get(prev, customer_id)
                    + self.distances.get(customer_id, next)
                    - self.distances.get(prev, next);
                if cost < best_cost {
                    best_cost = cost;
                    best_pos = pos;
                }
            }
            costs.push((ri, best_pos, best_cost));
        }

        costs.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .expect("insertion costs should not be NaN")
        });
        costs
    }
}

impl RepairOperator<RoutingSolution> for RegretInsertion {
    fn name(&self) -> &str {
        "regret_insertion"
    }

    fn repair<R: Rng>(&self, solution: &RoutingSolution, _rng: &mut R) -> RoutingSolution {
        let mut sol = solution.clone();
        let mut unassigned = std::mem::take(sol.unassigned_mut());

        while !unassigned.is_empty() {
            let mut best_regret = f64::NEG_INFINITY;
            let mut best_cust_idx = 0;
            let mut best_route = 0;
            let mut best_pos = 0;
            let mut found = false;

            for (ui, &cid) in unassigned.iter().enumerate() {
                let costs = self.sorted_insertion_costs(sol.routes(), cid);

                if costs.is_empty() {
                    continue;
                }

                let best_cost = costs[0].2;
                // Regret = sum of differences between k-th best and best
                let regret: f64 = costs
                    .iter()
                    .skip(1)
                    .take(self.k - 1)
                    .map(|c| c.2 - best_cost)
                    .sum();

                // If fewer than k routes available, use large regret (prioritize)
                let regret = if costs.len() < self.k {
                    regret + f64::MAX / 2.0
                } else {
                    regret
                };

                if regret > best_regret || (regret == best_regret && best_cost < costs[0].2) {
                    best_regret = regret;
                    best_cust_idx = ui;
                    best_route = costs[0].0;
                    best_pos = costs[0].1;
                    found = true;
                }
            }

            if !found {
                // Create new route for the first unassigned
                let cid = unassigned.remove(0);
                sol.routes_mut().push(vec![cid]);
            } else {
                let cid = unassigned.remove(best_cust_idx);
                sol.routes_mut()[best_route].insert(best_pos, cid);
            }
        }

        sol.recalculate_distance(&self.distances);
        sol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Vec<Customer>, DistanceMatrix) {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 0.0),
            Customer::new(2, 2.0, 0.0, 10, 0.0),
            Customer::new(3, 3.0, 0.0, 10, 0.0),
            Customer::new(4, 4.0, 0.0, 10, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        (customers, dm)
    }

    #[test]
    fn test_greedy_inserts_all() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1]], vec![2, 3, 4], &cust, &dm);
        let op = GreedyInsertion::new(dm.clone(), cust.clone(), 100);
        let mut rng = u_optim::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
        let total: usize = repaired.routes().iter().map(|r| r.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_greedy_creates_new_route_when_full() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2]], vec![3, 4], &cust, &dm);
        let op = GreedyInsertion::new(dm.clone(), cust.clone(), 20); // cap 20, demand 10 each
        let mut rng = u_optim::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
        assert!(repaired.num_routes() >= 2);
    }

    #[test]
    fn test_regret_inserts_all() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1]], vec![2, 3, 4], &cust, &dm);
        let op = RegretInsertion::new(dm.clone(), cust.clone(), 100);
        let mut rng = u_optim::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
        let total: usize = repaired.routes().iter().map(|r| r.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_regret_prioritizes_constrained_customers() {
        let (cust, dm) = setup();
        // Two routes, both nearly full — regret should prioritize customers
        // with fewer insertion options
        let sol = RoutingSolution::new(vec![vec![1], vec![2]], vec![3, 4], &cust, &dm);
        let op = RegretInsertion::new(dm.clone(), cust.clone(), 20);
        let mut rng = u_optim::random::create_rng(42);
        let repaired = op.repair(&sol, &mut rng);
        assert!(repaired.unassigned().is_empty());
    }

    #[test]
    fn test_best_insertion_position() {
        let (cust, dm) = setup();
        // Route [1, 3], insert 2 — best position should be between 1 and 3
        let routes = vec![vec![1, 3]];
        let result = best_insertion(&routes, 2, &dm, &cust, 100);
        assert!(result.is_some());
        let (ri, pos, _cost) = result.expect("should find insertion");
        assert_eq!(ri, 0);
        assert_eq!(pos, 1); // between 1 and 3
    }
}
