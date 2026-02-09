//! Destroy operators for ALNS-based VRP optimization.
//!
//! # Operators
//!
//! - [`RandomRemoval`] — Removes random customers
//! - [`WorstRemoval`] — Removes customers with highest removal cost savings
//! - [`ShawRemoval`] — Removes related (nearby) customers
//!
//! # Reference
//!
//! Ropke, S. & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search
//! Heuristic for the Pickup and Delivery Problem with Time Windows",
//! *Transportation Science* 40(4), 455-472.

use rand::Rng;
use u_metaheur::alns::DestroyOperator;

use crate::distance::DistanceMatrix;
use crate::models::Customer;

use super::solution_repr::RoutingSolution;

/// Removes random customers from the solution.
///
/// Simple but effective baseline operator that ensures diversity in the
/// search process.
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::alns::{RoutingSolution, destroy::RandomRemoval};
/// use u_metaheur::alns::DestroyOperator;
///
/// let cust = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&cust);
/// let sol = RoutingSolution::new(vec![vec![1, 2]], vec![], &cust, &dm);
///
/// let mut rng = u_numflow::random::create_rng(42);
/// let destroyed = RandomRemoval.destroy(&sol, 0.5, &mut rng);
/// assert!(!destroyed.unassigned().is_empty());
/// ```
pub struct RandomRemoval;

impl DestroyOperator<RoutingSolution> for RandomRemoval {
    fn name(&self) -> &str {
        "random_removal"
    }

    fn destroy<R: Rng>(
        &self,
        solution: &RoutingSolution,
        degree: f64,
        rng: &mut R,
    ) -> RoutingSolution {
        let mut sol = solution.clone();
        let total_customers: usize = sol.routes().iter().map(|r| r.len()).sum();
        let num_remove = ((total_customers as f64 * degree).round() as usize).max(1);

        for _ in 0..num_remove {
            let assigned: usize = sol.routes().iter().map(|r| r.len()).sum();
            if assigned == 0 {
                break;
            }

            // Pick random assigned customer
            let target = rng.random_range(0..assigned as u64) as usize;
            let mut count = 0;
            let mut removed = false;
            for route in sol.routes_mut() {
                if count + route.len() > target {
                    let pos = target - count;
                    let cid = route.remove(pos);
                    sol.unassigned_mut().push(cid);
                    removed = true;
                    break;
                }
                count += route.len();
            }
            if !removed {
                break;
            }
        }

        sol.remove_empty_routes();
        sol
    }
}

/// Removes customers with highest distance cost (most expensive to serve).
///
/// Identifies customers whose removal yields the largest cost savings,
/// i.e., the "worst-positioned" customers.
pub struct WorstRemoval {
    distances: DistanceMatrix,
}

impl WorstRemoval {
    /// Creates a new worst removal operator.
    pub fn new(distances: DistanceMatrix) -> Self {
        Self { distances }
    }

    /// Computes the cost saving from removing a customer at a given position.
    fn removal_saving(&self, route: &[usize], pos: usize) -> f64 {
        let depot = 0;
        let cid = route[pos];
        let prev = if pos == 0 { depot } else { route[pos - 1] };
        let next = if pos == route.len() - 1 {
            depot
        } else {
            route[pos + 1]
        };

        // Old: prev → cid → next, New: prev → next
        // Saving = old - new (positive means removing saves distance)
        self.distances.get(prev, cid) + self.distances.get(cid, next)
            - self.distances.get(prev, next)
    }
}

impl DestroyOperator<RoutingSolution> for WorstRemoval {
    fn name(&self) -> &str {
        "worst_removal"
    }

    fn destroy<R: Rng>(
        &self,
        solution: &RoutingSolution,
        degree: f64,
        rng: &mut R,
    ) -> RoutingSolution {
        let mut sol = solution.clone();
        let total_customers: usize = sol.routes().iter().map(|r| r.len()).sum();
        let num_remove = ((total_customers as f64 * degree).round() as usize).max(1);

        for _ in 0..num_remove {
            // Find customer with highest removal saving
            let mut best_saving = f64::NEG_INFINITY;
            let mut best_route = 0;
            let mut best_pos = 0;

            for (ri, route) in sol.routes().iter().enumerate() {
                for pos in 0..route.len() {
                    let saving = self.removal_saving(route, pos);
                    // Add small randomness to break ties
                    let noise = rng.random_range(0.0..0.01f64);
                    if saving + noise > best_saving {
                        best_saving = saving + noise;
                        best_route = ri;
                        best_pos = pos;
                    }
                }
            }

            if best_saving == f64::NEG_INFINITY {
                break;
            }

            let cid = sol.routes_mut()[best_route].remove(best_pos);
            sol.unassigned_mut().push(cid);
        }

        sol.remove_empty_routes();
        sol
    }
}

/// Removes related (nearby) customers using Shaw (1998) relatedness measure.
///
/// Starts from a random customer and iteratively removes the most
/// related (closest) unremoved customer to those already removed.
///
/// # Reference
///
/// Shaw, P. (1998). "Using Constraint Programming and Local Search Methods
/// to Solve Vehicle Routing Problems", *CP-98*, LNCS 1520, 417-431.
pub struct ShawRemoval {
    distances: DistanceMatrix,
    customers: Vec<Customer>,
}

impl ShawRemoval {
    /// Creates a new Shaw removal operator.
    pub fn new(distances: DistanceMatrix, customers: Vec<Customer>) -> Self {
        Self {
            distances,
            customers,
        }
    }

    /// Relatedness: inverse distance + demand similarity.
    fn relatedness(&self, a: usize, b: usize) -> f64 {
        let dist = self.distances.get(a, b);
        let demand_diff = (self.customers[a].demand() - self.customers[b].demand()).abs() as f64;
        // Higher relatedness = more similar
        1.0 / (dist + 0.1) + 1.0 / (demand_diff + 1.0)
    }
}

impl DestroyOperator<RoutingSolution> for ShawRemoval {
    fn name(&self) -> &str {
        "shaw_removal"
    }

    fn destroy<R: Rng>(
        &self,
        solution: &RoutingSolution,
        degree: f64,
        rng: &mut R,
    ) -> RoutingSolution {
        let mut sol = solution.clone();
        let total_customers: usize = sol.routes().iter().map(|r| r.len()).sum();
        let num_remove = ((total_customers as f64 * degree).round() as usize).max(1);

        if total_customers == 0 {
            return sol;
        }

        // Collect all assigned customers
        let mut assigned: Vec<usize> = sol
            .routes()
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();

        // Pick random seed customer
        let seed_idx = rng.random_range(0..assigned.len() as u64) as usize;
        let seed = assigned.remove(seed_idx);

        let mut removed = vec![seed];

        // Remove from solution
        remove_customer(&mut sol, seed);

        for _ in 1..num_remove {
            if assigned.is_empty() {
                break;
            }

            // Find most related unremoved customer to any removed customer
            let mut best_relatedness = f64::NEG_INFINITY;
            let mut best_idx = 0;

            for (idx, &cid) in assigned.iter().enumerate() {
                let max_rel = removed
                    .iter()
                    .map(|&r| self.relatedness(r, cid))
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_rel > best_relatedness {
                    best_relatedness = max_rel;
                    best_idx = idx;
                }
            }

            let next = assigned.remove(best_idx);
            removed.push(next);
            remove_customer(&mut sol, next);
        }

        sol.unassigned_mut().extend(&removed);
        sol.remove_empty_routes();
        sol
    }
}

/// Removes a customer from the solution's routes.
fn remove_customer(sol: &mut RoutingSolution, customer_id: usize) {
    for route in sol.routes_mut() {
        if let Some(pos) = route.iter().position(|&c| c == customer_id) {
            route.remove(pos);
            return;
        }
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
    fn test_random_removal() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2, 3, 4]], vec![], &cust, &dm);
        let op = RandomRemoval;
        let mut rng = u_numflow::random::create_rng(42);
        let destroyed = op.destroy(&sol, 0.5, &mut rng);
        // Should remove ~2 customers
        let assigned: usize = destroyed.routes().iter().map(|r| r.len()).sum();
        assert_eq!(assigned + destroyed.unassigned().len(), 4);
        assert!(!destroyed.unassigned().is_empty());
    }

    #[test]
    fn test_worst_removal() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2, 3, 4]], vec![], &cust, &dm);
        let op = WorstRemoval::new(dm.clone());
        let mut rng = u_numflow::random::create_rng(42);
        let destroyed = op.destroy(&sol, 0.25, &mut rng);
        // Should remove 1 customer (the worst-positioned)
        assert_eq!(destroyed.unassigned().len(), 1);
    }

    #[test]
    fn test_shaw_removal() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2, 3, 4]], vec![], &cust, &dm);
        let op = ShawRemoval::new(dm.clone(), cust.clone());
        let mut rng = u_numflow::random::create_rng(42);
        let destroyed = op.destroy(&sol, 0.5, &mut rng);
        // Should remove ~2 related customers
        assert_eq!(destroyed.unassigned().len(), 2);
        // Removed customers should be close to each other (on a line, consecutive IDs)
        let removed = destroyed.unassigned();
        assert!((removed[0] as i32 - removed[1] as i32).unsigned_abs() <= 2);
    }

    #[test]
    fn test_removal_preserves_all_customers() {
        let (cust, dm) = setup();
        let sol = RoutingSolution::new(vec![vec![1, 2], vec![3, 4]], vec![], &cust, &dm);
        let op = RandomRemoval;
        let mut rng = u_numflow::random::create_rng(42);
        let destroyed = op.destroy(&sol, 0.5, &mut rng);
        let mut all: Vec<usize> = destroyed
            .routes()
            .iter()
            .flat_map(|r| r.iter().copied())
            .chain(destroyed.unassigned().iter().copied())
            .collect();
        all.sort();
        assert_eq!(all, vec![1, 2, 3, 4]);
    }
}
