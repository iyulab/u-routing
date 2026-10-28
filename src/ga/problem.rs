//! GA problem definition for vehicle routing.
//!
//! Implements the [`GaProblem`](u_metaheur::ga::GaProblem) trait using giant
//! tour encoding with the Prins (2004) split algorithm for evaluation.
//!
//! # Operators
//!
//! - **Initial population**: the giant tours of the savings, sweep and (with
//!   time windows) Solomon I1 constructions, educated, then random tours
//! - **Crossover**: Order crossover (OX) — preserves relative customer ordering
//! - **Mutation**: Swap or invert, then education — local search on the split
//!   routes written back into the tour
//! - **Evaluation**: Split DP → local search (optional 2-opt) → total distance
//!
//! # Reference
//!
//! Prins, C. (2004). "A simple and effective evolutionary algorithm for the
//! vehicle routing problem", *Computers & Operations Research* 31(12), 1985-2002.

use rand::{Rng, RngExt};
use u_metaheur::ga::operators::{invert_mutation, order_crossover, swap_mutation};
use u_metaheur::ga::GaProblem;
use u_metaheur::Deadline;

use crate::constructive::{clarke_wright_savings, solomon_i1, sweep};
use crate::distance::DistanceMatrix;
use crate::evaluation::RouteLimits;
use crate::local_search::{minimize_routes, two_opt_improve};
use crate::models::{Customer, Solution, Vehicle};

use super::chromosome::GiantTour;
use super::educate::{concatenate, educate, route_vehicle};
use super::split::split;
use super::split_tw::split_tw;

/// GA problem for capacitated vehicle routing, with time windows where
/// customers carry them.
///
/// Uses giant tour encoding: each individual is a permutation of customer IDs.
/// Fitness is evaluated by splitting the permutation into feasible routes using
/// the Prins (2004) split DP algorithm, optionally followed by intra-route 2-opt.
///
/// When any customer has a time window, the split is [`split_tw`], and every
/// local-search move -- the 2-opt of the evaluation and the education of the
/// initial tours and mutations -- is taken only if it keeps every customer on
/// time and the route within its limits. A tour whose
/// split leaves customers unserved is ranked behind every tour that serves
/// them all: the split's partial cost would otherwise make dropping a customer
/// look cheaper than serving it.
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
/// use u_routing::ga::RoutingGaProblem;
/// use u_metaheur::ga::{GaProblem, GaConfig, GaRunner};
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 1.0, 0.0, 10, 0.0),
///     Customer::new(2, 2.0, 0.0, 10, 0.0),
///     Customer::new(3, 3.0, 0.0, 10, 0.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
///
/// let problem = RoutingGaProblem::new(customers.clone(), dm, 30);
/// let config = GaConfig::default()
///     .with_population_size(20)
///     .with_max_generations(50);
///
/// let result = GaRunner::run(&problem, &config).unwrap();
/// assert!(result.best_fitness < f64::INFINITY);
/// ```
pub struct RoutingGaProblem {
    customers: Vec<Customer>,
    distances: DistanceMatrix,
    capacity: i32,
    /// The vehicle each route is planned with -- the capacity and the limits.
    vehicle: Vehicle,
    apply_local_search: bool,
    /// Whether any customer carries a time window.
    time_windows: bool,
    /// The vehicle's distance and duration limits on each route.
    limits: RouteLimits,
    /// Cost of each route in distance units (see [`Self::with_route_cost`]).
    route_cost: f64,
    /// Added per unserved customer: more than any complete solution costs,
    /// which is at most a separate round trip to the farthest pair per customer.
    unserved_penalty: f64,
}

impl RoutingGaProblem {
    /// Creates a new routing GA problem.
    ///
    /// # Arguments
    ///
    /// * `customers` — All locations (index 0 = depot)
    /// * `distances` — Distance matrix
    /// * `capacity` — Vehicle capacity
    pub fn new(customers: Vec<Customer>, distances: DistanceMatrix, capacity: i32) -> Self {
        let n = customers.len();
        let time_windows = customers.iter().any(|c| c.time_window().is_some());
        let farthest = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| distances.get(i, j))
            .fold(0.0, f64::max);
        Self {
            customers,
            distances,
            capacity,
            vehicle: route_vehicle(capacity, &RouteLimits::NONE),
            apply_local_search: true,
            time_windows,
            limits: RouteLimits::NONE,
            route_cost: 0.0,
            unserved_penalty: 2.0 * farthest * n as f64 + 1.0,
        }
    }

    /// Adds `route_cost` (distance units — a vehicle's
    /// [`fixed_cost_in_distance`](crate::models::Vehicle::fixed_cost_in_distance))
    /// for every route: the fitness is total distance plus this per route, so
    /// a positive value trades distance for fewer routes. 0 by default.
    pub fn with_route_cost(mut self, route_cost: f64) -> Self {
        self.route_cost = route_cost;
        self.vehicle = self.vehicle.with_fixed_cost(route_cost);
        self
    }

    /// Plans every route within these distance and duration limits (none by
    /// default): the split only cuts routes that keep them, and 2-opt only
    /// takes moves that keep them.
    pub fn with_limits(mut self, limits: RouteLimits) -> Self {
        self.limits = limits;
        self.vehicle = route_vehicle(self.capacity, &limits).with_fixed_cost(self.route_cost);
        self
    }

    /// Disables intra-route 2-opt local search during evaluation.
    pub fn without_local_search(mut self) -> Self {
        self.apply_local_search = false;
        self
    }

    /// Returns the number of customers (excluding depot).
    fn num_customers(&self) -> usize {
        self.customers.len() - 1
    }

    /// The giant tour of a constructed solution.
    fn tour_of(&self, solution: &Solution) -> Vec<usize> {
        let routes: Vec<Vec<usize>> = solution.routes().iter().map(|r| r.customer_ids()).collect();
        concatenate(&routes, self.num_customers())
    }
}

/// Insertions the ejection pool may spend on each route it tries to remove
/// from a seed.
const SEED_EJECTION_BUDGET: usize = 200;

/// Seed of the random stream the ejection pool perturbs the seeds with.
const SEED_PERTURBATION_STREAM: u64 = 0x5eed;

impl GaProblem for RoutingGaProblem {
    type Individual = GiantTour;

    fn create_individual<R: Rng>(&self, rng: &mut R) -> GiantTour {
        let n = self.num_customers();
        let mut perm: Vec<usize> = (1..=n).collect();

        // Fisher-Yates shuffle
        for i in (1..perm.len()).rev() {
            let j = rng.random_range(0..=i as u64) as usize;
            perm.swap(i, j);
        }

        GiantTour::new(perm)
    }

    /// The savings and sweep constructions -- and Solomon's I1 insertion when
    /// customers carry time windows -- each educated: the population starts
    /// from good routes as well as from random ones (Prins 2004 seeds with
    /// constructive heuristics in the same way). The route minimisation of a
    /// seed stops at `deadline` with the routes it has.
    fn initial_individuals(&self, deadline: &Deadline) -> Vec<GiantTour> {
        if self.num_customers() == 0 {
            return Vec::new();
        }
        let mut seeds = vec![
            clarke_wright_savings(&self.customers, &self.distances, &self.vehicle),
            sweep(&self.customers, &self.distances, &self.vehicle),
        ];
        if self.time_windows {
            seeds.push(solomon_i1(&self.customers, &self.distances, &self.vehicle));
        }
        // The ejection pool's perturbation draws from its own stream: the
        // seeds stay the same constructions whatever the run's seed.
        let mut rng = u_numflow::random::create_rng(SEED_PERTURBATION_STREAM);
        seeds
            .iter()
            .map(|s| {
                let mut tour = self.tour_of(s);
                if self.route_cost > 0.0 {
                    // With routes priced, the seeds start from as few routes as
                    // the ejection pool reaches: the GA's own moves only remove
                    // a route whose customers fit elsewhere directly.
                    let routes: Vec<Vec<usize>> =
                        s.routes().iter().map(|r| r.customer_ids()).collect();
                    let routes = minimize_routes(
                        routes,
                        &self.customers,
                        &self.distances,
                        &self.vehicle,
                        SEED_EJECTION_BUDGET,
                        &mut rng,
                        deadline,
                    );
                    tour = concatenate(&routes, self.num_customers());
                }
                GiantTour::new(educate(
                    &tour,
                    &self.customers,
                    &self.distances,
                    &self.vehicle,
                ))
            })
            .collect()
    }

    fn evaluate(&self, individual: &GiantTour) -> f64 {
        // The constrained split cuts only routes that keep the windows and
        // limits; the plain one needs neither check.
        let result = if self.time_windows || !self.limits.is_unlimited() {
            split_tw(
                individual.customers(),
                &self.customers,
                &self.distances,
                self.capacity,
                &self.limits,
                self.route_cost,
            )
        } else {
            split(
                individual.customers(),
                &self.customers,
                &self.distances,
                self.capacity,
                self.route_cost,
            )
        };
        let routes_cost = self.route_cost * result.routes.len() as f64;

        let served: usize = result.routes.iter().map(Vec::len).sum();
        let unserved = self.num_customers().saturating_sub(served);
        if unserved > 0 {
            return result.total_distance + routes_cost + self.unserved_penalty * unserved as f64;
        }
        if !self.apply_local_search {
            return result.total_distance + routes_cost;
        }

        // Apply 2-opt to each route. With time windows, 2-opt only takes
        // reversals that keep every customer on time.
        let mut total = 0.0;
        for route in &result.routes {
            let (_, dist) =
                two_opt_improve(route, 0, &self.distances, &self.customers, &self.limits);
            total += dist;
        }
        total + routes_cost
    }

    fn crossover<R: Rng>(
        &self,
        parent1: &GiantTour,
        parent2: &GiantTour,
        rng: &mut R,
    ) -> Vec<GiantTour> {
        // OX expects 0-indexed permutation (0..n), convert customer IDs (1..=n)
        let p1: Vec<usize> = parent1.customers().iter().map(|&c| c - 1).collect();
        let p2: Vec<usize> = parent2.customers().iter().map(|&c| c - 1).collect();
        let (c1, c2) = order_crossover(&p1, &p2, rng);
        vec![
            GiantTour::new(c1.into_iter().map(|c| c + 1).collect()),
            GiantTour::new(c2.into_iter().map(|c| c + 1).collect()),
        ]
    }

    fn mutate<R: Rng>(&self, individual: &mut GiantTour, rng: &mut R) {
        if individual.len() < 2 {
            return;
        }
        // 50% swap, 50% invert (2-opt style), then local search on the
        // routes the tour splits into, written back into the tour.
        if rng.random_range(0..2u64) == 0 {
            swap_mutation(individual.customers_mut(), rng);
        } else {
            invert_mutation(individual.customers_mut(), rng);
        }
        let educated = educate(
            individual.customers(),
            &self.customers,
            &self.distances,
            &self.vehicle,
        );
        *individual = GiantTour::new(educated);
    }
}

// RoutingGaProblem contains only owned data, safe to share across threads
unsafe impl Send for RoutingGaProblem {}
unsafe impl Sync for RoutingGaProblem {}

#[cfg(test)]
mod tests {
    use super::*;
    use u_metaheur::ga::{GaConfig, GaRunner};

    fn setup() -> (Vec<Customer>, DistanceMatrix) {
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
    fn test_create_individual() {
        let (cust, dm) = setup();
        let problem = RoutingGaProblem::new(cust, dm, 30);
        let mut rng = u_numflow::random::create_rng(42);
        let ind = problem.create_individual(&mut rng);
        assert_eq!(ind.len(), 3);
        let mut sorted = ind.customers().to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn test_evaluate_optimal_tour() {
        let (cust, dm) = setup();
        let problem = RoutingGaProblem::new(cust, dm, 30).without_local_search();
        let tour = GiantTour::new(vec![1, 2, 3]);
        let fitness = problem.evaluate(&tour);
        // 0→1→2→3→0 = 6.0
        assert!((fitness - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_with_local_search() {
        let (cust, dm) = setup();
        let problem = RoutingGaProblem::new(cust, dm, 30);
        let tour = GiantTour::new(vec![3, 1, 2]);
        let fitness = problem.evaluate(&tour);
        // After split and 2-opt, should find optimal or near-optimal
        assert!(fitness <= 6.0 + 1e-10);
    }

    #[test]
    fn test_crossover_preserves_genes() {
        let (cust, dm) = setup();
        let problem = RoutingGaProblem::new(cust, dm, 30);
        let p1 = GiantTour::new(vec![1, 2, 3]);
        let p2 = GiantTour::new(vec![3, 1, 2]);
        let mut rng = u_numflow::random::create_rng(42);
        let children = problem.crossover(&p1, &p2, &mut rng);
        assert_eq!(children.len(), 2);
        for child in &children {
            assert_eq!(child.len(), 3);
            let mut sorted = child.customers().to_vec();
            sorted.sort();
            assert_eq!(sorted, vec![1, 2, 3]);
        }
    }

    #[test]
    fn test_mutate_preserves_genes() {
        let (cust, dm) = setup();
        let problem = RoutingGaProblem::new(cust, dm, 30);
        let mut tour = GiantTour::new(vec![1, 2, 3]);
        let mut rng = u_numflow::random::create_rng(42);
        problem.mutate(&mut tour, &mut rng);
        let mut sorted = tour.customers().to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn test_ga_runner_finds_solution() {
        let (cust, dm) = setup();
        let problem = RoutingGaProblem::new(cust, dm, 30);
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(30);

        let result = GaRunner::run(&problem, &config).unwrap();
        assert!(!result.best.is_empty());
        // Should find optimal: 6.0
        assert!(result.best_fitness <= 6.0 + 1e-10);
    }

    #[test]
    fn test_ga_runner_capacity_constrained() {
        let customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 15, 0.0),
            Customer::new(2, 2.0, 0.0, 15, 0.0),
            Customer::new(3, 3.0, 0.0, 15, 0.0),
        ];
        let dm = DistanceMatrix::from_customers(&customers);
        let problem = RoutingGaProblem::new(customers, dm, 25);
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(30);

        let result = GaRunner::run(&problem, &config).unwrap();
        assert!(result.best_fitness < f64::INFINITY);
        // Must split into at least 2 routes
    }
}

#[cfg(test)]
mod time_window_tests {
    use super::*;
    use crate::models::TimeWindow;

    /// Customer 1's service lasts until t = 6 and customer 2 closes at t = 3,
    /// so they cannot share a route; customer 3 is too far to reach by t = 5.
    fn problem(with_far_customer: bool) -> RoutingGaProblem {
        let mut customers = vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 1.0, 0.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 2.0).expect("valid")),
            Customer::new(2, 0.0, 1.0, 10, 5.0)
                .with_time_window(TimeWindow::new(0.0, 3.0).expect("valid")),
        ];
        if with_far_customer {
            customers.push(
                Customer::new(3, 10.0, 0.0, 10, 0.0)
                    .with_time_window(TimeWindow::new(0.0, 5.0).expect("valid")),
            );
        }
        let dm = DistanceMatrix::from_customers(&customers);
        RoutingGaProblem::new(customers, dm, 100)
    }

    #[test]
    fn a_tour_is_split_where_the_windows_require() {
        // Two round trips of 2, not one route of 1 + sqrt(2) + 1.
        let cost = problem(false).evaluate(&GiantTour::new(vec![1, 2]));
        assert!((cost - 4.0).abs() < 1e-9, "{cost}");
    }

    #[test]
    fn a_customer_out_of_reach_costs_the_same_wherever_it_sits_in_the_tour() {
        let p = problem(true);
        // Customer 3 cannot be served from anywhere in the tour. It used to cut
        // the tour where it stood -- [3, 1, 2] served nobody while [1, 2, 3]
        // served two, so the search ranked tours by where an unreachable
        // customer happened to fall instead of by what they serve.
        let last = p.evaluate(&GiantTour::new(vec![1, 2, 3]));
        let first = p.evaluate(&GiantTour::new(vec![3, 1, 2]));
        assert!((last - first).abs() < 1e-9, "{last} vs {first}");
        // ... and it still costs more than the plan that serves everyone.
        let reachable = problem(false).evaluate(&GiantTour::new(vec![1, 2]));
        assert!(last > reachable, "{last} vs {reachable}");
    }
}
