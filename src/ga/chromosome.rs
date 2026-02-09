//! Giant tour chromosome for VRP genetic algorithms.
//!
//! A giant tour encodes a VRP solution as a single permutation of all
//! customer IDs (excluding depot). The [`split`](super::split) algorithm
//! converts this permutation into feasible routes.
//!
//! # Reference
//!
//! Prins, C. (2004). "A simple and effective evolutionary algorithm for the
//! vehicle routing problem", *Computers & Operations Research* 31(12), 1985-2002.

use u_metaheur::ga::Individual;

/// A giant tour: a permutation of customer IDs that encodes a VRP solution.
///
/// The fitness value represents the total distance after optimal splitting
/// into sub-routes. Lower fitness = better solution.
///
/// # Examples
///
/// ```
/// use u_routing::ga::GiantTour;
/// use u_metaheur::ga::Individual;
///
/// let tour = GiantTour::new(vec![3, 1, 2]);
/// assert_eq!(tour.customers(), &[3, 1, 2]);
/// assert_eq!(tour.fitness(), f64::INFINITY);
/// ```
#[derive(Debug, Clone)]
pub struct GiantTour {
    customers: Vec<usize>,
    fitness: f64,
}

impl GiantTour {
    /// Creates a new giant tour from a customer permutation.
    pub fn new(customers: Vec<usize>) -> Self {
        Self {
            customers,
            fitness: f64::INFINITY,
        }
    }

    /// Returns the customer permutation.
    pub fn customers(&self) -> &[usize] {
        &self.customers
    }

    /// Returns a mutable reference to the customer permutation.
    pub fn customers_mut(&mut self) -> &mut Vec<usize> {
        &mut self.customers
    }

    /// Returns the number of customers in this tour.
    pub fn len(&self) -> usize {
        self.customers.len()
    }

    /// Returns true if the tour has no customers.
    pub fn is_empty(&self) -> bool {
        self.customers.is_empty()
    }
}

impl Individual for GiantTour {
    type Fitness = f64;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_giant_tour_new() {
        let tour = GiantTour::new(vec![1, 2, 3]);
        assert_eq!(tour.customers(), &[1, 2, 3]);
        assert_eq!(tour.len(), 3);
        assert!(!tour.is_empty());
        assert_eq!(tour.fitness(), f64::INFINITY);
    }

    #[test]
    fn test_giant_tour_empty() {
        let tour = GiantTour::new(vec![]);
        assert!(tour.is_empty());
        assert_eq!(tour.len(), 0);
    }

    #[test]
    fn test_giant_tour_set_fitness() {
        let mut tour = GiantTour::new(vec![1, 2, 3]);
        tour.set_fitness(42.5);
        assert_eq!(tour.fitness(), 42.5);
    }

    #[test]
    fn test_giant_tour_clone() {
        let mut tour = GiantTour::new(vec![1, 2, 3]);
        tour.set_fitness(10.0);
        let cloned = tour.clone();
        assert_eq!(cloned.customers(), &[1, 2, 3]);
        assert_eq!(cloned.fitness(), 10.0);
    }

    #[test]
    fn test_giant_tour_mutate_customers() {
        let mut tour = GiantTour::new(vec![1, 2, 3]);
        tour.customers_mut().swap(0, 2);
        assert_eq!(tour.customers(), &[3, 2, 1]);
    }
}
