//! Dense distance matrix.

use crate::models::Customer;

/// A dense n×n distance matrix stored in row-major order.
///
/// Supports both Euclidean distance computation from customer coordinates
/// and explicit distance specification.
///
/// # Examples
///
/// ```
/// use u_routing::models::Customer;
/// use u_routing::distance::DistanceMatrix;
///
/// let customers = vec![
///     Customer::depot(0.0, 0.0),
///     Customer::new(1, 3.0, 4.0, 10, 5.0),
///     Customer::new(2, 6.0, 8.0, 20, 5.0),
/// ];
/// let dm = DistanceMatrix::from_customers(&customers);
/// assert!((dm.get(0, 1) - 5.0).abs() < 1e-10);
/// assert_eq!(dm.size(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    data: Vec<f64>,
    size: usize,
}

impl DistanceMatrix {
    /// Creates a distance matrix of the given size, initialized to zero.
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size * size],
            size,
        }
    }

    /// Computes a Euclidean distance matrix from customer coordinates.
    pub fn from_customers(customers: &[Customer]) -> Self {
        let n = customers.len();
        let mut dm = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = customers[i].distance_to(&customers[j]);
                dm.set(i, j, d);
                dm.set(j, i, d);
            }
        }
        dm
    }

    /// Creates a distance matrix from an explicit n×n grid.
    ///
    /// Returns `None` if the data length doesn't match `size * size`.
    pub fn from_data(size: usize, data: Vec<f64>) -> Option<Self> {
        if data.len() != size * size {
            return None;
        }
        Some(Self { data, size })
    }

    /// Returns the distance from location `from` to location `to`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn get(&self, from: usize, to: usize) -> f64 {
        self.data[from * self.size + to]
    }

    /// Sets the distance from location `from` to location `to`.
    pub fn set(&mut self, from: usize, to: usize, distance: f64) {
        self.data[from * self.size + to] = distance;
    }

    /// Number of locations in this matrix.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns `true` if the matrix is symmetric within the given tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the nearest neighbor of `from` among the given candidates.
    ///
    /// Returns `None` if `candidates` is empty.
    pub fn nearest_neighbor(&self, from: usize, candidates: &[usize]) -> Option<usize> {
        candidates
            .iter()
            .copied()
            .min_by(|&a, &b| {
                self.get(from, a)
                    .partial_cmp(&self.get(from, b))
                    .expect("distance should not be NaN")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_customers() -> Vec<Customer> {
        vec![
            Customer::depot(0.0, 0.0),
            Customer::new(1, 3.0, 4.0, 10, 5.0),
            Customer::new(2, 0.0, 8.0, 20, 5.0),
        ]
    }

    #[test]
    fn test_from_customers() {
        let dm = DistanceMatrix::from_customers(&sample_customers());
        assert_eq!(dm.size(), 3);
        assert!((dm.get(0, 1) - 5.0).abs() < 1e-10);
        assert!((dm.get(0, 2) - 8.0).abs() < 1e-10);
        assert!((dm.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric() {
        let dm = DistanceMatrix::from_customers(&sample_customers());
        assert!(dm.is_symmetric(1e-10));
    }

    #[test]
    fn test_from_data() {
        let dm = DistanceMatrix::from_data(2, vec![0.0, 5.0, 5.0, 0.0]).expect("valid");
        assert_eq!(dm.get(0, 1), 5.0);
        assert_eq!(dm.get(1, 0), 5.0);
    }

    #[test]
    fn test_from_data_invalid_size() {
        assert!(DistanceMatrix::from_data(2, vec![0.0, 1.0, 2.0]).is_none());
    }

    #[test]
    fn test_set_get() {
        let mut dm = DistanceMatrix::new(3);
        dm.set(0, 1, 42.0);
        assert_eq!(dm.get(0, 1), 42.0);
        assert_eq!(dm.get(1, 0), 0.0);
    }

    #[test]
    fn test_nearest_neighbor() {
        let dm = DistanceMatrix::from_customers(&sample_customers());
        // From depot (0,0): customer 1 at (3,4) is dist 5, customer 2 at (0,8) is dist 8
        assert_eq!(dm.nearest_neighbor(0, &[1, 2]), Some(1));
        assert_eq!(dm.nearest_neighbor(0, &[2]), Some(2));
        assert_eq!(dm.nearest_neighbor(0, &[]), None);
    }

    #[test]
    fn test_asymmetric_matrix() {
        let mut dm = DistanceMatrix::new(2);
        dm.set(0, 1, 10.0);
        dm.set(1, 0, 15.0);
        assert!(!dm.is_symmetric(1e-10));
    }
}
