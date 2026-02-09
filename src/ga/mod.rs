//! Genetic algorithm components for vehicle routing.
//!
//! - [`GiantTour`] — Permutation chromosome encoding all customers
//! - [`split`] — Prins (2004) split DP to partition giant tour into routes
//! - [`RoutingGaProblem`] — [`GaProblem`](u_metaheur::ga::GaProblem) implementation

mod chromosome;
mod problem;
pub mod split;

pub use chromosome::GiantTour;
pub use problem::RoutingGaProblem;
pub use split::split;
