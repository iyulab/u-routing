//! Genetic algorithm components for vehicle routing.
//!
//! - [`GiantTour`] — Permutation chromosome encoding all customers
//! - [`split()`] — Prins (2004) split DP to partition giant tour into routes
//! - [`split_tw()`] — Time-window-aware split for VRPTW
//! - [`RoutingGaProblem`] — [`GaProblem`](u_metaheur::ga::GaProblem) implementation
//! - `educate` — local search on a giant tour (the hybrid GA's mutation)

mod chromosome;
mod educate;
mod problem;
pub mod split;
pub mod split_tw;

pub use chromosome::GiantTour;
pub use problem::RoutingGaProblem;
pub use split::split;
pub use split_tw::split_tw;
