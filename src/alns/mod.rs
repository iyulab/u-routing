//! ALNS (Adaptive Large Neighborhood Search) components for vehicle routing.
//!
//! - [`RoutingSolution`] — Lightweight solution representation for ALNS
//! - [`RoutingAlnsProblem`] — [`AlnsProblem`](u_metaheur::alns::AlnsProblem) implementation
//! - [`destroy`] — Destroy operators (random, worst, Shaw)
//! - [`repair`] — Repair operators (greedy insertion, regret insertion)

pub mod destroy;
mod problem;
pub mod repair;
mod solution_repr;

pub use problem::RoutingAlnsProblem;
pub use solution_repr::RoutingSolution;
