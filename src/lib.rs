//! # u-routing
//!
//! Vehicle routing optimization library providing models, heuristics, and
//! metaheuristic bridges for TSP, CVRP, and VRPTW variants.
//!
//! ## Modules
//!
//! - [`models`] — Domain model types (Customer, Vehicle, Route, Solution, Problem trait)
//! - [`distance`] — Distance and travel time matrix
//! - [`evaluation`] — Route feasibility checking and cost evaluation
//! - [`constructive`] — Constructive heuristics (Nearest Neighbor, Clarke-Wright)

pub mod constructive;
pub mod distance;
pub mod evaluation;
pub mod models;
