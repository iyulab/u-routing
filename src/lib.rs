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
//! - [`local_search`] — Local search operators (2-opt, Relocate)
//! - [`ga`] — Genetic algorithm with Prins split (giant tour encoding)

pub mod constructive;
pub mod distance;
pub mod evaluation;
pub mod ga;
pub mod local_search;
pub mod models;
