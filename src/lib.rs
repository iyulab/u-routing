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
//! - [`alns`] — ALNS with destroy/repair operators

pub mod alns;
pub mod constructive;
pub mod distance;
pub mod evaluation;
pub mod ga;
pub mod local_search;
pub mod models;

#[cfg(any(feature = "wasm", feature = "ffi"))]
mod service;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "ffi")]
pub mod ffi;

// The README's Rust examples are the first code most users copy, so they are
// compiled and run with the doc-tests. Without this they were checked by
// nothing.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
