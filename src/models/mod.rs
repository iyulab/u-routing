//! Domain model types for vehicle routing problems.
//!
//! Provides the core abstractions: customers with demands and time windows,
//! vehicles with capacity constraints, routes as ordered sequences of visits,
//! and a problem trait that ties everything together.

mod customer;
mod problem;
mod route;
mod solution;
mod vehicle;

pub use customer::{Customer, TimeWindow};
pub use problem::RoutingProblem;
pub use route::{Route, Visit};
pub use solution::{Solution, Violation, ViolationType};
pub use vehicle::Vehicle;
