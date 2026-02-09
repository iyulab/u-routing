//! Local search operators for improving VRP solutions.
//!
//! - [`two_opt`] — Intra-route 2-opt edge reversal
//! - [`or_opt`] — Intra-route segment relocation
//! - [`relocate`] — Inter-route customer relocation

mod or_opt;
mod relocate;
mod two_opt;

pub use or_opt::{or_opt_improve, route_distance};
pub use relocate::relocate_improve;
pub use two_opt::two_opt_improve;
