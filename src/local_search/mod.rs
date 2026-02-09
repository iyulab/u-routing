//! Local search operators for improving VRP solutions.
//!
//! - [`two_opt_improve()`] — Intra-route 2-opt edge reversal
//! - [`or_opt_improve()`] — Intra-route segment relocation
//! - [`three_opt_improve()`] — Intra-route 3-opt reconnection (Lin 1965)
//! - [`relocate_improve()`] — Inter-route customer relocation
//! - [`exchange_improve()`] — Inter-route cross-exchange / 2-opt* (Potvin & Rousseau, 1995)

mod exchange;
mod or_opt;
mod relocate;
mod three_opt;
mod two_opt;

pub use exchange::exchange_improve;
pub use or_opt::{or_opt_improve, route_distance};
pub use relocate::relocate_improve;
pub use three_opt::three_opt_improve;
pub use two_opt::two_opt_improve;
