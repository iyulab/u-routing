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
mod route_elimination;
mod route_minimization;
mod three_opt;
mod two_opt;

pub use exchange::exchange_improve;
pub use or_opt::{or_opt_improve, route_distance};
pub use relocate::relocate_improve;
pub(crate) use route_elimination::eliminate_routes;
pub(crate) use route_minimization::minimize_routes;
pub use three_opt::three_opt_improve;
pub use two_opt::two_opt_improve;

/// The least gain a move must have to count as one. A move's gain is a
/// difference of sums of distances, whose rounding is a part in 10^16 of them;
/// 1e-12 of the largest distance (an absolute 1e-10 on a map of order 100)
/// tells a gain from that noise on a map of any size, where a fixed amount
/// would ignore every gain on a map in small units.
pub(crate) fn gain_floor(distances: &crate::distance::DistanceMatrix) -> f64 {
    1e-12 * distances.scale()
}
