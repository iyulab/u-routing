//! Constructive heuristics for building initial VRP solutions.
//!
//! - [`nearest_neighbor`] — Greedy nearest-neighbor insertion, O(n²)
//! - [`nearest_neighbor_tw`] — Time-window-aware nearest-neighbor (Solomon, 1987), O(n²)
//! - [`clarke_wright`] — Clarke-Wright savings algorithm (1964), O(n² log n)
//! - [`sweep`] — Polar-angle sweep clustering (Gillett & Miller, 1974), O(n log n)
//! - [`solomon_i1`] — Solomon's I1 sequential insertion for VRPTW (1987), O(n²m)

mod clarke_wright;
mod nearest_neighbor;
mod nn_tw;
mod solomon_i1;
mod sweep;

pub use clarke_wright::clarke_wright_savings;
pub use nearest_neighbor::nearest_neighbor;
pub use nn_tw::nearest_neighbor_tw;
pub use solomon_i1::solomon_i1;
pub use sweep::sweep;
