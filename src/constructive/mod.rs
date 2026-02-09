//! Constructive heuristics for building initial VRP solutions.
//!
//! - [`nearest_neighbor`] — Greedy nearest-neighbor insertion, O(n²)
//! - [`clarke_wright`] — Clarke-Wright savings algorithm (1964), O(n² log n)
//! - [`sweep`] — Polar-angle sweep clustering (Gillett & Miller, 1974), O(n log n)

mod clarke_wright;
mod nearest_neighbor;
mod sweep;

pub use clarke_wright::clarke_wright_savings;
pub use nearest_neighbor::nearest_neighbor;
pub use sweep::sweep;
