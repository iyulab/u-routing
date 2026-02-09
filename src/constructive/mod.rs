//! Constructive heuristics for building initial VRP solutions.
//!
//! - [`nearest_neighbor`] — Greedy nearest-neighbor insertion, O(n²)
//! - [`clarke_wright`] — Clarke-Wright savings algorithm (1964), O(n² log n)

mod clarke_wright;
mod nearest_neighbor;

pub use clarke_wright::clarke_wright_savings;
pub use nearest_neighbor::nearest_neighbor;
