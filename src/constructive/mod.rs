//! Constructive heuristics for building initial VRP solutions.
//!
//! - [`nearest_neighbor()`] — Greedy nearest-neighbor insertion, O(n²)
//! - [`nearest_neighbor_tw()`] — Time-window-aware nearest-neighbor (Solomon, 1987), O(n²)
//! - [`clarke_wright_savings()`] — Clarke-Wright savings algorithm (1964), O(n² log n)
//! - [`sweep()`] — Polar-angle sweep clustering (Gillett & Miller, 1974), O(n log n)
//! - [`solomon_i1()`] — Solomon's I1 sequential insertion for VRPTW (1987), O(n²m)
//! - [`construct_by_depot()`] — any of them for a problem with several depots: each depot's customers routed from that depot
//!
//! The first five take a table with one depot (entry 0): given several, the others would be
//! customers to visit. [`construct_by_depot()`] is how a problem with several depots uses them.

mod clarke_wright;
mod multi_depot;
mod nearest_neighbor;
mod nn_tw;
#[cfg(test)]
mod soft_tests;
mod solomon_i1;
mod sweep;

pub use clarke_wright::clarke_wright_savings;
pub use multi_depot::construct_by_depot;
pub(crate) use nearest_neighbor::fleet_vehicle;
pub use nearest_neighbor::nearest_neighbor;
pub use nn_tw::nearest_neighbor_tw;
pub use solomon_i1::solomon_i1;
pub use sweep::sweep;
