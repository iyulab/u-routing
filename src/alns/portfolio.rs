//! The operators one ALNS run chooses among.
//!
//! The runner takes a slice of one destroy type and one repair type, and the
//! operator traits have generic methods, so they cannot be trait objects. An
//! adaptive search needs several of each to adapt between (Ropke & Pisinger
//! 2006 use random, worst and Shaw removal with greedy and regret insertion);
//! these enums carry them as one type each.

use rand::Rng;
use u_metaheur::alns::{DestroyOperator, RepairOperator};
use u_metaheur::Deadline;

use super::destroy::{RandomRemoval, RouteRemoval, ShawRemoval, WorstRemoval};
use super::repair::{GreedyInsertion, RegretInsertion};
use super::solution_repr::RoutingSolution;

/// One of the destroy operators.
pub enum Destroy {
    /// [`RandomRemoval`].
    Random(RandomRemoval),
    /// [`WorstRemoval`].
    Worst(WorstRemoval),
    /// [`ShawRemoval`].
    Shaw(ShawRemoval),
    /// [`RouteRemoval`].
    Route(RouteRemoval),
}

impl DestroyOperator<RoutingSolution> for Destroy {
    fn name(&self) -> &str {
        match self {
            Self::Random(op) => op.name(),
            Self::Worst(op) => op.name(),
            Self::Shaw(op) => op.name(),
            Self::Route(op) => op.name(),
        }
    }

    fn destroy<R: Rng>(
        &self,
        solution: &RoutingSolution,
        degree: f64,
        rng: &mut R,
    ) -> RoutingSolution {
        match self {
            Self::Random(op) => op.destroy(solution, degree, rng),
            Self::Worst(op) => op.destroy(solution, degree, rng),
            Self::Shaw(op) => op.destroy(solution, degree, rng),
            Self::Route(op) => op.destroy(solution, degree, rng),
        }
    }

    fn destroy_within<R: Rng>(
        &self,
        solution: &RoutingSolution,
        degree: f64,
        rng: &mut R,
        deadline: &Deadline,
    ) -> Option<RoutingSolution> {
        match self {
            Self::Random(op) => op.destroy_within(solution, degree, rng, deadline),
            Self::Worst(op) => op.destroy_within(solution, degree, rng, deadline),
            Self::Shaw(op) => op.destroy_within(solution, degree, rng, deadline),
            Self::Route(op) => op.destroy_within(solution, degree, rng, deadline),
        }
    }
}

/// One of the repair operators.
pub enum Repair {
    /// [`GreedyInsertion`].
    Greedy(GreedyInsertion),
    /// [`RegretInsertion`].
    Regret(RegretInsertion),
}

impl RepairOperator<RoutingSolution> for Repair {
    fn name(&self) -> &str {
        match self {
            Self::Greedy(op) => op.name(),
            Self::Regret(op) => op.name(),
        }
    }

    fn repair<R: Rng>(&self, solution: &RoutingSolution, rng: &mut R) -> RoutingSolution {
        match self {
            Self::Greedy(op) => op.repair(solution, rng),
            Self::Regret(op) => op.repair(solution, rng),
        }
    }

    fn repair_within<R: Rng>(
        &self,
        solution: &RoutingSolution,
        rng: &mut R,
        deadline: &Deadline,
    ) -> Option<RoutingSolution> {
        match self {
            Self::Greedy(op) => op.repair_within(solution, rng, deadline),
            Self::Regret(op) => op.repair_within(solution, rng, deadline),
        }
    }
}
