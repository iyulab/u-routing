# u-routing

[![Crates.io](https://img.shields.io/crates/v/u-routing.svg)](https://crates.io/crates/u-routing)
[![docs.rs](https://docs.rs/u-routing/badge.svg)](https://docs.rs/u-routing)
[![CI](https://github.com/iyulab/u-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-routing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Vehicle routing optimization library providing building-block algorithms for
TSP, CVRP, and VRPTW variants.

## Features

- **Models** — Customer, Vehicle, Route, Solution, TimeWindow, RoutingProblem trait
- **Distance** — Dense distance/travel-time matrix with nearest-neighbor lookup
- **Evaluation** — Route feasibility checking (capacity, time windows, max distance/duration)
- **Constructive heuristics** — Nearest Neighbor (O(n²)), Clarke-Wright Savings (O(n² log n))
- **Local search** — Intra-route 2-opt (Croes 1958), inter-route Relocate (Or 1976)
- **Genetic algorithm** — Giant tour + Prins (2004) split DP, OX crossover, 2-opt refinement
- **ALNS** — Random/Worst/Shaw removal + Greedy/Regret-k insertion (Ropke & Pisinger 2006)

## Quick Start

```rust
use u_routing::models::{Customer, Vehicle};
use u_routing::distance::DistanceMatrix;
use u_routing::constructive::nearest_neighbor;
use u_routing::local_search::{two_opt_improve, relocate_improve};

let customers = vec![
    Customer::depot(0.0, 0.0),
    Customer::new(1, 1.0, 0.0, 10, 0.0),
    Customer::new(2, 2.0, 0.0, 10, 0.0),
    Customer::new(3, 3.0, 0.0, 10, 0.0),
];
let dm = DistanceMatrix::from_customers(&customers);
let vehicles = vec![Vehicle::new(0, 30)];

// Constructive → Local search pipeline
let initial = nearest_neighbor(&customers, &dm, &vehicles);
let improved = relocate_improve(&initial, &customers, &dm, &vehicles[0]);
println!("Distance: {}", improved.total_distance());
```

### GA Solver

```rust
use u_routing::models::Customer;
use u_routing::distance::DistanceMatrix;
use u_routing::ga::RoutingGaProblem;
use u_metaheur::ga::{GaConfig, GaRunner};

let customers = vec![
    Customer::depot(0.0, 0.0),
    Customer::new(1, 1.0, 0.0, 10, 0.0),
    Customer::new(2, 2.0, 0.0, 10, 0.0),
    Customer::new(3, 3.0, 0.0, 10, 0.0),
];
let dm = DistanceMatrix::from_customers(&customers);

let problem = RoutingGaProblem::new(customers, dm, 30);
let config = GaConfig::default()
    .with_population_size(50)
    .with_max_generations(200);

let result = GaRunner::run(&problem, &config);
println!("Best distance: {}", result.best_fitness);
```

### ALNS Solver

```rust
use u_routing::models::Customer;
use u_routing::distance::DistanceMatrix;
use u_routing::alns::{RoutingAlnsProblem, destroy::RandomRemoval, repair::GreedyInsertion};
use u_metaheur::alns::{AlnsConfig, AlnsRunner};

let customers = vec![
    Customer::depot(0.0, 0.0),
    Customer::new(1, 1.0, 0.0, 10, 0.0),
    Customer::new(2, 2.0, 0.0, 10, 0.0),
    Customer::new(3, 3.0, 0.0, 10, 0.0),
];
let dm = DistanceMatrix::from_customers(&customers);

let problem = RoutingAlnsProblem::new(customers.clone(), dm.clone(), 30);
let destroy = vec![RandomRemoval];
let repair = vec![GreedyInsertion::new(dm, customers, 30)];
let config = AlnsConfig::default().with_max_iterations(5000).with_seed(42);

let result = AlnsRunner::run(&problem, &destroy, &repair, &config);
println!("Best cost: {}", result.best_cost);
```

## Architecture

```
u-routing
├── models/          Domain types (Customer, Vehicle, Route, Solution)
├── distance/        Distance matrix
├── evaluation/      Route evaluator + constraint checking
├── constructive/    Nearest Neighbor, Clarke-Wright Savings
├── local_search/    2-opt, Relocate
├── ga/              Giant tour + Split DP + GaProblem bridge
└── alns/            Destroy/Repair operators + AlnsProblem bridge
```

## Dependencies

- [`u-metaheur`](https://crates.io/crates/u-metaheur) — GA/ALNS framework
- [`u-numflow`](https://crates.io/crates/u-numflow) — Math primitives, RNG

## References

- Clarke, G. & Wright, J.W. (1964). "Scheduling of Vehicles from a Central Depot to a Number of Delivery Points"
- Croes, G.A. (1958). "A method for solving traveling salesman problems"
- Or, I. (1976). "Traveling Salesman-Type Combinatorial Problems and Their Relation to the Logistics of Blood Banking"
- Prins, C. (2004). "A simple and effective evolutionary algorithm for the vehicle routing problem"
- Ropke, S. & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows"
- Shaw, P. (1998). "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems"

## WebAssembly / npm

Available as an npm package via [wasm-pack](https://rustwasm.github.io/wasm-pack/).

```bash
npm install @iyulab/u-routing
```

### Quick Start

```javascript
import init, { solve_vrp } from '@iyulab/u-routing';

await init();
const result = solve_vrp({
  customers: [
    { id: 1, x: 1.0, y: 2.0, demand: 10 },
    { id: 2, x: 3.0, y: 4.0, demand: 15 },
  ],
  vehicles: [{ capacity: 100 }],
  depot: { x: 0.0, y: 0.0 },
  method: "nn"
});
```

### Functions

#### `solve_vrp(input) -> VrpOutput`

Solve a capacitated VRP with optional time windows. Four solver methods available.

**Methods:** `"nn"` (Nearest Neighbor), `"savings"` (Clarke-Wright), `"ga"` (Genetic Algorithm + Split DP), `"alns"` (Adaptive Large Neighborhood Search).

**Input:**
```json
{
  "customers": [
    { "id": 1, "x": 1.0, "y": 2.0, "demand": 10, "service_time": 0.5, "time_window": [8.0, 12.0] }
  ],
  "vehicles": [{ "capacity": 100 }],
  "depot": { "x": 0.0, "y": 0.0 },
  "method": "ga",
  "config": {
    "population_size": 50, "max_generations": 200,
    "mutation_rate": 0.1, "elite_ratio": 0.1,
    "max_iterations": 500, "seed": 42
  }
}
```

GA uses `population_size`, `max_generations`, `mutation_rate`, `elite_ratio`. ALNS uses `max_iterations`. Both accept `seed`.

**Output:**
```json
{
  "routes": [[1, 3, 5], [2, 4]],
  "total_distance": 42.5,
  "num_vehicles": 2,
  "method_used": "ga",
  "computation_time_ms": 120.0
}
```

## Related

- [u-numflow](https://crates.io/crates/u-numflow) — Mathematical optimization primitives
- [u-metaheur](https://crates.io/crates/u-metaheur) — Metaheuristic algorithms
- [u-schedule](https://crates.io/crates/u-schedule) — Scheduling optimization
