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
- **Time windows** — time-window Nearest Neighbor, Solomon (1987) I1 insertion, a time-window split for the GA, and every local-search move, savings merge and ALNS insertion gated on the windows
- **Local search** — Intra-route 2-opt (Croes 1958), or-opt, 3-opt; inter-route Relocate (Or 1976) and 2-opt* (exchange of route tails, Potvin & Rousseau 1995) — all of them window-aware
- **Genetic algorithm** — Prins (2004) hybrid GA: giant tour + split DP, a population seeded with the savings, sweep and (with time windows) Solomon I1 solutions, OX crossover, and mutation followed by local search (relocate, 2-opt*, 2-opt, or-opt) written back into the tour
- **ALNS** — Random/Worst/Shaw removal + Greedy/Regret-k insertion (Ropke & Pisinger 2006)

## Quick Start

```rust
use u_routing::models::{Customer, Vehicle};
use u_routing::distance::DistanceMatrix;
use u_routing::constructive::nearest_neighbor;
use u_routing::local_search::relocate_improve;

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

let result = GaRunner::run(&problem, &config).expect("valid GA config");
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

let result = AlnsRunner::run(&problem, &destroy, &repair, &config).expect("valid ALNS config");
println!("Best cost: {}", result.best_cost);
```

## Architecture

```text
u-routing
├── models/          Domain types (Customer, Vehicle, Route, Solution)
├── distance/        Distance matrix
├── evaluation/      Route evaluator, feasibility, solution checker
├── constructive/    Nearest Neighbor (+ time windows), Clarke-Wright Savings, Sweep, Solomon I1
├── local_search/    2-opt, or-opt, 3-opt, Relocate, 2-opt*
├── fleet.rs         Fixed fleet: empty the lightest routes into the rest
├── ga/              Giant tour + Split DP (+ time windows) + education + GaProblem bridge
├── alns/            Random/Worst/Shaw removal, Greedy/Regret insertion + AlnsProblem bridge
└── service.rs       The request/response contract the WebAssembly, C and .NET bindings share
```

## Benchmarks

`examples/solomon.rs` measures the four methods on Solomon's VRPTW instances
through the C entry points, and checks every answer with
`urouting_check_solution`. The instances are not in this repository: download
the 100-customer set (SINTEF's "Solomon benchmark" page has a zip) and run

```bash
cargo run --release --features ffi --example solomon -- <dir> [bks.csv] [nn,savings,ga,alns] [settings]
```

`bks.csv` lines are `instance,vehicles,distance`; `settings` is a JSON object
merged into the `ga` / `alns` config (e.g. `{"max_generations":1000}`). Best known results rank
**vehicles first, then distance**, so a distance below the best known one with
more routes is not a better solution — read the gap together with the route
count. The depot's due date is each vehicle's `max_duration`; the fleet is the
instance's vehicle count and capacity.

All 56 instances, default settings; `ga` and `alns` with seeds 1, 2 and 3
(168 runs each), `savings` once. Mean routes and mean distance gap to the best
known solution, by Solomon's classes; every answer checked feasible:

| Class | Best known routes | `savings` routes / gap | `ga` routes / gap | `alns` routes / gap |
|---|---|---|---|---|
| C1 (9) | 10.0 | 13.4 / +25.8 % | 10.0 / +0.0 % | 10.2 / +0.9 % |
| C2 (8) | 3.0 | 5.5 / +31.6 % | 3.0 / +0.0 % | 3.2 / +0.6 % |
| R1 (12) | 11.9 | 18.2 / +16.8 % ¹ | 13.8 / −1.4 % | 13.9 / +2.6 % |
| R2 (11) | 2.7 | 10.4 / +10.3 % | 5.5 / −6.9 % | 4.4 / −1.9 % |
| RC1 (8) | 11.5 | 17.4 / +16.6 % | 13.4 / −0.2 % | 13.9 / +5.3 % |
| RC2 (8) | 3.2 | 10.6 / +9.6 % | 6.3 / −8.8 % | 5.5 / −4.7 % |
| all (56) | 7.2 | 12.9 / +18.0 % | 8.9 / −2.9 % | 8.7 / +0.5 % |

¹ One `savings` run (r101) leaves a customer unassigned; no `ga` or `alns` run
does. A negative gap comes with more routes than the best known solution —
the methods minimise distance, the benchmark ranks routes first. `ga` was at
least as good as `savings` (unassigned, then routes, then distance) on all 168
runs; it took 1.7–3.1 s per instance on average by class (seed 1), `alns` 1.3 s
(0.7–2.4 s).

A vehicle `fixed_cost` makes the methods trade distance for routes — pass
`{"fixed_cost":1000}` as the harness's `settings`. `alns` over all 56 instances
(seeds 1–3) goes from 8.7 routes at +0.5 % to 7.9 routes at +1.4 % (best known
7.2): R2 4.4 → 3.3 routes (best known 2.7), RC2 5.5 → 3.9 (3.2), C2 3.2 → 3.0
(3.0), every answer feasible. Its search removes whole routes and prices a new
route against every insertion, so the fixed cost decides how many routes it
keeps. `ga` (seeds 1–3) goes from 8.9 routes at −2.9 % to 7.25 routes at
+3.0 % (best known 7.23): R1 13.8 → 12.0 (11.9), R2 5.5 → 2.73 (2.73), RC1 13.4 → 11.5
(11.5), RC2 6.3 → 3.25 (3.25), C1 and C2 at the best known, every answer feasible,
5.3 s per instance on average. Its local
search removes a whole route when the other routes take its customers for less
than the route costs, distance plus fixed cost — without a fixed cost, only
when the distance falls. With a fixed cost its seed solutions also go through
an ejection pool first (Nagata & Bräysy 2009): a customer that fits nowhere
goes in by ejecting one or two others, which then look for a place in turn,
and after each ejection random feasible exchanges and tail swaps between
routes move the search to another arrangement of the same customers.

Best known values: SINTEF TOP, 100-customer table, read without its marks
(`rc101b`, `r205b`, `r111*`).

`examples/cvrplib.rs` measures `ga` on capacitated instances in the CVRPLIB
format through the same C entry points, the rounded Euclidean distances
passed as `distances`, against the `.sol` files' costs:

```bash
cargo run --release --features ffi --example cvrplib -- <dir> [generations] [seeds]
```

On the first ten of Uchoa et al.'s X set (X-n101-k25 … X-n143-k7, seeds 1–3,
default settings) the mean gap is 1.7 % (0.0–4.4 %), 0.9 s each, every answer
feasible. Larger ones (seed 1): X-n200-k36 3.1 % in 1.9 s, X-n251-k28 2.7 %
in 2.5 s, X-n303-k21 3.7 % in 4.7 s, X-n401-k29 2.0 % in 8.6 s --
`config.time_limit_ms` bounds the time.

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

### TypeScript

Every exported function declares its parameter and return types, and the
declarations are generated from the same structs the binding reads and
serialises, so they cannot drift from what it actually accepts and returns:

```ts
export function solve_vrp(problem: VrpInput): VrpOutput;
```

An absent optional value is declared `T | undefined`, which is what the binding
sends. Nothing needs an `as` cast -- and a wrong assumption about a result's
shape is a compile error rather than something that fails at run time.

The same holds on the way in: a method the solver does not have, or a setting
`config` does not have, does not compile. The binding still validates every
input at the boundary, for JavaScript callers and for values that reach it
through a cast, and a rejected one says what was wrong.

### Quick Start

```javascript
import { solve_vrp } from '@iyulab/u-routing';

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

#### `codes() -> Codes`

Every refusal `code` this package's functions throw, with the fields that can come
with it, and every violation `code` `check_vrp_solution` reports:
`{ refusals: [{ code, fields }], violations: [{ code, fields, checkers }] }`. It is
built from the tables the refusals and violations come from, so a caller that
translates codes can compare its list with the package's; the tables below list
the same codes with their meaning. The C library returns the same JSON from
`urouting_codes()` and the .NET package from `RoutingClient.GetCodes()`.

#### `available_methods() -> MethodInfo[]`

Every method `solve_vrp` takes, with the `config` settings each reads (`{ name, reads }`).
It is the table `solve_vrp`'s refusal of an unread setting is made from, so the two cannot
disagree. See the example under `solve_vrp`.

#### `solve_vrp(input) -> VrpOutput`

Solve a capacitated VRP with optional time windows. Four solver methods available,
all of which keep time windows (see below).

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
    "mutation_rate": 0.1, "elite_ratio": 0.1, "seed": 42
  }
}
```

**Distances: `distances`.** By default every length is the Euclidean distance between
the points' `x`/`y`. A road network, rounded benchmark distances or any other metric
goes in `distances`: one row per point, the depot first and then the customers in
the order given (`distances[0][1]` is the depot to `customers[0]`), each row as long.
It replaces the coordinates for every length -- routes, `max_distance`, the travel
time behind time windows and `max_duration`, the totals -- and `check_vrp_solution`
reads it the same way. It need not be symmetric -- a one-way street makes the way
there shorter than the way back, and the moves that reverse part of a route price
the reversed edges. The coordinates still order the customers for `"ga"`'s sweep
seed only.

**Travel times: `durations`.** Travel time equals distance unless `durations` gives it,
laid out as `distances` (with or without `distances` beside it). Time windows,
`max_duration` and each route's `duration` are kept on it; lengths, `max_distance` and
costs stay the distances -- a slow road is long in time and short in kilometres.

```javascript
solve_vrp({
  customers: [{ id: 1, x: 0, y: 0, demand: 1 }, { id: 2, x: 0, y: 0, demand: 1 }],
  depot: { x: 0, y: 0 },
  distances: [[0, 4, 6], [4, 0, 3], [6, 3, 0]],
});
```

**GA config** uses `population_size`, `max_generations`, `mutation_rate`, `elite_ratio`.
**ALNS config** uses `max_iterations`. Both accept `seed` and `time_limit_ms`; every method reads
`max_vehicles`. A setting the request's `method` does not read is refused
(`invalid_option`, naming the methods that do) rather than accepted and ignored.
`available_methods()` returns that table, so a form can offer only the settings that
change the chosen method's run:

```js
import { available_methods } from '@iyulab/u-routing';

const methods = available_methods();
// [{ name: "nn", reads: ["max_vehicles"] }, { name: "savings", reads: ["max_vehicles"] },
//  { name: "ga", reads: ["population_size", "max_generations", "mutation_rate", "elite_ratio", "seed", "time_limit_ms", "max_vehicles"] },
//  { name: "alns", reads: ["max_iterations", "seed", "time_limit_ms", "max_vehicles"] }]
const reads = (method, setting) =>
  methods.some((m) => m.name === method && m.reads.includes(setting));
console.log(reads('ga', 'max_iterations')); // false -- leave the field out
```

**How each method reads `vehicles`.** `"nn"` assigns every route to a vehicle
in the list, so it uses at most that many vehicles, each with its own capacity;
customers the fleet cannot carry come back in `unassigned`. `"savings"`, `"ga"`
and `"alns"` plan every route with one capacity and open as many routes as the
demand needs, so `num_vehicles` can exceed the length of `vehicles`; they
reject a fleet whose vehicles differ in capacity, limits or costs. Without
`vehicles`, there is one vehicle of unlimited capacity.

**What a route costs: `fixed_cost` and `cost_per_distance` on a vehicle.** A
route costs its vehicle's `fixed_cost` (default 0, `>= 0`) plus
`cost_per_distance` (default 1, `> 0`) times its distance, and the plan's
`total_cost` is the sum. `"savings"`, `"ga"` and `"alns"` minimize it: with the
default costs that is total distance; a fixed cost makes a route worth saving
even at some extra distance, so a plan uses fewer routes where time windows
leave a longer single route open. A large fixed cost approximates the
"routes first, then distance" ranking of the Solomon benchmarks. `"nn"` builds
its routes greedily and reports their cost.

```js
solve_vrp({ depot: { x: 0, y: 0 }, customers,
            vehicles: [{ capacity: 200, fixed_cost: 1000 }],
            method: "alns" });
```

**Route limits: `max_distance` and `max_duration` on a vehicle.** A vehicle may
also bound one route: `max_distance` is the longest distance it may travel, depot
to depot, and `max_duration` the latest it may be back at the depot, on the
clock time windows use (departure at 0, travel time equal to distance, waiting
for a window to open, then service). Every method keeps them the way it keeps
time windows -- in construction, in every search move, when a fixed fleet's
routes are emptied and their customers moved -- and a customer no route can
reach within them comes back in `unassigned`. Left out (or `null`), there is no
limit.

```js
solve_vrp({ depot: { x: 0, y: 0 }, customers,
            vehicles: [{ capacity: 100, max_distance: 120, max_duration: 480 }],
            method: "savings" });
```

**How many routes a plan may use: `config.max_vehicles`.** The length of
`vehicles` is not that number. For `"savings"`, `"ga"` and `"alns"` a
single-entry list is how a caller states one capacity, which is why the
example above passes `vehicles: [{ capacity: 100 }]` and still gets as many
routes as the demand needs. A caller whose fleet really is fixed says so with
`config.max_vehicles`, and every method keeps to it, `"nn"` included (`"nn"`
cannot exceed its vehicle list, but it can exceed a smaller `max_vehicles`).
When the plan a method found needs more routes, the lightest routes are
emptied one at a time and their customers moved, cheapest insertion first, into
the routes that remain -- never past a vehicle's capacity and never making a
customer late. Customers that fit nowhere are listed in `unassigned`, so the
result is always a plan the fleet can run, and what it leaves out is named.
Emptying the lightest route first keeps the most demand served. Left out,
nothing is capped and the methods behave exactly as before.

```js
import { solve_vrp } from '@iyulab/u-routing';

// Six customers of demand 10, two trucks of 20: one route's worth does not fit.
const customers = [1, 2, 3, 4, 5, 6].map((id) => {
  const a = (2 * Math.PI * id) / 6;
  return { id, x: 10 * Math.cos(a), y: 10 * Math.sin(a), demand: 10 };
});
const out = solve_vrp({ depot: { x: 0, y: 0 }, customers, vehicles: [{ capacity: 20 }],
                        method: "ga", config: { max_vehicles: 2, seed: 1 } });
console.log(out.num_vehicles, out.unassigned); // 2, the two customers no truck can take
if (out.num_vehicles > 2 || out.unassigned.length !== 2) throw new Error("fleet not kept");
```

In Rust the same step is `u_routing::fleet::limit_routes`, which takes any
plan, a capacity per route and the limit.

**Time windows.** `time_window: [ready, due]` is measured in distance units: a
vehicle leaves the depot at time 0, travels one distance unit per time unit,
waits if it arrives before `ready`, and must arrive by `due`; `service_time`
is spent at the customer. Every method keeps the windows: a savings merge, an
ALNS insertion and every local-search move (2-opt, or-opt, 3-opt, relocate,
exchange) are taken only when the resulting route reaches each customer by its
`due`. A customer no route can reach on time comes back in `unassigned` rather
than being served late.

**Config constraints:**

| Parameter | Method | Constraint | Default |
|---|---|---|---|
| `population_size` | GA | >= 2 | 50 |
| `max_generations` | GA | >= 1 | 200 |
| `mutation_rate` | GA | (0, 1] — outside is refused (`parameter_out_of_range`, `exclusive_min: 0`, `max: 1`) | 0.1 |
| `elite_ratio` | GA | (0, 1) — outside is refused (`parameter_out_of_range`, `exclusive_min: 0`, `exclusive_max: 1`; 1 would make every individual an elite); must leave at least one elite in the population | 0.1 |
| `max_iterations` | ALNS | >= 1 | 500 |
| `seed` | Both | a whole number from 0 to 2^64 − 1 (optional) | random |
| `time_limit_ms` | Both | a whole number of milliseconds, >= 1 (optional): the run returns the best plan found when it is spent | no limit |
| `max_vehicles` | All | >= 1 (optional) | no limit |

Counts, the seed and a customer's `id` are whole numbers. A negative one is
refused by the same range check as a too-small one — `population_size: -1` is
`parameter_out_of_range` with `min: 2`, as `1` is — and one with a fractional
part (`2.5`) is `malformed_input` at its field.

**Errors:**

`solve_vrp` throws an `Error` whose `message` is readable text and which
carries a `code` naming the reason, next to the values behind it — so a program
can explain a refusal and point at what to change without parsing the message.
It is synchronous, so catch it with `try`/`catch` (there is no promise to
reject); it never surfaces as a `RuntimeError: unreachable` panic.

```js
import { solve_vrp } from '@iyulab/u-routing';

try {
  solve_vrp({
    customers: [
      { id: 1, x: 1.0, y: 0.0 },
      { id: 1, x: 2.0, y: 0.0 },
    ],
    depot: { x: 0.0, y: 0.0 },
  });
} catch (err) {
  console.log(err.code, err.id, err.first, err.second); // duplicate_id 1 0 1
}
```

| `code` | Fields | Meaning |
|---|---|---|
| `unknown_option` | `parameter`, `got`, `expected` | A `method` that names none of `"nn"`, `"savings"`, `"ga"`, `"alns"` |
| `duplicate_id` | `id`, `first`, `second` | Two customers share an `id` (at positions `first` and `second`, from 0) — `routes` and `unassigned` name customers by id, so each needs its own |
| `invalid_time_window` | `id`, `index`, `ready`, `due` | A `time_window` with `ready > due` |
| `not_whole_units` | `parameter` (`"demand"` or `"capacity"`), `index`, `id` (customers only, else `null`), `value` | A demand or capacity with a fractional part. Load is counted in whole units; a fractional amount is refused rather than rounded, so scale the unit (kilograms to grams, say) to keep it |
| `mixed_fleet` | `method`, `parameter` (`"capacity"`, `"fixed_cost"`, `"cost_per_distance"`, `"max_distance"` or `"max_duration"`), `values`, `capacities`, `index` | Vehicles that differ in capacity, cost or limits with `"savings"`, `"ga"` or `"alns"`, which plan every route with one vehicle; `index` is the first vehicle that differs from the first |
| `parameter_out_of_range` | `parameter`, `index` (vehicles and customers), `id` (customers, for a demand), `method` (method settings), `min`, `max`, `exclusive_min`, `exclusive_max`, `got` | A vehicle's `max_distance` / `max_duration` / `cost_per_distance` of 0 or less (no route could keep it), a negative `fixed_cost` (`min: 0`), a whole demand or capacity outside 0 to 2147483647 (`min: 0`, `max: 2147483647`), customers whose demands (those a vehicle can carry) add up past 2147483647 (`parameter: "total_demand"`, `min: 0`, `max: 2147483647` — a route's load is counted in 32 bits), a GA `mutation_rate` outside (0, 1] or `elite_ratio` outside (0, 1), a `config.max_vehicles` below 1, a negative `config.seed`, a customer `id` below 0 (with `index`), a negative entry of `distances` (`parameter: "distances[i]"`, `index: j`, `min: 0`), or a setting the method's runner checks (`config.population_size` below 2, `config.max_iterations` below 1, … — with `method`; a negative count is refused here like a too-small one). `min` / `max` are the closed (included) bounds and are always present — `null` on a side that is open or unbounded; `exclusive_min` / `exclusive_max` are the open (excluded) bounds, present only when that end is open: `max_distance` (> 0) is refused with `min: null, exclusive_min: 0`, `mutation_rate` (0, 1] with `exclusive_min: 0, max: 1` |
| `invalid_option` | `parameter`, `method` | A setting the method does not read (`config.max_iterations` with `"ga"`), or settings the method rejects only in combination (an `elite_ratio` leaving no elite in the population) |
| `value_not_finite` | `parameter`, `index` | A NaN or ±Infinity anywhere in an argument — `parameter` is the path to it (`config.nodes[1]`), `index` its position in that array, or `null` |
| `malformed_input` | `parameter`, `index` (inside an array) | An argument of the wrong shape or type (a missing or unknown key), a count, seed or `id` that is not a whole number (`config.population_size: 2.5` — `parameter` is the field), or a JSON string. The C ABI names the field the same way (`customers[1].id`); text that is not JSON at all is `request` |
| `dimension_mismatch` | `parameter`, `expected`, `got` | `distances` without one row per point (`parameter: "distances"`) or a row of another length (`"distances[2]"`), or (`check_vrp_solution`) a `solution.route_vehicles` without one entry per route |
| `internal` | — | A fault in this package rather than in the request — please report it |

The C library and the .NET package report the same `code` and fields: the C
error body is `{"error": "<message>", "code": ..., ...}`, and `RoutingException`
exposes them as `ErrorCode`, `Parameter`, `Index` and `Details`.

**Output:**
```json
{
  "routes": [[1, 3, 5], [2, 4]],
  "route_vehicles": [0, 0],
  "route_summaries": [
    { "distance": 25.5, "load": 14, "cost": 25.5, "duration": 25.5 },
    { "distance": 17.0, "load": 9, "cost": 17.0, "duration": 17.0 }
  ],
  "total_distance": 42.5,
  "total_cost": 42.5,
  "num_vehicles": 2,
  "method_used": "ga",
  "computation_time_ms": 120.0,
  "unassigned": [],
  "stop_reason": "no_improvement"
}
```

`stop_reason` says why the method stopped: `"completed"` for `"nn"` and
`"savings"`, which build the plan in one pass; `"iteration_limit"` when `"ga"`
or `"alns"` spent its `max_generations` / `max_iterations`; `"no_improvement"`
when `"ga"` went its stall budget without a better plan; `"time_limit"` when
`time_limit_ms` ran out first.

`unassigned` lists the customers (by `id`) that no route serves — for example
when a fixed fleet under `"nn"` cannot carry the total demand. A non-empty list
means the plan is partial. `route_vehicles` says which vehicle drives each route:
its index in `vehicles` — `"nn"` gives each route its own vehicle from the list,
`"savings"`, `"ga"` and `"alns"` drive every route with the single vehicle type,
`0` (also `0` when the request lists no vehicles). `total_cost` prices each
route with its vehicle's `fixed_cost` and `cost_per_distance` (equal to
`total_distance` with the defaults).

`route_summaries` has one entry per route, in the order of `routes`: its `distance`
(depot to depot), the `load` it carries, its `cost` (the vehicle's `fixed_cost` plus
`cost_per_distance` × `distance`) and its `duration` — when it is back at the depot,
leaving at time 0, with one unit of distance one unit of time, plus waiting for time
windows to open and service times (equal to `distance` for a route with neither).
`total_distance` is the sum of the `distance`s and `total_cost` of the `cost`s.

#### `check_vrp_solution(problem, solution) -> CheckOutput`

Checks a plan against the problem it was solved for — the `solve_vrp` request,
read the same way (the same refusals, and `method` decides whether each vehicle
drives one route, `"nn"`, or is a type any number of routes use). The plan is in
the shape `solve_vrp` returns: `routes`, `route_vehicles` (optional — `0` for
every route), `unassigned` and, optionally, `total_distance` and `total_cost` —
these five keys and no others (the rest of `solve_vrp`'s output says nothing a check reads, and
an unread key is refused rather than ignored). The plan can come from
`solve_vrp` or from anywhere else (a hand edit, another solver). A plan that
breaks the problem is a result, not an error:

```js
import { solve_vrp, check_vrp_solution } from '@iyulab/u-routing';

const problem = {
  customers: [
    { id: 1, x: 1.0, y: 0.0, demand: 10 },
    { id: 2, x: 0.0, y: 1.0, demand: 10 },
  ],
  vehicles: [{ capacity: 15 }],
  depot: { x: 0.0, y: 0.0 },
  method: "savings",
};
const { routes, route_vehicles, unassigned, total_distance } = solve_vrp(problem);
const plan = { routes, route_vehicles, unassigned, total_distance };
console.log(check_vrp_solution(problem, plan).feasible); // true

const edited = { routes: [[1, 2]], unassigned: [] };
console.log(check_vrp_solution(problem, edited).violations[0]);
// { code: "capacity_exceeded", route: 0, load: 20, capacity: 15 }
```

`violations` lists every rule broken, each with a `code` and the values behind
it (`route` is an index into `routes`; `customer` and `vehicle` are as the
request and plan name them):

| `code` | Fields | Meaning |
|---|---|---|
| `capacity_exceeded` | `route`, `load`, `capacity` | The route carries more than its vehicle's capacity |
| `time_window_violated` | `route`, `customer`, `arrival`, `due` | The route reaches the customer after its window closes |
| `max_distance_exceeded` | `route`, `distance`, `max_distance` | The route is longer than its vehicle's `max_distance` |
| `max_duration_exceeded` | `route`, `duration`, `max_duration` | The route returns later than its vehicle's `max_duration` |
| `unknown_customer` | `route`, `customer` | The route visits an id no customer has |
| `customer_visited_twice` | `route`, `customer` | The customer is visited a second time, on this route |
| `customer_missing` | `customer` | The customer is on no route and not in `unassigned` |
| `wrongly_unassigned` | `customer` | In `unassigned` but served, listed twice, or no customer of the request |
| `unknown_vehicle` | `route`, `vehicle` | The route's vehicle is not in `vehicles` |
| `vehicle_reused` | `route`, `vehicle` | `"nn"` drives each vehicle once; this route uses one again |
| `too_many_routes` | `routes`, `max_routes` | More routes than `config.max_vehicles` |
| `distance_mismatch` | `reported`, `actual` | The reported `total_distance` differs from the routes' own (compared only when every route can be driven) |
| `cost_mismatch` | `reported`, `actual` | The reported `total_cost` differs from what the routes cost with their vehicles (compared only when every route can be driven by a known vehicle) |

Refusals are those of `solve_vrp`, and `dimension_mismatch` when `route_vehicles`
does not have one entry per route. The C library takes `{"problem": <the solve request>,
"solution": {...}}` (`urouting_check_solution`) and the .NET package
`RoutingClient.CheckSolution(request, VrpPlan.Of(solution))`.

## npm (WebAssembly)

```bash
npm install @iyulab/u-routing
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

A browser **without** a bundler is not supported: the package loads its `.wasm`
file with an ES module import, which browsers refuse (`application/wasm` is not a
module script type), so `<script type="module">` from a CDN fails, and CDN
re-bundling services fail on the same import. Use a bundler or Node.

## Related

- [u-numflow](https://crates.io/crates/u-numflow) — Mathematical optimization primitives
- [u-metaheur](https://crates.io/crates/u-metaheur) — Metaheuristic algorithms
- [u-schedule](https://crates.io/crates/u-schedule) — Scheduling optimization
