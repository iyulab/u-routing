# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.4 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Added

- `VERSION`: the crate's own version (`CARGO_PKG_VERSION`) as a constant at the
  crate root, so a consumer can report what it was built against without
  reading a lock file -- which names the wrong version once the crate is linked
  from another package.

## [0.22.1] - 2026-10-20

Depends on u-numflow 0.14 and u-metaheur 0.14. The .NET client is `URouting` 0.19.1.

### Fixed

- A vehicle given no `capacity` was limited to 1,000,000,000 units while the
  request's "no vehicles" default, and every demand, may run to 2,147,483,647: a
  customer asking for more than a billion units was left unassigned by a vehicle
  documented as unlimited. A vehicle without a capacity now has the most the model
  stores.
- Route loads are added in 32 bits, so demands adding up past 2,147,483,647 could
  overflow: a panic in a debug build, and in a release build a wrapped, negative
  load that every capacity check accepts, so a plan could be reported feasible while
  carrying more than the vehicle holds. Customers whose demands (those a vehicle can
  carry) add up past that are refused: `parameter_out_of_range`,
  `parameter: "total_demand"`, `min: 0`, `max: 2147483647`, `got` the sum. Use a
  larger unit (kilograms for grams).

## [0.22.0] - 2026-10-18

Depends on u-numflow 0.14 and u-metaheur 0.14. The .NET client is `URouting` 0.19.0.

### Changed

- The `"alns"` method cools to the end of its iteration budget. Its acceptance
  temperature was an absolute 100 falling by 0.05 % an iteration, which a 500-iteration
  run never cooled: the search accepted nearly every candidate to the last iteration and
  kept only the best layout seen. The temperatures are now multiples of the typical
  worsening of a candidate, measured on a few before the run, so the unit of the
  instance's distances does not matter; the start accepts a typical worsening one time
  in three and the schedule reaches a hundredth of that at the last iteration. On
  Solomon's 56 instances the mean distance over three seeds falls from 1025.1 to 1015.7
  (every seed below every seed before), with no violation and nothing unassigned.

## [0.21.0] - 2026-10-17

Depends on u-numflow 0.14 and u-metaheur 0.13. The .NET client is `URouting` 0.18.0.

### Added

- `solve_vrp` returns `route_summaries`: for each route, in the order of
  `routes`, its `distance`, `load`, `cost` and `duration` (back at the depot,
  counting waiting and service times). `total_distance` and `total_cost` are
  their sums, so a plan view no longer re-adds loads from the customers or
  rebuilds the distance metric to show a route's length or cost. .NET:
  `VrpSolution.RouteSummaries` (`RouteSummary`).
- `"alns"` removes whole routes (`RouteRemoval`, Pisinger & Ropke 2007) beside
  random, worst and Shaw removal, and its greedy and regret insertions price a
  new route — the vehicle's fixed cost plus the round trip — against every
  insertion; a route used to open only when no insertion fit. With
  `fixed_cost: 1000` on Solomon's 56 instances (three seeds) the mean route
  count falls to 7.9 (best known 7.2; R2 3.3, RC2 3.9), every answer feasible;
  with the default costs it is 8.7 routes at +0.5 % distance. Rust:
  `GreedyInsertion::with_route_cost`, `RegretInsertion::with_route_cost`,
  `alns::Destroy::Route`.

### Changed

- **Breaking** `VrpSolution` (.NET) has a `RouteSummaries` constructor
  parameter.
- **Breaking** `alns::Destroy` has a `Route` variant (a `match` over it must
  handle it).

### Fixed

- `config.time_limit_ms` bounds `"ga"` within a generation, not only between
  generations (u-metaheur's GA runner now counts its first population and its
  breeding against the budget): a 1.5 s limit on 300 customers ended near 2.7 s
  and now ends near 1.6 s; 500 ms ended near 1.1 s and now near 0.6 s.
- A customer whose demand exceeds every vehicle's capacity is reported in
  `unassigned` by every method. `"savings"` and `"alns"` used to give it a route
  of its own, and `"ga"` folded the whole tour into one overloaded route, so
  `check_vrp_solution` flagged `capacity_exceeded` on the solver's own plan.
- `ga::split` leaves a customer no vehicle can carry out of the routes instead
  of returning one route holding the whole tour, and `ga::split_tw` leaves out a
  customer that cannot be served even alone instead of dropping every customer
  after it in the tour. Rust callers of either function get the same behaviour.

## [0.20.0] - 2026-10-16

Depends on u-numflow 0.14. The .NET client is `URouting` 0.17.0.

### Added

- Vehicle costs: `fixed_cost` (default 0) and `cost_per_distance` (default 1)
  on each vehicle — a route costs `fixed_cost + cost_per_distance × distance`.
  `"savings"`, `"ga"` and `"alns"` minimize the total (the savings gain a
  route's fixed cost, the split and the GA fitness add it per route, ALNS
  counts it per route that serves a customer, relocate saves it when a move
  empties a route); with the default costs the objective is total distance as
  before. A large fixed cost approximates the routes-first ranking of the
  Solomon benchmarks. The solution reports `total_cost`, and
  `check_vrp_solution` compares a reported `total_cost` (`cost_mismatch`).
  Rust: `Vehicle::route_cost`, `Vehicle::fixed_cost_in_distance`,
  `Solution::price`, `RoutingGaProblem::with_route_cost`,
  `RoutingAlnsProblem::with_route_cost`. .NET: `Vehicle.FixedCost`,
  `Vehicle.CostPerDistance`, `VrpSolution.TotalCost`, `VrpPlan.TotalCost`,
  `PlanViolation.CostMismatch`.


### Changed

- `"alns"` adapts among the operators of Ropke & Pisinger (2006) — random,
  worst and related (Shaw) removal, greedy and regret-2 insertion — where it
  had one of each (random removal, greedy insertion), so there was nothing to
  adapt between. On Solomon's 56 instances (three seeds) the mean distance gap
  to the best known solutions falls from +5.7 % to +0.3 % and the mean route
  count from 10.0 to 8.8; a run takes about twice as long (1.1–1.6 s on
  100 customers). Rust: `alns::Destroy` / `alns::Repair` carry the operators
  as one type each for `AlnsRunner`.
- The README's *Benchmarks* section measures all 56 Solomon instances with
  three seeds, summarised by class.
- **Breaking** `split` and `split_tw` take a `route_cost` (0 for the old
  behaviour). `Vehicle::fixed_cost` and `cost_per_distance` were stored but
  read by nothing; `Solution::total_cost` was the total distance. Every
  constructor and improver now prices its solution with its vehicles.
- `mixed_fleet` also names `fixed_cost` and `cost_per_distance`.
- The README and docs name the inter-route tail exchange 2-opt* (it was
  called cross-exchange, a different operator), and the README's
  architecture tree lists the modules as they are.

## [0.19.0] - 2026-10-16

Depends on u-metaheur 0.12 and u-numflow 0.13. The .NET client is `URouting` 0.16.0.

### Changed

- `"ga"` is now a hybrid GA in the manner of Prins (2004): the initial
  population is seeded with the savings, sweep and (with time windows) Solomon
  I1 solutions, and each mutation is followed by local search on the split
  routes (relocate, cross-exchange, 2-opt, or-opt) written back into the
  tour. The permutation GA it replaces used every vehicle of the fleet on
  Solomon's instances and ended further from the best known distance than
  `"savings"` (+21 to +338 %); the hybrid reaches the best known solutions of
  c101 and c201 and is ahead of `"savings"` on all six measured instances.
  A run takes longer (0.7–2.7 s against 0.1–0.5 s on 100 customers with the
  default settings).

### Added

- `config.time_limit_ms` for `"ga"` and `"alns"` (.NET `VrpConfig.TimeLimitMs`):
  a wall-clock budget in milliseconds, a whole number of at least 1. A run that
  spends it returns the best plan found by then with `stop_reason:
  "time_limit"`. The other methods refuse it (`invalid_option`); 0 is
  `parameter_out_of_range` (`min: 1`).

- `examples/solomon.rs`: measures every method on Solomon's VRPTW instances
  through the C entry points and checks each answer with
  `urouting_check_solution`; the README's *Benchmarks* section has the first
  measurement. The instances are downloaded, not shipped. An optional JSON
  argument is merged into the `ga` / `alns` settings.

### Fixed

- A result the package fails to hand back to JavaScript is refused as
  `internal`, not as `malformed_input` with `parameter: "result"` — the fault
  is the package's, not the caller's input.

## [0.18.0] - 2026-10-16

Depends on u-metaheur 0.11 and u-numflow 0.13. The .NET client is `URouting` 0.15.0.

### Added

- `codes()` (WebAssembly): every refusal `code` the package throws (with the
  fields that can come with it) and every `check_vrp_solution` violation
  `code`, as data — `{ refusals: [{ code, fields }], violations: [{ code,
  fields, checkers }] }` — built from the tables the refusals and violations are
  built from, so a caller that translates codes can check its list against the
  package's rather than parse the README.
- The same catalog over the C ABI (`urouting_codes()`) and in .NET
  (`RoutingClient.GetCodes()` → `Codes(Refusals, Violations)`).

### Fixed

- The README's refusal table lists `dimension_mismatch` (from
  `check_vrp_solution`) and `internal`, which the package throws; a test keeps
  the tables and `codes()` equal.

## [0.17.0] - 2026-10-15

### Added

- **A plan says why its method stopped**: `stop_reason` on every `solve_vrp`
  output (WASM, C ABI, .NET `VrpSolution.StopReason`) — `"completed"` for `"nn"`
  and `"savings"`, which build the plan in one pass; `"iteration_limit"` when
  `"ga"` or `"alns"` spent its budget; `"no_improvement"` when `"ga"` stalled. The
  vocabulary is `u-metaheur`'s `StopReason`, the one every search in this family
  reports.

- **A plan checker**: `evaluation::check_solution(customers, distances, &Fleet, &Solution)`
  reports every way a plan breaks its problem — each customer served once or
  reported unassigned, each route's vehicle in the fleet (`Fleet::usage`: one
  route per vehicle, or any number), at most `Fleet::max_routes` routes, each
  route within its own vehicle's capacity, time windows, distance and duration
  limits, and each route's reported distance equal to its own. On the wire:
  WASM `check_vrp_solution(problem, solution)`, C ABI `urouting_check_solution`
  and .NET `RoutingClient.CheckSolution(request, plan)` (typed `PlanViolation`
  records) take the `solve_vrp` request unchanged and a plan in the shape it
  returns, and answer `{feasible, violations: [{code, ...}]}`.
- `solve_vrp` reports `route_vehicles`: the vehicle that drives each route, as
  its index in the request's `vehicles` (.NET `VrpSolution.RouteVehicles`).

### Changed

- **Breaking (Rust API):** `fleet::limit_routes` takes each route's vehicle
  (`vehicles: Vec<usize>` and the `fleet: &[Vehicle]` they index) instead of
  parallel capacity and limit lists, and `LimitedPlan` carries `vehicles` in
  place of `capacities` / `limits`. `ViolationType::TimeWindowViolated` carries
  `route_index`, and the type gains the plan-level violations above.
- **Breaking (Rust API):** `RouteEvaluator::evaluate_solution` is removed. It
  judged every route with the evaluator's one vehicle whatever the route's own,
  so a mixed fleet was checked against the wrong capacities; use
  `check_solution`.

## [0.16.0] - 2026-10-14

### Added

- **Which settings each method reads, as data**: `available_methods()` (WASM,
  typed `MethodInfo[]`), `urouting_methods()` (C ABI, JSON — free with
  `urouting_free_string`) and `RoutingClient.GetMethods()` (.NET, `MethodInfo`
  records with a `RoutingSetting` enum): `[{"name": "nn", "reads":
  ["max_vehicles"]}, ...]`. It is made from the same table as the refusal of a
  setting the method does not read, so a form can show only the fields that change
  the run without restating it.

### Changed

- **Breaking (.NET `URouting`):** `RoutingException` names its members the way the other
  .NET bindings' exceptions do — `Status` is the native status (was `Code`),
  `ErrorCode` the error body's `code` (was `Reason`), and `Parameter` / `Index`
  now say which value the refusal is about without reading `Details`. Migrate: `ex.Code` → `ex.Status`, `ex.Reason` → `ex.ErrorCode`.

## [0.15.0] - 2026-10-13

### Changed

- **Breaking:** a `config` setting the request's `method` does not read is
  refused as `invalid_option` (`parameter` = the setting, `method`), with the
  methods that read it in the message: `population_size`, `max_generations`,
  `mutation_rate` and `elite_ratio` are read by `ga`, `max_iterations` by
  `alns`, `seed` by both. They used to be accepted and ignored -- a `"nn"`
  request with a population size ran exactly as one without.

### Added

- `src/ffi/request_keys.json` lists the keys and option names `urouting_solve_vrp`
  reads, the keys of each customer, vehicle and `config` included. The crate's tests
  measure it from the entry point itself -- a request with a key no request has, and a
  value with a name no option has, at every level -- and fail when the file and the
  entry point differ.
- .NET: every request the client writes is tested against that list in both
  directions -- each key and name it writes is one the engine reads, and each one the
  engine reads can be written -- and a request with every setting given is solved under
  each method against the native library.

## [0.14.0] - 2026-10-13

Depends on u-numflow 0.11 and u-metaheur 0.8. The .NET client is `URouting` 0.11.0.

### Changed

- **Breaking:** a `parameter_out_of_range` refusal — WebAssembly `Error` fields and
  the C error body alike — carries an open (excluded) bound as `exclusive_min` /
  `exclusive_max` instead of `min` / `max`. A vehicle's `max_distance` or
  `max_duration` of 0 or less is refused with `{ min: null, max: null,
  exclusive_min: 0 }`, a GA `mutation_rate` or `elite_ratio` outside (0, 1] with
  `{ min: null, max: 1, exclusive_min: 0 }`; both used to say `min: 0`, as if 0
  were accepted. `min` and `max` are still always present and now hold closed
  bounds only; the `exclusive_*` fields appear only on an open end, so refusals
  with closed bounds (`config.population_size` below 2) keep their shape.
- A setting the method's runner refuses forwards the runner's `exclusive_min` /
  `exclusive_max` beside `min` / `max`.
- .NET: `RoutingException.Details` documents the four bound fields.
- **Breaking:** `config.population_size`, `max_generations`, `max_iterations`,
  `max_vehicles`, `seed` and a customer's `id` are read as whole numbers rather
  than unsigned integers — WebAssembly and the C ABI alike — so a negative value
  is refused by the range check like a too-small one instead of as
  `malformed_input` ("invalid value: integer `-1`, expected usize"). A count the
  method's runner checks carries the runner's bounds: `population_size: -1` is
  `parameter_out_of_range` `{ parameter: "config.population_size", method: "ga",
  min: 2, max: null, got: -1 }`, as `1` is. A negative `seed` is `{ min: 0 }`, a
  negative `id` `{ parameter: "id", index, min: 0 }`. A value with a fractional
  part (`2.5`) stays `malformed_input`, naming the field and saying it must be a
  whole number.
- **Breaking:** a `config.max_vehicles` below 1 is `parameter_out_of_range`
  `{ parameter: "config.max_vehicles", min: 1, max: null, got }` — 0 used to be
  `invalid_option` with `value: 0`, and a negative value `malformed_input`.
- **Breaking:** a whole demand or capacity outside 0 to 2147483647 is
  `parameter_out_of_range` `{ parameter: "demand" | "capacity", index, id, min: 0,
  max: 2147483647, got }` — WebAssembly and the C ABI alike. It was
  `not_whole_units`, which now means only a fractional amount.
- **Breaking:** a GA `config.elite_ratio` of 1 — every individual an elite — is
  refused as `parameter_out_of_range` with `exclusive_min: 0` and
  `exclusive_max: 1`; the accepted range is (0, 1), previously stated as (0, 1]
  with 1 refused as `invalid_option`.
- C ABI: a request that does not deserialize names the field it stopped at —
  `parameter` (`customers[1].id`, `config.population_size`) and, inside an array,
  `index` — as the WebAssembly binding does, instead of `request` for every
  shape error. Text that is not JSON at all is still `request`.

## [0.13.0] - 2026-10-07

Depends on u-numflow 0.10 and u-metaheur 0.7. The .NET client is `URouting` 0.10.0.

### Added

- `URouting` carries the native library for `linux-arm64` (glibc 2.39 or later).

### Changed

- Solver settings a method's runner refuses name their field: a value outside its
  range is `parameter_out_of_range` with `parameter` (`config.population_size`,
  `config.max_iterations`), `method`, `min`, `max` and `got`; a setting wrong only
  beside the others is `invalid_option` on that field. Both were `invalid_option` on
  `config` with the runner's text. A `max_vehicles` of 0 is reported on
  `config.max_vehicles`, like every other setting (it was `max_vehicles`).
- **Breaking (`URouting`):** `RoutingClient.SolveVrp` takes a `VrpRequest` (depot,
  `Customer`s with an optional `TimeWindow`, `Vehicle`s with `MaxDistance` /
  `MaxDuration`, a `RoutingMethod` and a `VrpConfig`) and returns a `VrpSolution`,
  instead of an `object` serialized by reflection and a `JsonElement`.
- `URouting` no longer uses reflection, so it works in trimmed and NativeAOT
  applications (it threw `InvalidOperationException` there); the package is marked
  `IsAotCompatible`. A NaN or infinity in the request is refused as
  `value_not_finite` with the path to it, as the WebAssembly binding does.

### Fixed

- The `URouting` README listed arm64 Linux, which the package does not carry; it now
  lists the runtimes it ships and the glibc floor.

## [0.12.0] - 2026-10-07

The .NET client is `URouting` 0.9.0.

### Added

- `vehicles[i].max_distance` and `vehicles[i].max_duration` in the request
  (WebAssembly, C ABI, URouting): every method keeps them in construction, in
  every search move and when a fixed fleet's routes are emptied, and lists a
  customer no route can reach within them in `unassigned`. A limit of 0 or less is
  `parameter_out_of_range` at the vehicle's `index`. `"savings"`, `"ga"` and
  `"alns"` refuse a fleet that differs in its limits as `mixed_fleet`, which now
  names the differing field as `parameter` with both `values`.
- `RoutingGaProblem::with_limits`, `RoutingAlnsProblem::with_limits`,
  `GreedyInsertion::with_limits`, `RegretInsertion::with_limits`.

### Changed

- **Breaking:** a vehicle's `max_distance` and `max_duration` bound every route the
  heuristics build and every move the searches take, the way time windows do --
  they were only reported by `RouteEvaluator` after the fact. `evaluation::RouteLimits`
  carries them and `evaluation::route_feasible(route, depot, distances, customers,
  &limits)` replaces `time_windows_respected` (windows, distance and duration in one
  pass). `two_opt_improve`, `or_opt_improve` and `three_opt_improve` take the limits
  as a fifth argument (`&RouteLimits::NONE` for none); the heuristics and
  inter-route searches that take a `Vehicle` read them from it. A customer no route
  can reach within the limits is unassigned.
  `split_tw` takes the limits as a fifth argument; `fleet::cheapest_insertion`
  takes a per-route `limits_of` beside `capacity_of`, and `fleet::limit_routes` a
  `Vec<RouteLimits>` beside the capacities (returned in `LimitedPlan::limits`).

### Fixed

- `sweep` ignored time windows: it now starts a new route when the next customer
  would be late (or break the vehicle's limits), and leaves a customer no route can
  reach on time unassigned.
- `solomon_i1` opened a route with any seed, even one over capacity or unreachable
  in its window on its own; such a seed is now unassigned.
- WebAssembly: a value of the wrong type inside an argument -- a `null` or a
  string where a number belongs (`customers[1].x`), or a missing field -- is refused as
  `malformed_input` with `parameter` naming the field and `index` its position in
  its array. It named only the argument ("invalid type: null, expected f64"),
  so a caller could not say which row was wrong.

## [0.11.1] - 2026-10-07

Depends on u-numflow 0.9. No other change.

## [0.11.0] - 2026-10-04

Depends on u-numflow 0.8. The .NET client is `URouting` 0.8.0.

### Changed

- **Breaking:** `config.max_vehicles` is kept, not only checked. A plan that
  needs more routes has its lightest routes emptied into the others (cheapest
  insertion, within capacity and time windows), and the customers that fit
  nowhere are reported in `unassigned` -- every method, `"nn"` included. It
  used to be refused with `routes_exceed_max_vehicles`; that code no longer
  occurs. A fixed fleet now gets a plan it can run instead of an error.

### Added

- `fleet::limit_routes` and `fleet::cheapest_insertion`: bring any plan within
  a route limit, keeping each route's own capacity and every time window.

## [0.10.0] - 2026-10-03

### Changed

- Depends on u-metaheur 0.6.

C# `URouting` NuGet 0.6.0 → **0.7.0** (binding only): rebuilt on this release so the
refusal of a GA rate outside (0, 1] reaches .NET. The C# surface is unchanged.

### Fixed

- **Breaking:** a GA `config.mutation_rate` or `config.elite_ratio` outside
  (0, 1] is refused with `parameter_out_of_range` (`parameter`, `min`, `max`,
  `got`). A mutation rate above 1 used to be clamped to 1 without a word,
  although the documented range was (0, 1].

## [0.9.1] - 2026-10-03

### Fixed

- WASM: a NaN or ±Infinity anywhere in an argument is refused with
  `value_not_finite`, with `parameter` the path to it and `index` its position
  in that array. JSON has no such numbers, so it used to reach the wire schema
  as `null` and be refused as `malformed_input` ("invalid type: null, expected
  f64") — the wrong reason, and the library's own non-finite checks behind the
  binding could not be reached.

## [0.9.0] - 2026-09-30

### Changed

- C# `URouting` NuGet 0.5.1 → **0.6.0**: `RoutingException.Reason` and
  `Details` (below).
- Depends on u-numflow 0.7 and u-metaheur 0.5.

- **Breaking:** every refusal now carries a stable `code` and the values behind
  it (`duplicate_id` with `id`, `first`, `second`; `not_whole_units` with
  `parameter`, `index`, `id`, `value`; ...). The WebAssembly binding throws an
  `Error` with these as properties instead of a bare string — `err.message`
  reads as before, but `String(err)` now starts with `Error: ` and
  `typeof err` is `"object"`. The C library writes them next to `"error"` in its
  error body. The README lists every code and its fields.
- C# `RoutingException` takes its `Message` from the error body's text instead
  of the whole JSON body, and exposes `Reason` (the code) and `Details` (the
  body with its fields).

- The README says a browser without a bundler is not supported (the package
  loads its `.wasm` through an ES module import, which browsers refuse), instead
  of listing only the environments that work.

## [0.8.0] - 2026-09-30

C# `URouting` NuGet 0.5.0 → **0.5.1** (binding only): rebuilt on this release so the
refusal of a repeated customer `id` reaches .NET. The C# surface is unchanged.

### Changed

- The publishing workflow runs the README's JavaScript examples against the
  built package before it publishes, so an example that throws is caught
  before a reader copies it.

### Fixed

- The README's JavaScript example imported a default `init` and called
  `await init()`. This package has no default export -- it initialises when it
  is imported, in Node and in bundlers alike -- so the example threw
  `init is not a function` on its first line. It now imports the functions
  directly.
- The README said rejected inputs come back as rejected promises. `solve_vrp`
  is synchronous and throws the message string; the README now says so.
- **A customer `id` given to two customers is refused**, naming the id and both
  positions. The output names customers by id alone, so such a plan came back
  as `routes: [[1, 1]]` with no way to tell which point was visited when.
  Input that used to be accepted is now refused.
- The README's GA and ALNS examples did not compile: both runners return a
  `Result`, which the examples read fields from directly.
  The README's Rust examples are now compiled and run with the doc-tests,
  so an example that stops matching the API fails CI.

## [0.7.0] - 2026-09-29

### Changed

- **`solve_vrp` declares its parameter type.** The problem was typed `any`; it
  is `VrpInput`, declared from the structs the binding deserialises, with
  `method` as `"nn" | "savings" | "ga" | "alns"`. **TypeScript code that passed
  a wrong shape or an unknown method now fails to compile**; the runtime path
  is unchanged.
- The publishing workflow now also fails if an exported function takes a
  parameter typed `any` (`check-typed-dts.sh`).

## [0.6.1] - 2026-09-20

C# `URouting` NuGet 0.4.0 → **0.5.0** (2026-09-25, binding only): rebuilt on
this release. 0.4.0 shipped the 0.5.0 engine, so `config.max_vehicles` (0.6.0)
did not reach .NET; a `SolveVrp` request may now state it. The C# surface is
unchanged.

### Added

- **Every exported WASM function declares its return type.** They were typed
  `(...) => any`, with the output's field *names* in the doc comment and the
  element types only in the README -- so a consumer's wrong assumption about a
  result's shape compiled and shipped. `as` is the only thing that can be
  written against `any`, and it is exactly the construct that silences this.

  The declarations are derived from the structs the binding already
  serialises, so there is no second copy to drift: `tsify` emits the interface
  and `unchecked_return_type` names it in the signature. The runtime path is
  unchanged -- same serializer, same bytes. An optional field is declared
  `T | undefined`, which is what the binding sends.

  A publish-path check (`scripts/check-typed-dts.sh`) fails the release if any
  exported function returns `any`, or if a declaration names a type the file
  does not declare. It runs before publishing rather than beside it in CI,
  because the two run on the same push.

  Inputs remain `any`; they are validated at the boundary.

## [0.6.0] - 2026-09-16

### Added

- **`config.max_vehicles` -- a plan may be told how many routes it may use.**
  `"savings"`, `"ga"` and `"alns"` open as many routes as the demand needs, and
  nothing in the result said whether that exceeded the caller's own fleet, so a
  caller with a fixed fleet received a plan it could not run. Stating
  `max_vehicles` makes the solver refuse such a plan, naming both the number of
  routes needed and the limit. Read by every method: `"nn"` cannot exceed its
  vehicle list, but it can exceed a smaller `max_vehicles`. A `max_vehicles` of
  0 is refused where it is stated.

  The refusal is checked against the same number the result reports as
  `num_vehicles`, so a caller comparing that field to its own fleet and the
  crate refusing can never disagree.

  This is additive: the length of `vehicles` keeps its meaning (for the three
  methods above, a single entry states one capacity for an unbounded fleet),
  and a request without `max_vehicles` behaves exactly as before.

### Changed

- `u-numflow` pin moves to 0.6 (tail-precise normal functions). No change in
  this crate's own code or output.

## [0.5.0] - 2026-09-13

### Added

- **Every method keeps time windows.** `evaluation::has_time_windows` and
  `evaluation::time_windows_respected` -- the O(n), allocation-free check of
  a route against the windows, with the same clock as `RouteEvaluator` --
  now gate every place a route is built or changed: the Clarke-Wright merge,
  the ALNS greedy and regret insertions (a customer that cannot be placed on
  time, not even on a route of its own, is left unassigned rather than
  served late), and the five local-search moves. The service therefore no
  longer refuses `"savings"` and `"alns"` for a problem with windows, and
  the GA polishes its time-window split with 2-opt and or-opt instead of
  skipping them. `URouting` NuGet 0.3.0 → **0.4.0**: the C# surface is unchanged,
  but the bundled engine now keeps time windows in every method.

### Changed

- **Breaking:** `two_opt_improve`, `or_opt_improve` and `three_opt_improve`
  take the customers as a fourth argument, so that the moves can see the
  windows. `relocate_improve` and `exchange_improve` already did.

## [0.4.0] - 2026-09-12

### Fixed

- **Breaking (C FFI):** `urouting_solve_vrp` returns its failure status for a
  rejected request (`-2` malformed JSON, `-3` rejected input). It returned `0`
  with an `{"error": ...}` body, so a caller that branches on the status -- the
  C# client does -- received the error as a successful result.
- The C FFI solver was a separate, older copy of the WebAssembly one. It knew
  only `"nn"` and `"savings"`, solved any other method name with nearest
  neighbour while reporting the requested name as `method_used`, and ignored
  `config`. Both bindings now call the same solver: all four methods and their
  settings are available over the C FFI, and an unknown method is rejected.
- **Breaking (C FFI):** request objects reject unknown keys, as the
  WebAssembly binding's already do. An unrecognised key used to be dropped
  without notice -- which is how `config` itself was being ignored.
- **Breaking:** a customer `time_window` with `ready > due` is rejected, naming
  the customer. It used to be dropped, so the problem was solved without the
  constraint and reported as solved.
- **Breaking:** a customer `demand` or vehicle `capacity` that is not a whole
  number from 0 to 2147483647 is rejected, naming the customer or vehicle. The
  wire takes JSON numbers but the model counts load in `i32`, and a value like
  `2.4` was rounded to `2` -- a fractional demand in kilograms or cubic metres
  was solved as a different problem and reported as solved. Negative values and
  values beyond the range were rounded or clamped the same way.
- **Breaking:** customer time windows are kept. Every method used to solve as
  if there were none -- the crate's time-window algorithms were never called
  -- while the documentation said windows were honoured. `"nn"` now uses the
  time-window nearest neighbour, and `"ga"` splits with time windows both when
  scoring a tour and for the final routes, without the 2-opt pass that would
  reorder them. A customer no feasible route reaches is reported in
  `unassigned`. `"savings"` and `"alns"` have no time model and reject a
  problem whose customers carry windows.
- `RoutingGaProblem` keeps customers' time windows: it splits with `split_tw`
  and skips 2-opt when any customer has one, and ranks a tour that leaves
  customers unserved behind every tour that serves them all. The time-window
  split's partial cost used to make dropping a customer look cheaper than
  serving it.
- **Breaking:** `"savings"`, `"ga"` and `"alns"` reject a fleet whose vehicles
  differ in capacity. They plan every route with one capacity and took the
  first vehicle's for the whole fleet, so vehicles of 10 and 100 were solved as
  two of 10. `"nn"`, which assigns routes to particular vehicles, takes a mixed
  fleet as before. The README now states how each method reads `vehicles`.
- JSON inputs parse to the nearest `f64` (`serde_json` `float_roundtrip`).
- Both bindings report `unassigned`: the customers no route serves. When a
  fixed fleet could not carry every customer, the ones left out were simply
  missing from `routes`, so a partial plan read as a complete one.

### Changed

- The C FFI response carries `computation_time_ms`, like the WebAssembly one.
- An unknown method is reported even when the problem has no customers.

## [0.3.3] - 2026-09-07

### Changed

- **`rand` is now 0.10** and **`getrandom` 0.4** on WebAssembly targets. This
  crate does not name `rand` types in its public signatures, so the change is
  internal and the API is unaffected; the sequences produced for a given seed are
  unchanged. The `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that `getrandom`
  0.3 required is no longer needed.
- **`u-metaheur` is now required at 0.4 and `u-numflow` at 0.4** (previously 0.3
  for both), following those crates' own `rand` 0.10 breaks.
- **The minimum supported Rust version is now declared as 1.87** and is verified
  by building on that exact toolchain; 1.86 and below fail. The requirement comes
  from this crate's own use of `unsigned_is_multiple_of`, stabilised in 1.87.

## [0.3.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.3.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Rust API unchanged.

### Changed

- `u-numflow` dependency `^0.2` → `^0.3` (compatible; 0.3.0 publishes the
  previously-unreleased `wasm` feature and input-validation hardening —
  no API used by this crate changed).


## [0.3.0] - 2026-06-12

### Changed — BREAKING (WASM)

- WASM input objects (`solve_vrp` — including nested customer, vehicle, depot,
  and `config` objects) now **reject unknown keys** with an explicit
  `unknown field` error instead of silently ignoring them
  (`serde(deny_unknown_fields)`).
- Known consumer pitfall this surfaces: GA options must be sent under
  `config` (not `ga_config`) — previously a misnamed key was silently
  dropped and the solver ran with default GA parameters.

### Changed

- Dependency: `u-metaheur` `^0.2` → `^0.3`.

## [0.2.4] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes — exported functions
  take native JS objects/arrays, and JSON-string arguments are now rejected
  early with a descriptive error.

## Earlier releases

- 0.2.3 — 2026-06-10
- 0.2.2 — 2026-03-09
- 0.2.1 — 2026-03-08
- 0.2.0 — 2026-03-08
- 0.1.0 — 2026-02-09
