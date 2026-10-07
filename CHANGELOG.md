# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.4 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
