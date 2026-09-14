# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.4 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Changed

- `u-numflow` pin moves to 0.6 (tail-precise normal functions). No change in
  this crate's own code or output; not a release on its own.

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
