# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.4 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
