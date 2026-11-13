// Binding behaviour gate: `config.max_vehicles` is kept, never ignored.
//
//   PKG_NAME=<package-name> node scripts/npm-behaviour-gate.mjs
//
// Run it from a directory where the package is installed -- the publish
// workflow's smoke directory, or any directory with a local build under
// node_modules. The package is resolved from the working directory, not from
// this file. The publish workflow runs it after the README examples and before
// `npm publish`; keeping the assertions in a file rather than in the workflow
// means the same checks can run against a local build before the push that
// publishes, so an assertion the engine no longer satisfies is found there.

import { createRequire } from 'node:module'
import { join } from 'node:path'

const pkg = process.env.PKG_NAME
if (!pkg) {
  console.error('usage: PKG_NAME=<package-name> node npm-behaviour-gate.mjs')
  process.exit(2)
}
const requireFromCwd = createRequire(join(process.cwd(), 'noop.js'))
const m = requireFromCwd(pkg)
let checks = 0
const ok = (c, msg) => { checks++; if (!c) throw new Error(msg) }

const depot = { x: 0, y: 0 }
// The settings each method reads -- one it does not read is refused.
const settings = {
  ga: { population_size: 10, max_generations: 5, seed: 1 },
  alns: { max_iterations: 20, seed: 1 },
}
const base = (method) => settings[method] || {}
// Six customers of demand 10, two vehicles of 20: one route's worth
// cannot be carried.
const customers = []
for (let i = 1; i <= 6; i++) {
  const a = 2 * Math.PI * i / 6
  customers.push({ id: i, x: Math.cos(a) * 10, y: Math.sin(a) * 10, demand: 10 })
}
const two = [{ capacity: 20 }, { capacity: 20 }]
for (const method of ['nn', 'savings', 'ga', 'alns']) {
  const out = m.solve_vrp({ depots: [depot], customers, vehicles: two, method,
                            config: Object.assign({}, base(method), { max_vehicles: 2 }) })
  ok(out.num_vehicles <= 2, method + ': ' + out.num_vehicles + ' routes')
  const served = out.routes.reduce((n, r) => n + r.length, 0)
  ok(out.unassigned.length !== 0 && served + out.unassigned.length === 6,
    method + ': unassigned ' + JSON.stringify(out.unassigned))
}
// Without the limit the same request needs a third route, so the
// check above measures the limit.
for (const method of ['savings', 'ga', 'alns']) {
  const out = m.solve_vrp({ depots: [depot], customers, vehicles: two, method, config: base(method) })
  ok(out.num_vehicles === 3, method + ': unlimited gave ' + out.num_vehicles)
}

// The length of vehicles is not a limit: a single entry states one
// capacity, and savings/ga/alns still open as many routes as needed.
for (const method of ['savings', 'ga', 'alns']) {
  const out = m.solve_vrp({ depots: [depot], customers, vehicles: [{ capacity: 20 }],
                            method, config: base(method) })
  ok(out.num_vehicles > 1, method + ': a single capacity entry became a fleet cap')
  ok(out.unassigned.length === 0, method + ': unexpected unassigned ' + JSON.stringify(out.unassigned))
}
// A setting the method does not read is refused, not ignored.
let refused = null
try {
  m.solve_vrp({ depots: [depot], customers, vehicles: two, method: 'nn', config: { population_size: 10 } })
} catch (e) { refused = e }
ok(refused && refused.code === 'invalid_option' && refused.parameter === 'config.population_size',
   'nn with population_size: ' + (refused ? refused.code + ' ' + refused.parameter : 'accepted'))
// The time limit holds in WebAssembly too: a budget far longer than the
// limit ends on the clock (performance.now()), not on the budget -- on
// sixty customers, which neither search finishes within a millisecond.
const sixty = []
for (let i = 1; i <= 60; i++) {
  sixty.push({ id: i, x: Math.cos(i) * 10 * (1 + i % 3), y: Math.sin(i) * 10, demand: 1 })
}
for (const [method, budget] of [['ga', { max_generations: 1000000 }], ['alns', { max_iterations: 100000000 }]]) {
  const started = Date.now()
  const out = m.solve_vrp({ depots: [depot], customers: sixty, vehicles: [{ capacity: 20 }], method,
                            config: Object.assign({ seed: 1, time_limit_ms: 1 }, budget) })
  ok(out.stop_reason === 'time_limit', method + ': stop_reason ' + out.stop_reason)
  ok(Date.now() - started < 5000, method + ': ran ' + (Date.now() - started) + ' ms past a 1 ms limit')
}
// A fleet of several kinds: every method plans with all of them, each route is
// on a kind that carries it, and `count` bounds the vehicles of a kind.
const kinds = [{ capacity: 20, fixed_cost: 1 }, { capacity: 60, fixed_cost: 30 }]
for (const method of ['nn', 'savings', 'ga', 'alns']) {
  const out = m.solve_vrp({ depots: [depot], customers, vehicles: kinds, method, config: base(method) })
  ok(out.unassigned.length === 0, method + ': mixed fleet left out ' + JSON.stringify(out.unassigned))
  out.routes.forEach((route, i) => {
    ok(out.route_summaries[i].load <= kinds[out.route_vehicles[i]].capacity,
      method + ': route ' + i + ' of load ' + out.route_summaries[i].load + ' on kind ' + out.route_vehicles[i])
  })
  const counted = m.solve_vrp({ depots: [depot], customers, vehicles: [{ capacity: 20, count: 1 }], method,
                                config: base(method) })
  ok(counted.routes.length === 1 && counted.unassigned.length === 4,
    method + ': one vehicle of 20 gave ' + counted.routes.length + ' routes, left out ' + counted.unassigned.length)
}
// Several depots: each route runs out of its vehicle's depot, the plan measures it from there, the
// plan check agrees, and what cannot be said with several depots is refused by name.
const places = [{ x: 0, y: 0 }, { x: 100, y: 0 }]
const beside = [
  { id: 1, x: 10, y: 0, demand: 1 }, { id: 2, x: 20, y: 0, demand: 1 },
  { id: 3, x: 90, y: 0, demand: 1 }, { id: 4, x: 80, y: 0, demand: 1 },
]
const atBoth = [{ capacity: 5 }, { capacity: 5, depot: 1 }]
for (const method of ['nn', 'savings', 'ga', 'alns']) {
  const problem = { depots: places, customers: beside, vehicles: atBoth, method, config: base(method) }
  const out = m.solve_vrp(problem)
  ok(out.unassigned.length === 0 && out.routes.length === 2, method + ': two depots left out ' + JSON.stringify(out.unassigned))
  // 0 -> 10 -> 20 -> 0 and 100 -> 90 -> 80 -> 100: 40 each.
  ok(Math.abs(out.total_distance - 80) < 1e-9, method + ': two depots ' + out.total_distance)
  out.routes.forEach((route, i) => {
    const from = atBoth[out.route_vehicles[i]].depot || 0
    ok(route.every((id) => (id <= 2) === (from === 0)), method + ': route ' + JSON.stringify(route) + ' from depot ' + from)
  })
  const verdict = m.check_vrp_solution(problem, {
    routes: out.routes, route_vehicles: out.route_vehicles, unassigned: out.unassigned,
    total_distance: out.total_distance, total_cost: out.total_cost,
  })
  ok(verdict.feasible, method + ': two depots checked ' + JSON.stringify(verdict.violations))
}
let spanned = null
try {
  m.solve_vrp({ depots: places, customers: beside, vehicles: atBoth, method: 'savings', config: { max_vehicles: 2 } })
} catch (e) { spanned = e }
ok(spanned && spanned.code === 'invalid_option' && spanned.parameter === 'config.max_vehicles',
   'a fixed fleet with two depots: ' + (spanned ? spanned.code + ' ' + spanned.parameter : 'accepted'))
let none = null
try {
  m.solve_vrp({ depots: places, customers: beside, vehicles: [{ capacity: 5, depot: 2 }], method: 'nn' })
} catch (e) { none = e }
ok(none && none.code === 'parameter_out_of_range' && none.parameter === 'depot' && none.index === 0,
   'a depot that is none: ' + (none ? none.code + ' ' + none.parameter : 'accepted'))

console.log(`behaviour gate: ok (${checks} checks)`)
