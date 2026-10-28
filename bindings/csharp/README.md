# URouting

.NET bindings for the u-routing vehicle routing engine.

## Features

- **Methods**: nearest neighbour, Clarke-Wright savings, a genetic algorithm with
  route splitting, and adaptive large neighbourhood search
- **Time windows**: every method keeps them. A merge, an insertion and every
  local-search move are taken only when the resulting route reaches each
  customer by its due time; a customer no route can reach on time comes back
  unassigned rather than being served late
- **Capacities**: load is counted in whole units, and a fractional demand is
  refused rather than rounded
- **Fleet size**: `max_vehicles` states how many routes a plan may use; every
  method keeps to it and lists the customers that do not fit in `Unassigned`
- **Route limits**: a vehicle's `max_distance` and `max_duration` bound every
  route, depot to depot; a customer no route can reach within them is unassigned

## Installation

```bash
dotnet add package URouting
```

## Usage

```csharp
using URouting;

using var routing = new RoutingClient();

VrpSolution solution = routing.SolveVrp(new VrpRequest(DepotX: 0, DepotY: 0,
[
    new Customer(1, 10, 5, Demand: 10),
    new Customer(2, -4, 8, Demand: 15),
    new Customer(3, 6, -9, Demand: 20, TimeWindow: new TimeWindow(0, 40)),
])
{
    Vehicles = [new Vehicle(Capacity: 30)],
    Method = RoutingMethod.Savings,
});

Console.WriteLine($"{solution.NumVehicles} routes, {solution.TotalDistance:F1} long");
foreach (var route in solution.Routes)
    Console.WriteLine(string.Join(" → ", route));
if (solution.Unassigned.Count > 0)
    Console.WriteLine($"not served: {string.Join(", ", solution.Unassigned)}");
```

`VrpSolution` carries `Routes` (customer ids per route, the depot implied at both ends),
`TotalDistance`, `NumVehicles`, `MethodUsed`, `ComputationTimeMs` and `Unassigned` — the
customers no route serves within capacities, time windows, each vehicle's `MaxDistance` /
`MaxDuration` and `VrpConfig.MaxVehicles`.

Lengths are the Euclidean distance between the coordinates unless `VrpRequest.Distances` gives
them: one row per point, the depot first and then `Customers` in order — a road network's
distances, for instance. It replaces the coordinates for every length and travel time, and may differ
by direction (a one-way street). `VrpRequest.Durations`, laid out the same way, gives travel
times of their own: time windows and `MaxDuration` are kept on it, lengths and costs stay the
distances.

Each method reads only some `VrpConfig` settings — the GA ones by `Genetic`,
`MaxIterations` by `Alns`, `Seed` and `TimeLimitMs` by both, `MaxVehicles` by all — and a setting the
method does not read is refused (`invalid_option`). `GetMethods()` returns that table,
the one the refusal is made from:

```csharp
foreach (var m in routing.GetMethods())      // MethodInfo(Name, Reads)
    Console.WriteLine($"{m.Name}: {string.Join(", ", m.Reads)}");
bool shown = routing.GetMethods()
    .Single(m => m.Name == RoutingMethod.Genetic)
    .Reads.Contains(RoutingSetting.MaxIterations);   // false
```

`VrpSolution.RouteVehicles` says which vehicle drives each route (its index in
`Vehicles`), and `RouteSummaries` gives each route's distance, load, cost and return
time (`TotalDistance` and `TotalCost` are their sums). `CheckSolution` checks any plan — this one, an edited one, one from another
solver — against the request, and returns every rule it breaks as a typed record:

```csharp
PlanCheck check = routing.CheckSolution(request, VrpPlan.Of(solution));
foreach (var v in check.Violations)
    Console.WriteLine(v switch
    {
        PlanViolation.CapacityExceeded c => $"route {c.Route}: load {c.Load} over {c.Capacity}",
        PlanViolation.TimeWindowViolated t => $"route {t.Route}: customer {t.Customer} late",
        _ => v.ToString(),
    });
```

A request the solver cannot honour raises `RoutingException`. `Message` is
readable text; `ErrorCode` is a stable code to branch on (the error body's `code`),
`Parameter` and `Index` say which value it is about, `Details` is the error body
with the values behind it, and `Status` is the native status:

```csharp
try
{
    routing.SolveVrp(request);
}
catch (RoutingException ex) when (ex.ErrorCode == "duplicate_id")
{
    var id = ex.Details!.Value.GetProperty("id").GetInt64();
    Console.WriteLine($"customer {id} appears twice");
}
```

The codes and their fields are listed in the crate README's *Errors* section, and
`GetCodes()` returns them as data — every refusal code with the fields beside it, and
every `PlanViolation` code — from the tables the engine makes them from:

```csharp
Codes codes = routing.GetCodes();   // Codes(Refusals, Violations)
bool known = codes.Refusals.Any(r => r.Code == "duplicate_id");   // true
```

A NaN or
infinity in the request is refused before it reaches the engine, as `value_not_finite`
with the path to it as `Parameter` (`customers[1].x`) — `Index` is set only when
the value itself is an array element.

## Trimming and NativeAOT

The client uses no reflection: the request is built as JSON nodes and the solution is read
through source-generated serialization, and the package is marked `IsAotCompatible`. It
runs unchanged in trimmed and NativeAOT applications.

## Platforms

The package carries the native library for `win-x64`, `linux-x64` and `linux-arm64`
(glibc 2.39 or later), `osx-x64` and `osx-arm64`; no separate install is needed.

## License

MIT
