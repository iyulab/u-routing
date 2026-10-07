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

A request the solver cannot honour raises `RoutingException`. `Message` is
readable text; `Reason` is a stable code to branch on and `Details` is the
error body with the values behind it:

```csharp
try
{
    routing.SolveVrp(request);
}
catch (RoutingException ex) when (ex.Reason == "duplicate_id")
{
    var id = ex.Details!.Value.GetProperty("id").GetInt64();
    Console.WriteLine($"customer {id} appears twice");
}
```

The codes and their fields are listed in the crate README's *Errors* section. A NaN or
infinity in the request is refused before it reaches the engine, as `value_not_finite`
with the path to it (`customers[1].x`).

## Trimming and NativeAOT

The client uses no reflection: the request is built as JSON nodes and the solution is read
through source-generated serialization, and the package is marked `IsAotCompatible`. It
runs unchanged in trimmed and NativeAOT applications.

## Platforms

The package carries the native library for `win-x64`, `linux-x64` and `linux-arm64`
(glibc 2.39 or later), `osx-x64` and `osx-arm64`; no separate install is needed.

## License

MIT
