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
- **Fleet size**: `max_vehicles` states how many routes a plan may use, and a
  plan that would need more is refused, naming both numbers

## Installation

```bash
dotnet add package URouting
```

## Usage

```csharp
using URouting;

using var routing = new RoutingClient();

var solution = routing.SolveVrp(new
{
    depot_x = 0.0,
    depot_y = 0.0,
    customers = new[]
    {
        new { id = 1, x = 10.0, y = 5.0, demand = 10.0 },
        new { id = 2, x = -4.0, y = 8.0, demand = 15.0 },
        new { id = 3, x = 6.0,  y = -9.0, demand = 20.0 },
    },
    vehicles = new[] { new { capacity = 30.0 } },
    method = "savings",
});

Console.WriteLine(solution.GetProperty("total_distance").GetDouble());
```

The request is serialized with snake_case names, so anonymous objects can be
written as above. The result is a `System.Text.Json.JsonElement` carrying
`routes`, `total_distance`, `num_vehicles`, `method_used` and `unassigned`.

A request the solver cannot honour raises `RoutingException`, which carries the
status code alongside the message.

## Platforms

Windows, Linux and macOS (x64 and arm64). The native library ships inside the
package; no separate install is needed.

## License

MIT
