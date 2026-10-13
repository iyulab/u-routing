// Run by scripts/nuget_package_smoke.sh against the freshly packed package:
// the native library loads, answers, and refuses with the documented surface.
using URouting;

using var routing = new RoutingClient();
Console.WriteLine($"native version: {routing.GetVersion()}");

var solution = routing.SolveVrp(new VrpRequest(DepotX: 0, DepotY: 0,
[
    new Customer(1, 10, 5, Demand: 10),
    new Customer(2, -4, 8, Demand: 15),
    new Customer(3, 6, -9, Demand: 20),
])
{
    Vehicles = [new Vehicle(Capacity: 30)],
    Method = RoutingMethod.Savings,
});
var served = solution.Routes.Sum(route => route.Count);
if (served != 3 || solution.Unassigned.Count != 0)
    throw new Exception($"{served} customers routed, {solution.Unassigned.Count} unassigned");

try
{
    routing.SolveVrp(new VrpRequest(0, 0, [new Customer(1, 1, 1), new Customer(1, 2, 2)]));
    throw new Exception("a duplicate customer id was accepted");
}
catch (RoutingException e) when (e.ErrorCode == "duplicate_id")
{
}

Console.WriteLine("nuget package smoke test passed");
