using System.Text.Json.Serialization;

namespace URouting;

/// <summary>How the solver builds routes.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<RoutingMethod>))]
public enum RoutingMethod
{
    /// <summary>Nearest neighbour: one route per vehicle in the list, filled greedily.</summary>
    [JsonStringEnumMemberName("nn")] NearestNeighbor,
    /// <summary>Clarke-Wright savings, then local search.</summary>
    [JsonStringEnumMemberName("savings")] Savings,
    /// <summary>Genetic algorithm with route splitting.</summary>
    [JsonStringEnumMemberName("ga")] Genetic,
    /// <summary>Adaptive large neighbourhood search.</summary>
    [JsonStringEnumMemberName("alns")] Alns,
}

/// <summary>When a customer may be served: arrival before <see cref="Ready"/> waits, after <see cref="Due"/> is not allowed.</summary>
public readonly record struct TimeWindow(double Ready, double Due);

/// <summary>A customer to visit. <see cref="Id"/> is yours and comes back in the routes.</summary>
public sealed record Customer(
    long Id,
    double X,
    double Y,
    double Demand = 0,
    double ServiceTime = 0,
    TimeWindow? TimeWindow = null);

/// <summary>
/// A vehicle. <see cref="Capacity"/> <c>null</c> means effectively unlimited; a
/// <see cref="MaxDistance"/> or <see cref="MaxDuration"/> bounds every route the vehicle drives
/// (duration on the clock that starts at 0 at the depot, travel time equal to distance).
/// </summary>
public sealed record Vehicle(double? Capacity = null, double? MaxDistance = null, double? MaxDuration = null);

/// <summary>
/// Solver settings; each method reads the ones that apply to it, and a <c>null</c> takes the
/// engine's default. <see cref="MaxVehicles"/> is the fixed fleet every method keeps to.
/// </summary>
public sealed record VrpConfig
{
    /// <summary>GA population size (default 50).</summary>
    public int? PopulationSize { get; init; }
    /// <summary>GA generations (default 200).</summary>
    public int? MaxGenerations { get; init; }
    /// <summary>GA mutation rate in (0, 1] (default 0.1).</summary>
    public double? MutationRate { get; init; }
    /// <summary>GA elite ratio in (0, 1] (default 0.1).</summary>
    public double? EliteRatio { get; init; }
    /// <summary>ALNS iterations (default 500).</summary>
    public int? MaxIterations { get; init; }
    /// <summary>Random seed, for reproducible runs.</summary>
    public ulong? Seed { get; init; }
    /// <summary>The most routes the plan may use; customers that fit in none are <see cref="VrpSolution.Unassigned"/>.</summary>
    public int? MaxVehicles { get; init; }
}

/// <summary>A routing problem: a depot, the customers, the fleet and how to solve it.</summary>
public sealed record VrpRequest(double DepotX, double DepotY, IReadOnlyList<Customer> Customers)
{
    /// <summary>
    /// The fleet. For <see cref="RoutingMethod.NearestNeighbor"/> each entry is one route; for the
    /// other methods a single entry states one vehicle type for a fleet bounded only by
    /// <see cref="VrpConfig.MaxVehicles"/>. Empty: one vehicle of unlimited capacity.
    /// </summary>
    public IReadOnlyList<Vehicle> Vehicles { get; init; } = [];

    /// <summary>The method (nearest neighbour when not set).</summary>
    public RoutingMethod Method { get; init; } = RoutingMethod.NearestNeighbor;

    /// <summary>Solver settings, or <c>null</c> for the defaults.</summary>
    public VrpConfig? Config { get; init; }
}

/// <summary>
/// A plan. <see cref="Routes"/> lists customer ids per route (the depot is implied at both
/// ends); <see cref="Unassigned"/> lists the customers no route serves, so a partial plan does
/// not read as a complete one.
/// </summary>
public sealed record VrpSolution(
    IReadOnlyList<IReadOnlyList<long>> Routes,
    double TotalDistance,
    int NumVehicles,
    RoutingMethod MethodUsed,
    double ComputationTimeMs,
    IReadOnlyList<long> Unassigned);

/// <summary>Source-generated reading of results — no reflection, so trimmed and NativeAOT hosts work.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(VrpSolution))]
[JsonSerializable(typeof(RoutingMethod))]
internal sealed partial class RoutingJson : JsonSerializerContext;
