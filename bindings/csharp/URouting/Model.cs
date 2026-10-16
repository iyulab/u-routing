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

/// <summary>A setting of <see cref="VrpConfig"/>, by the key the engine reads.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<RoutingSetting>))]
public enum RoutingSetting
{
    /// <summary><see cref="VrpConfig.PopulationSize"/>.</summary>
    [JsonStringEnumMemberName("population_size")] PopulationSize,
    /// <summary><see cref="VrpConfig.MaxGenerations"/>.</summary>
    [JsonStringEnumMemberName("max_generations")] MaxGenerations,
    /// <summary><see cref="VrpConfig.MutationRate"/>.</summary>
    [JsonStringEnumMemberName("mutation_rate")] MutationRate,
    /// <summary><see cref="VrpConfig.EliteRatio"/>.</summary>
    [JsonStringEnumMemberName("elite_ratio")] EliteRatio,
    /// <summary><see cref="VrpConfig.MaxIterations"/>.</summary>
    [JsonStringEnumMemberName("max_iterations")] MaxIterations,
    /// <summary><see cref="VrpConfig.Seed"/>.</summary>
    [JsonStringEnumMemberName("seed")] Seed,
    /// <summary><see cref="VrpConfig.TimeLimitMs"/>.</summary>
    [JsonStringEnumMemberName("time_limit_ms")] TimeLimitMs,
    /// <summary><see cref="VrpConfig.MaxVehicles"/>.</summary>
    [JsonStringEnumMemberName("max_vehicles")] MaxVehicles,
}

/// <summary>
/// A method and the settings it reads — the table the engine's refusal of a setting the
/// method does not read (<c>invalid_option</c>) is made from.
/// </summary>
/// <param name="Name">The method.</param>
/// <param name="Reads">The <see cref="VrpConfig"/> settings it reads; any other is refused.</param>
public sealed record MethodInfo(RoutingMethod Name, IReadOnlyList<RoutingSetting> Reads);

/// <summary>A refusal reason: the <c>code</c> an exception's <c>ErrorCode</c> carries and the
/// body fields that can stand beside it.</summary>
/// <param name="Code">The stable reason, e.g. <c>parameter_out_of_range</c>.</param>
/// <param name="Fields">The error body's fields beside it (<c>parameter</c>, <c>min</c>, …).</param>
public sealed record RefusalCode(string Code, IReadOnlyList<string> Fields);

/// <summary>A violation a checker reports: its <c>code</c>, the fields beside it, and the
/// engine operations that report it.</summary>
/// <param name="Code">The stable reason, e.g. <c>capacity_exceeded</c>.</param>
/// <param name="Fields">The violation's fields beside <c>code</c>, in wire names.</param>
/// <param name="Checkers">The engine operations reporting it, by their engine names
/// (<c>check_vrp_solution</c> is <see cref="RoutingClient.CheckSolution"/>).</param>
public sealed record ViolationCode(string Code, IReadOnlyList<string> Fields, IReadOnlyList<string> Checkers);

/// <summary>Every code the engine can answer with besides a result — the same catalog the
/// WebAssembly package's <c>codes()</c> returns.</summary>
/// <param name="Refusals">Each refusal reason.</param>
/// <param name="Violations">Each checker violation.</param>
public sealed record Codes(IReadOnlyList<RefusalCode> Refusals, IReadOnlyList<ViolationCode> Violations);

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
/// Solver settings; a <c>null</c> takes the engine's default. Each method reads the ones that
/// apply to it -- the GA ones by <see cref="RoutingMethod.Genetic"/>, <see cref="MaxIterations"/>
/// by <see cref="RoutingMethod.Alns"/>, <see cref="Seed"/> and <see cref="TimeLimitMs"/> by both -- and a setting the request's
/// method does not read is refused (<c>invalid_option</c>), not ignored.
/// <see cref="MaxVehicles"/> is the fixed fleet every method keeps to.
/// </summary>
public sealed record VrpConfig
{
    /// <summary>GA population size (default 50).</summary>
    public int? PopulationSize { get; init; }
    /// <summary>GA generations (default 200).</summary>
    public int? MaxGenerations { get; init; }
    /// <summary>GA mutation rate in (0, 1] (default 0.1).</summary>
    public double? MutationRate { get; init; }
    /// <summary>GA elite ratio in (0, 1) (default 0.1).</summary>
    public double? EliteRatio { get; init; }
    /// <summary>ALNS iterations (default 500).</summary>
    public int? MaxIterations { get; init; }
    /// <summary>Random seed, for reproducible runs.</summary>
    public ulong? Seed { get; init; }
    /// <summary>Wall-clock budget in milliseconds, at least 1: the run returns the best plan found by then (<c>StopReason</c> <c>time_limit</c>).</summary>
    public ulong? TimeLimitMs { get; init; }
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
/// ends) and <see cref="RouteVehicles"/> the vehicle that drives each — its index in
/// <see cref="VrpRequest.Vehicles"/> (<c>0</c> for the single vehicle type of the methods
/// other than nearest neighbour, and when the request lists none); <see cref="Unassigned"/>
/// lists the customers no route serves, so a partial plan does not read as a complete one.
/// <see cref="StopReason"/> says why the method stopped.
/// </summary>
public sealed record VrpSolution(
    IReadOnlyList<IReadOnlyList<long>> Routes,
    IReadOnlyList<int> RouteVehicles,
    double TotalDistance,
    int NumVehicles,
    RoutingMethod MethodUsed,
    double ComputationTimeMs,
    IReadOnlyList<long> Unassigned,
    StopReason StopReason);

/// <summary>
/// Why the method stopped — the run's own account of the plan it returned.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<StopReason>))]
public enum StopReason
{
    /// <summary>The method ran to its own end: nearest neighbour and savings build the plan in one pass.</summary>
    [JsonStringEnumMemberName("completed")] Completed,
    /// <summary>An exact method proved the plan optimal.</summary>
    [JsonStringEnumMemberName("optimal")] Optimal,
    /// <summary>The plan met the target the caller set.</summary>
    [JsonStringEnumMemberName("target_reached")] TargetReached,
    /// <summary>The search spent its generation or iteration budget.</summary>
    [JsonStringEnumMemberName("iteration_limit")] IterationLimit,
    /// <summary>The search went its stall budget without a better plan.</summary>
    [JsonStringEnumMemberName("no_improvement")] NoImprovement,
    /// <summary>The time limit ran out; the plan is the best found by then.</summary>
    [JsonStringEnumMemberName("time_limit")] TimeLimit,
    /// <summary>The caller cancelled the run.</summary>
    [JsonStringEnumMemberName("cancelled")] Cancelled,
}

/// <summary>
/// A plan to check with <see cref="RoutingClient.CheckSolution"/>: customer ids per route, the
/// vehicle of each (its index in <see cref="VrpRequest.Vehicles"/>; <c>null</c> = <c>0</c> for
/// every route), the customers reported unassigned and, if set, the total distance reported.
/// </summary>
public sealed record VrpPlan(IReadOnlyList<IReadOnlyList<long>> Routes)
{
    /// <summary>The vehicle of each route, or <c>null</c> for vehicle <c>0</c> on every route.</summary>
    public IReadOnlyList<int>? RouteVehicles { get; init; }
    /// <summary>Customers reported as served by no route.</summary>
    public IReadOnlyList<long> Unassigned { get; init; } = [];
    /// <summary>The total distance reported, or <c>null</c> to leave it unchecked.</summary>
    public double? TotalDistance { get; init; }

    /// <summary>The plan a solve returned, as a plan to check.</summary>
    public static VrpPlan Of(VrpSolution solution) => new(solution.Routes)
    {
        RouteVehicles = solution.RouteVehicles,
        Unassigned = solution.Unassigned,
        TotalDistance = solution.TotalDistance,
    };
}

/// <summary>Whether a plan keeps every rule of its problem, and each rule it breaks.</summary>
/// <param name="Feasible">No violations.</param>
/// <param name="Violations">Every rule the plan breaks; empty when <paramref name="Feasible"/>.</param>
public sealed record PlanCheck(bool Feasible, IReadOnlyList<PlanViolation> Violations);

/// <summary>
/// One way a plan breaks its problem — the engine's <c>code</c> picks the record. <c>Route</c> is
/// an index into the plan's routes; <c>Customer</c> and <c>Vehicle</c> are as the request and the
/// plan name them.
/// </summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "code")]
[JsonDerivedType(typeof(CapacityExceeded), "capacity_exceeded")]
[JsonDerivedType(typeof(TimeWindowViolated), "time_window_violated")]
[JsonDerivedType(typeof(MaxDistanceExceeded), "max_distance_exceeded")]
[JsonDerivedType(typeof(MaxDurationExceeded), "max_duration_exceeded")]
[JsonDerivedType(typeof(UnknownCustomer), "unknown_customer")]
[JsonDerivedType(typeof(CustomerVisitedTwice), "customer_visited_twice")]
[JsonDerivedType(typeof(CustomerMissing), "customer_missing")]
[JsonDerivedType(typeof(WronglyUnassigned), "wrongly_unassigned")]
[JsonDerivedType(typeof(UnknownVehicle), "unknown_vehicle")]
[JsonDerivedType(typeof(VehicleReused), "vehicle_reused")]
[JsonDerivedType(typeof(TooManyRoutes), "too_many_routes")]
[JsonDerivedType(typeof(DistanceMismatch), "distance_mismatch")]
public abstract record PlanViolation
{
    private PlanViolation() { }

    /// <summary>The route carries more than its vehicle's capacity.</summary>
    public sealed record CapacityExceeded(int Route, long Load, long Capacity) : PlanViolation;
    /// <summary>The route reaches the customer after its window closes.</summary>
    public sealed record TimeWindowViolated(int Route, long Customer, double Arrival, double Due) : PlanViolation;
    /// <summary>The route is longer than its vehicle's <see cref="Vehicle.MaxDistance"/>.</summary>
    public sealed record MaxDistanceExceeded(int Route, double Distance, double MaxDistance) : PlanViolation;
    /// <summary>The route returns later than its vehicle's <see cref="Vehicle.MaxDuration"/>.</summary>
    public sealed record MaxDurationExceeded(int Route, double Duration, double MaxDuration) : PlanViolation;
    /// <summary>The route visits an id no customer of the request has.</summary>
    public sealed record UnknownCustomer(int Route, long Customer) : PlanViolation;
    /// <summary>The customer is visited a second time, on this route.</summary>
    public sealed record CustomerVisitedTwice(int Route, long Customer) : PlanViolation;
    /// <summary>The customer is on no route and not reported unassigned.</summary>
    public sealed record CustomerMissing(long Customer) : PlanViolation;
    /// <summary>The customer is reported unassigned but served, listed twice, or unknown.</summary>
    public sealed record WronglyUnassigned(long Customer) : PlanViolation;
    /// <summary>The route's vehicle is not in the request's vehicles.</summary>
    public sealed record UnknownVehicle(int Route, long Vehicle) : PlanViolation;
    /// <summary>Nearest neighbour drives each vehicle once; this route uses one again.</summary>
    public sealed record VehicleReused(int Route, long Vehicle) : PlanViolation;
    /// <summary>More routes than <see cref="VrpConfig.MaxVehicles"/>.</summary>
    public sealed record TooManyRoutes(int Routes, int MaxRoutes) : PlanViolation;
    /// <summary>The reported total distance differs from the routes' own.</summary>
    public sealed record DistanceMismatch(double Reported, double Actual) : PlanViolation;
}

/// <summary>Source-generated reading of results — no reflection, so trimmed and NativeAOT hosts work.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(VrpSolution))]
[JsonSerializable(typeof(RoutingMethod))]
[JsonSerializable(typeof(IReadOnlyList<MethodInfo>))]
[JsonSerializable(typeof(Codes))]
[JsonSerializable(typeof(PlanCheck))]
internal sealed partial class RoutingJson : JsonSerializerContext;
