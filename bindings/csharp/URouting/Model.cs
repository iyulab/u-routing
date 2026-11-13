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
    /// <summary><see cref="VrpConfig.Balance"/>.</summary>
    [JsonStringEnumMemberName("balance")] Balance,
}

/// <summary>What a route's workload counts in a <see cref="Balance"/>.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<WorkloadQuantity>))]
public enum WorkloadQuantity
{
    /// <summary>The load as a share of the route's vehicle's capacity.</summary>
    [JsonStringEnumMemberName("load")] Load,
    /// <summary>The route's length, depot to depot.</summary>
    [JsonStringEnumMemberName("distance")] Distance,
    /// <summary>When the route is back at the depot; needs travel times, service times or time windows.</summary>
    [JsonStringEnumMemberName("duration")] Duration,
}

/// <summary>
/// Weigh the busiest route against cost: the plan minimises its total cost plus
/// <paramref name="Weight"/> × the largest workload of any route (<see cref="VrpSolution.Equity"/>).
/// </summary>
/// <param name="Quantity">What a route's workload counts.</param>
/// <param name="Weight">Units of cost one unit of the largest workload is worth (<c>&gt;= 0</c>; <c>0</c> is cost alone).</param>
public sealed record Balance(WorkloadQuantity Quantity, double Weight);

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
/// <remarks>
/// <paramref name="LatenessPenalty"/> makes the end of the <paramref name="TimeWindow"/> soft: what each
/// unit of time a vehicle is late costs, in units of distance (<c>&gt;= 0</c>). A route may then arrive after
/// <see cref="TimeWindow.Due"/> and is priced at its distance plus this times the lateness
/// (<see cref="RouteSummary.LatenessCost"/>); <c>null</c> is a hard window, where a late arrival is not
/// allowed. It has no effect on a customer without a time window.
/// </remarks>
public sealed record Customer(
    long Id,
    double X,
    double Y,
    double Demand = 0,
    double ServiceTime = 0,
    TimeWindow? TimeWindow = null,
    double? LatenessPenalty = null);

/// <summary>
/// A vehicle. <see cref="Capacity"/> <c>null</c> means effectively unlimited; a
/// <see cref="MaxDistance"/> or <see cref="MaxDuration"/> bounds every route the vehicle drives
/// (duration on the clock that starts at 0 at the depot, travel time equal to distance).
/// A route the vehicle drives costs <see cref="FixedCost"/> plus <see cref="CostPerDistance"/> ×
/// its distance (defaults 0 and 1); every method minimizes the plan's total cost, so a fixed
/// cost trades distance for fewer routes. A vehicle stands for a <em>kind</em>:
/// <see cref="Count"/> says how many vehicles of it there are (at least 1); <c>null</c> is as many as the
/// plan needs. A fleet may mix kinds that differ in capacity, costs and limits -- but not in
/// <see cref="Break"/>: every kind has the same one, or none.
/// </summary>
public sealed record Vehicle(
    double? Capacity = null,
    double? MaxDistance = null,
    double? MaxDuration = null,
    double? FixedCost = null,
    double? CostPerDistance = null,
    int? Count = null,
    VehicleBreak? Break = null,
    int? Depot = null);

/// <summary>
/// A rest the driver takes during every route of a vehicle: <see cref="Duration"/> long, starting
/// at some time in [<see cref="Earliest"/>, <see cref="Latest"/>] (on the clock that starts at 0)
/// at the stop the vehicle has just reached, before that stop's service -- so it can overlap the
/// wait for a time window to open. A route has to contain it unless the vehicle is back at the
/// depot by <see cref="Earliest"/>. It costs nothing itself; it lengthens the route, which
/// <see cref="Vehicle.MaxDuration"/> and the windows after it feel.
/// </summary>
public sealed record VehicleBreak(double Earliest, double Latest, double Duration);

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
    /// <summary>Weigh the busiest route against cost (every method reads it); <c>null</c> is cost alone.</summary>
    public Balance? Balance { get; init; }
}

/// <summary>A point routes start from and return to.</summary>
public sealed record Depot(double X, double Y);

/// <summary>
/// A routing problem: the depots (at least one), the customers, the fleet and how to solve it.
/// A vehicle starts from the depot its <see cref="Vehicle.Depot"/> indexes (the first by default).
/// </summary>
public sealed record VrpRequest(IReadOnlyList<Depot> Depots, IReadOnlyList<Customer> Customers)
{
    /// <summary>
    /// The fleet. For <see cref="RoutingMethod.NearestNeighbor"/> each entry is one route; for the
    /// other methods a single entry states one vehicle type for a fleet bounded only by
    /// <see cref="VrpConfig.MaxVehicles"/>. Empty: one vehicle of unlimited capacity.
    /// </summary>
    public IReadOnlyList<Vehicle> Vehicles { get; init; } = [];

    /// <summary>
    /// The distance (and travel time) between every two points, or <c>null</c> for the Euclidean
    /// distance between the coordinates: one row per point, the depots first and then
    /// <see cref="Customers"/> in order, each row as long; the way back may differ from the way
    /// there. The engine refuses a row count or length that does not match
    /// (<c>dimension_mismatch</c>) and a negative entry (<c>parameter_out_of_range</c>).
    /// </summary>
    public IReadOnlyList<IReadOnlyList<double>>? Distances { get; init; }

    /// <summary>
    /// The travel time between every two points, laid out as <see cref="Distances"/>, or
    /// <c>null</c> for travel time equal to distance. Time windows, <see cref="Vehicle.MaxDuration"/>
    /// and each route's duration are kept on it; lengths and costs stay the distances. Refused
    /// as <see cref="Distances"/> is.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<double>>? Durations { get; init; }

    /// <summary>The method (nearest neighbour when not set).</summary>
    public RoutingMethod Method { get; init; } = RoutingMethod.NearestNeighbor;

    /// <summary>Solver settings, or <c>null</c> for the defaults.</summary>
    public VrpConfig? Config { get; init; }
}

/// <summary>
/// A plan. <see cref="Routes"/> lists customer ids per route (the depot is implied at both
/// ends) and <see cref="RouteVehicles"/> the vehicle that drives each — its index in
/// <see cref="VrpRequest.Vehicles"/> (<c>0</c> for the single vehicle type of the methods
/// other than nearest neighbour, and when the request lists none); <see cref="RouteSummaries"/>
/// the distance, load, cost and return time of each route, whose sums are
/// <see cref="TotalDistance"/> and <see cref="TotalCost"/>; <see cref="Unassigned"/>
/// lists the customers no route serves, so a partial plan does not read as a complete one.
/// <see cref="StopReason"/> says why the method stopped. <see cref="TotalCost"/> sums, over the
/// routes, the route's vehicle's <see cref="Vehicle.FixedCost"/> plus
/// <see cref="Vehicle.CostPerDistance"/> × the route's distance. With a <see cref="VrpConfig.Balance"/>,
/// <see cref="Equity"/> is the largest <see cref="RouteSummary.Workload"/> and <see cref="Objective"/>
/// the total cost plus the balance's weight × <see cref="Equity"/>; both <c>null</c> without one.
/// </summary>
public sealed record VrpSolution(
    IReadOnlyList<IReadOnlyList<long>> Routes,
    IReadOnlyList<int> RouteVehicles,
    IReadOnlyList<RouteSummary> RouteSummaries,
    double TotalDistance,
    double TotalCost,
    double? Equity,
    double? Objective,
    int NumVehicles,
    RoutingMethod MethodUsed,
    double ComputationTimeMs,
    IReadOnlyList<long> Unassigned,
    StopReason StopReason);

/// <summary>One route's figures, as the engine evaluates the route with its vehicle.</summary>
/// <param name="Distance">The route's length, depot to depot.</param>
/// <param name="Load">The demand the route carries.</param>
/// <param name="Cost">The vehicle's fixed cost plus its cost per distance × (<paramref name="Distance"/> plus <paramref name="LatenessCost"/>).</param>
/// <param name="Duration">
/// When the route is back at the depot, leaving it at time 0: travel (one unit of distance is
/// one unit of time), waiting for time windows to open and service times. Equal to
/// <paramref name="Distance"/> for a route with neither.
/// </param>
/// <param name="Lateness">How late the route is, summed over its customers with a soft window (<see cref="Customer.LatenessPenalty"/>); <c>0</c> when every window is met or hard.</param>
/// <param name="LatenessCost">What that lateness costs, in units of distance: each customer's lateness times its penalty, summed.</param>
/// <param name="Workload">What <see cref="VrpConfig.Balance"/> counts for this route; <c>null</c> without a balance.</param>
public sealed record RouteSummary(double Distance, long Load, double Cost, double Duration, double Lateness, double LatenessCost, double? Workload);

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
    /// <summary>The total cost reported, or <c>null</c> to leave it unchecked.</summary>
    public double? TotalCost { get; init; }
    /// <summary>The largest route workload reported (needs <see cref="VrpConfig.Balance"/>), or <c>null</c> to leave it unchecked.</summary>
    public double? Equity { get; init; }
    /// <summary>The objective reported (needs <see cref="VrpConfig.Balance"/>), or <c>null</c> to leave it unchecked.</summary>
    public double? Objective { get; init; }

    /// <summary>The plan a solve returned, as a plan to check.</summary>
    public static VrpPlan Of(VrpSolution solution) => new(solution.Routes)
    {
        RouteVehicles = solution.RouteVehicles,
        Unassigned = solution.Unassigned,
        TotalDistance = solution.TotalDistance,
        TotalCost = solution.TotalCost,
        Equity = solution.Equity,
        Objective = solution.Objective,
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
[JsonDerivedType(typeof(BreakNotPlaceable), "break_not_placeable")]
[JsonDerivedType(typeof(PairOrder), "pair_order")]
[JsonDerivedType(typeof(PairSplit), "pair_split")]
[JsonDerivedType(typeof(UnknownCustomer), "unknown_customer")]
[JsonDerivedType(typeof(CustomerVisitedTwice), "customer_visited_twice")]
[JsonDerivedType(typeof(CustomerMissing), "customer_missing")]
[JsonDerivedType(typeof(WronglyUnassigned), "wrongly_unassigned")]
[JsonDerivedType(typeof(UnknownVehicle), "unknown_vehicle")]
[JsonDerivedType(typeof(FleetCountExceeded), "fleet_count_exceeded")]
[JsonDerivedType(typeof(TooManyRoutes), "too_many_routes")]
[JsonDerivedType(typeof(DistanceMismatch), "distance_mismatch")]
[JsonDerivedType(typeof(CostMismatch), "cost_mismatch")]
[JsonDerivedType(typeof(EquityMismatch), "equity_mismatch")]
[JsonDerivedType(typeof(ObjectiveMismatch), "objective_mismatch")]
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
    /// <summary>
    /// The vehicle's <see cref="Vehicle.Break"/> cannot be placed on the route: the vehicle is still out
    /// when the break is due and no stop lets it start by its latest time without closing a hard window.
    /// </summary>
    public sealed record BreakNotPlaceable(int Route) : PlanViolation;
    /// <summary>The route visits the delivery before its pickup.</summary>
    public sealed record PairOrder(int Route, long Pickup, long Delivery) : PlanViolation;
    /// <summary>The pickup and its delivery are not on one route: on two routes, or one is unassigned.</summary>
    public sealed record PairSplit(long Pickup, long Delivery) : PlanViolation;
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
    /// <summary>
    /// The request has fewer vehicles of this kind (<paramref name="Count"/>) than the plan has routes
    /// for it; <paramref name="Route"/> is the first route past the count.
    /// </summary>
    public sealed record FleetCountExceeded(int Route, long Vehicle, int Count) : PlanViolation;
    /// <summary>More routes than <see cref="VrpConfig.MaxVehicles"/>.</summary>
    public sealed record TooManyRoutes(int Routes, int MaxRoutes) : PlanViolation;
    /// <summary>The reported total distance differs from the routes' own.</summary>
    public sealed record DistanceMismatch(double Reported, double Actual) : PlanViolation;
    /// <summary>The reported total cost differs from what the routes cost with their vehicles.</summary>
    public sealed record CostMismatch(double Reported, double Actual) : PlanViolation;
    /// <summary>The reported equity differs from the routes' largest workload.</summary>
    public sealed record EquityMismatch(double Reported, double Actual) : PlanViolation;
    /// <summary>The reported objective differs from the routes' cost plus weight × their largest workload.</summary>
    public sealed record ObjectiveMismatch(double Reported, double Actual) : PlanViolation;
}

/// <summary>Source-generated reading of results — no reflection, so trimmed and NativeAOT hosts work.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(VrpSolution))]
[JsonSerializable(typeof(RoutingMethod))]
[JsonSerializable(typeof(WorkloadQuantity))]
[JsonSerializable(typeof(IReadOnlyList<MethodInfo>))]
[JsonSerializable(typeof(Codes))]
[JsonSerializable(typeof(PlanCheck))]
internal sealed partial class RoutingJson : JsonSerializerContext;
