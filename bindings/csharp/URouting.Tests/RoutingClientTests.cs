using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Xunit;

namespace URouting.Tests;

/// <summary>Runs with reflection-based serialization disabled, as a trimmed or NativeAOT host does.</summary>
public class RoutingClientTests
{
    private static readonly Customer[] Customers =
    [
        new(1, 10, 5, Demand: 10),
        new(2, -4, 8, Demand: 15),
        new(3, 6, -9, Demand: 20),
        new(4, -7, -6, Demand: 5),
        new(5, 12, -2, Demand: 8),
    ];

    private readonly RoutingClient _client = new();

    public static TheoryData<RoutingMethod> AllMethods() =>
        [RoutingMethod.NearestNeighbor, RoutingMethod.Savings, RoutingMethod.Genetic, RoutingMethod.Alns];

    /// <summary>A solved plan checks clean; a broken one comes back as typed violations.</summary>
    [Fact]
    public void A_plan_is_checked_and_the_check_carries_exactly_what_the_engine_sent()
    {
        var request = new VrpRequest([new(0, 0)], Customers) { Vehicles = [new(30), new(30)] };
        var solution = _client.SolveVrp(request);
        Assert.Equal(solution.Routes.Count, solution.RouteVehicles.Count);
        var clean = _client.CheckSolution(request, VrpPlan.Of(solution));
        Assert.True(clean.Feasible);
        Assert.Empty(clean.Violations);

        string? raw = null;
        _client.ResponseObserver = body => raw = body;
        var everyone = new[] { Customers.Select(c => c.Id).ToArray() };
        var broken = _client.CheckSolution(request, new VrpPlan(everyone) { RouteVehicles = [0], TotalDistance = 1 });
        var written = JsonSerializer.SerializeToNode(broken, RoutingJson.Default.PlanCheck);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw!), written), $"engine: {raw}\nrecord: {written}");
        Assert.False(broken.Feasible);
        Assert.Contains(broken.Violations, v => v is PlanViolation.CapacityExceeded { Route: 0, Capacity: 30 });
        Assert.Contains(broken.Violations, v => v is PlanViolation.DistanceMismatch { Reported: 1 });
    }

    [Fact]
    public void The_method_list_carries_exactly_what_the_engine_sent()
    {
        var raw = RoutingClient.MethodsJson();
        var methods = _client.GetMethods();
        var written = JsonSerializer.SerializeToNode(methods, RoutingJson.Default.IReadOnlyListMethodInfo);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw), written), $"engine: {raw}\nrecord: {written}");

        Assert.Equal(Enum.GetValues<RoutingMethod>(), methods.Select(m => m.Name));
        var ga = methods.Single(m => m.Name == RoutingMethod.Genetic);
        Assert.Contains(RoutingSetting.EliteRatio, ga.Reads);
        Assert.DoesNotContain(RoutingSetting.MaxIterations, ga.Reads);
        Assert.All(methods, m => Assert.Contains(RoutingSetting.MaxVehicles, m.Reads));
    }

    [Fact]
    public void The_code_catalog_carries_exactly_what_the_engine_sent()
    {
        var raw = RoutingClient.CodesJson();
        var codes = _client.GetCodes();
        var written = JsonSerializer.SerializeToNode(codes, RoutingJson.Default.Codes);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw), written), $"engine: {raw}\nrecord: {written}");

        // The client raises value_not_finite itself; the catalog must name it.
        Assert.Contains(codes.Refusals, r => r.Code == "value_not_finite");
        // Every violation the engine names has a PlanViolation type, and no type is missing one.
        var typed = typeof(PlanViolation)
            .GetCustomAttributes(typeof(JsonDerivedTypeAttribute), false)
            .Cast<JsonDerivedTypeAttribute>()
            .Select(a => (string)a.TypeDiscriminator!)
            .Order(StringComparer.Ordinal);
        Assert.Equal(codes.Violations.Select(v => v.Code).Order(StringComparer.Ordinal), typed);
    }

    [Theory]
    [MemberData(nameof(AllMethods))]
    public void The_solution_carries_exactly_what_the_engine_sent(RoutingMethod method)
    {
        string? raw = null;
        _client.ResponseObserver = body => raw = body;
        var request = new VrpRequest([new(0, 0)], Customers)
        {
            Vehicles = method == RoutingMethod.NearestNeighbor ? [new(30), new(30)] : [new(30)],
            Method = method,
            Config = method switch
            {
                RoutingMethod.Genetic => new VrpConfig { Seed = 7, MaxGenerations = 20 },
                RoutingMethod.Alns => new VrpConfig { Seed = 7, MaxIterations = 50 },
                // A balanced plan: the record carries the workloads, equity and objective too.
                RoutingMethod.Savings => new VrpConfig { Balance = new Balance(WorkloadQuantity.Distance, 2) },
                _ => null,
            },
        };

        var solution = _client.SolveVrp(request);

        Assert.Equal(method, solution.MethodUsed);
        Assert.Equal(method == RoutingMethod.Savings, solution.Equity is not null);
        Assert.Equal(method == RoutingMethod.Savings, solution.RouteSummaries.All(r => r.Workload is not null));
        var written = JsonSerializer.SerializeToNode(solution, RoutingJson.Default.VrpSolution);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw!), written), $"engine: {raw}\nrecord: {written}");
        var served = solution.Routes.SelectMany(r => r).Concat(solution.Unassigned).Order();
        Assert.Equal(Customers.Select(c => c.Id).Order(), served);
        // One summary per route, and the totals are their sums.
        Assert.Equal(solution.Routes.Count, solution.RouteSummaries.Count);
        Assert.Equal(solution.TotalDistance, solution.RouteSummaries.Sum(r => r.Distance), 6);
        Assert.Equal(solution.TotalCost, solution.RouteSummaries.Sum(r => r.Cost), 6);
    }

    [Fact]
    public void A_balanced_plan_checks_against_its_balance()
    {
        var request = new VrpRequest([new(0, 0)], Customers)
        {
            Vehicles = [new(30)],
            Method = RoutingMethod.Savings,
            Config = new VrpConfig { Balance = new Balance(WorkloadQuantity.Load, 5) },
        };
        var solution = _client.SolveVrp(request);
        Assert.Equal(solution.TotalCost + (5 * solution.Equity!.Value), solution.Objective!.Value, 9);

        Assert.True(_client.CheckSolution(request, VrpPlan.Of(solution)).Feasible);
        var wrong = VrpPlan.Of(solution) with { Equity = solution.Equity + 1 };
        var check = _client.CheckSolution(request, wrong);
        Assert.IsType<PlanViolation.EquityMismatch>(Assert.Single(check.Violations));
    }

    [Fact]
    public void A_time_limit_ends_a_search_and_says_so()
    {
        var request = new VrpRequest([new(0, 0)], Customers)
        {
            Vehicles = [new(30)],
            Method = RoutingMethod.Alns,
            Config = new VrpConfig { Seed = 7, MaxIterations = 100_000_000, TimeLimitMs = 1 },
        };
        Assert.Equal(StopReason.TimeLimit, _client.SolveVrp(request).StopReason);
        var ga = _client.GetMethods().Single(m => m.Name == RoutingMethod.Genetic);
        Assert.Contains(RoutingSetting.TimeLimitMs, ga.Reads);
    }

    [Fact]
    public void Limits_and_windows_reach_the_engine()
    {
        // The far customer cannot be reached within the vehicle's distance limit.
        Customer[] customers = [new(1, 3, 4), new(2, 300, 400)];
        var solution = _client.SolveVrp(new VrpRequest([new(0, 0)], customers)
        {
            Vehicles = [new(MaxDistance: 50)],
            Method = RoutingMethod.Savings,
        });
        Assert.Equal([2L], solution.Unassigned);

        // A window that closes before the vehicle can arrive.
        var late = _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 30, 40, TimeWindow: new(0, 10))]));
        Assert.Equal([1L], late.Unassigned);
    }

    [Fact]
    public void A_soft_window_is_served_late_and_the_lateness_is_priced_like_distance()
    {
        // 50 away and due at 10: late by 40 whatever is done, at 2 a unit of time.
        var solution = _client.SolveVrp(new VrpRequest([new(0, 0)],
            [new(1, 30, 40, TimeWindow: new(0, 10), LatenessPenalty: 2)]));
        Assert.Empty(solution.Unassigned);
        var route = Assert.Single(solution.RouteSummaries);
        Assert.Equal(100.0, route.Distance, 9);
        Assert.Equal(40.0, route.Lateness, 9);
        Assert.Equal(80.0, route.LatenessCost, 9);
        Assert.Equal(180.0, route.Cost, 9);
        Assert.Equal(180.0, solution.TotalCost, 9);
    }

    [Fact]
    public void A_lateness_penalty_below_zero_is_refused_at_its_customer()
    {
        var refusal = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 1, 1, TimeWindow: new(0, 10), LatenessPenalty: -1)])));
        Assert.Equal("parameter_out_of_range", refusal.ErrorCode);
        Assert.Equal("lateness_penalty", refusal.Parameter);
        Assert.Equal(0, refusal.Index);
    }

    [Fact]
    public void A_pickup_and_its_delivery_ride_one_route_pickup_first_within_the_capacity_along_the_way()
    {
        // Two pairs of 3 on a line, a capacity of 4: both are served, one after the other, so the
        // route carries 3 at most although the demands add up to 6.
        Customer[] customers =
        [
            new(1, 10, 0, Demand: 3, Delivery: 11), new(2, 20, 0, Demand: 3, Delivery: 12),
            new(11, 12, 0), new(12, 22, 0),
        ];
        foreach (var method in new[] { RoutingMethod.NearestNeighbor, RoutingMethod.Savings, RoutingMethod.Alns })
        {
            var request = new VrpRequest([new(0, 0)], customers)
            {
                Vehicles = [new(4)],
                Method = method,
                Config = method == RoutingMethod.Alns ? new VrpConfig { MaxIterations = 20, Seed = 1 } : null,
            };
            var solution = _client.SolveVrp(request);
            Assert.Empty(solution.Unassigned);
            foreach (var stops in solution.Routes.Select(r => r.ToList()))
            {
                Assert.True(stops.IndexOf(1) < stops.IndexOf(11) || !stops.Contains(1), $"{method}: {string.Join(",", stops)}");
                Assert.True(stops.IndexOf(2) < stops.IndexOf(12) || !stops.Contains(2), $"{method}: {string.Join(",", stops)}");
            }
            Assert.All(solution.RouteSummaries, s => Assert.True(s.Load <= 4, $"{method}: load {s.Load}"));
            Assert.Empty(_client.CheckSolution(request, VrpPlan.Of(solution)).Violations);
        }

        // The checker names a delivery that comes before its pickup, and a pair split over routes.
        var plan = new VrpPlan([[11L, 1L, 2L, 12L]]);
        var order = _client.CheckSolution(new VrpRequest([new(0, 0)], customers) { Vehicles = [new(8)] }, plan);
        Assert.Contains(order.Violations, v => v is PlanViolation.PairOrder { Pickup: 1, Delivery: 11 });
        var split = _client.CheckSolution(new VrpRequest([new(0, 0)], customers) { Vehicles = [new(8), new(8)] },
            new VrpPlan([[1L, 2L, 12L], [11L]]) { RouteVehicles = [0, 1] });
        Assert.Contains(split.Violations, v => v is PlanViolation.PairSplit { Pickup: 1, Delivery: 11 });
    }

    [Fact]
    public void A_delivery_the_engine_cannot_pair_is_refused_where_it_is_given()
    {
        var unknown = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 1, 1, Demand: 1, Delivery: 9)])));
        Assert.Equal("invalid_option", unknown.ErrorCode);
        Assert.Equal("customers[0].delivery", unknown.Parameter);

        // The genetic algorithm does not plan pairs.
        var genetic = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 1, 1, Demand: 1, Delivery: 2), new(2, 2, 2)])
            {
                Method = RoutingMethod.Genetic,
            }));
        Assert.Equal("invalid_option", genetic.ErrorCode);
        Assert.Equal("method", genetic.Parameter);
    }

    [Fact]
    public void A_fixed_fleet_is_kept()
    {
        var solution = _client.SolveVrp(new VrpRequest([new(0, 0)], Customers)
        {
            Vehicles = [new(20)],
            Method = RoutingMethod.Savings,
            Config = new VrpConfig { MaxVehicles = 2 },
        });
        Assert.True(solution.NumVehicles <= 2);
        Assert.NotEmpty(solution.Unassigned);
    }

    [Fact]
    public void A_request_with_two_depots_is_planned_from_each_vehicles_depot()
    {
        // Depots at 0 and 100 on a line, a customer by each: 10 and 90.
        var solution = _client.SolveVrp(new VrpRequest([new(0, 0), new(100, 0)],
            [new(1, 10, 0, Demand: 1), new(2, 90, 0, Demand: 1)])
        {
            Vehicles = [new(Capacity: 5), new(Capacity: 5, Depot: 1)],
            Method = RoutingMethod.Savings,
        });
        Assert.Empty(solution.Unassigned);
        Assert.Equal(2, solution.Routes.Count);
        Assert.Equal(40.0, solution.TotalDistance, 9);
        // Each route is driven by the vehicle at the depot beside its customer.
        for (var r = 0; r < solution.Routes.Count; r++)
            Assert.Equal(solution.Routes[r][0] == 1 ? 0 : 1, solution.RouteVehicles[r]);
    }

    [Fact]
    public void A_fixed_fleet_with_several_depots_and_a_depot_that_is_none_are_refused_by_name()
    {
        var two = new List<Depot> { new(0, 0), new(100, 0) };
        var x = Assert.Throws<RoutingException>(() => _client.SolveVrp(new VrpRequest(two, Customers)
        {
            Method = RoutingMethod.Savings,
            Config = new VrpConfig { MaxVehicles = 2 },
        }));
        Assert.Equal("invalid_option", x.ErrorCode);
        Assert.Equal("config.max_vehicles", x.Details!.Value.GetProperty("parameter").GetString());
        var y = Assert.Throws<RoutingException>(() => _client.SolveVrp(new VrpRequest(two, Customers)
        {
            Vehicles = [new(Capacity: 5, Depot: 2)],
        }));
        Assert.Equal("parameter_out_of_range", y.ErrorCode);
        Assert.Equal("depot", y.Parameter);
        Assert.Equal(0, y.Index);
    }

    [Fact]
    public void A_setting_the_method_does_not_read_is_refused()
    {
        var x = Assert.Throws<RoutingException>(() => _client.SolveVrp(new VrpRequest([new(0, 0)], Customers)
        {
            Method = RoutingMethod.NearestNeighbor,
            Config = new VrpConfig { PopulationSize = 10 },
        }));
        Assert.Equal("invalid_option", x.ErrorCode);
        Assert.Equal("config.population_size", x.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal("nn", x.Details!.Value.GetProperty("method").GetString());
    }

    [Fact]
    public void A_non_finite_value_is_refused_where_it_sits()
    {
        var x = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 1, 1), new(2, double.NaN, 1)])));
        Assert.Equal("value_not_finite", x.ErrorCode);
        Assert.Equal("customers[1].x", x.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal("customers[1].x", x.Parameter);
        Assert.Null(x.Index);

        var window = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 1, 1, TimeWindow: new(0, double.PositiveInfinity))])));
        Assert.Equal("customers[0].time_window", window.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(1, window.Details!.Value.GetProperty("index").GetInt32());
        Assert.Equal(1, window.Index);
    }

    [Fact]
    public void Given_distances_replace_the_coordinates()
    {
        // Two customers on the depot: by coordinates every length is 0, by the matrix the
        // round trip is 4 + 3 + 6.
        var request = new VrpRequest([new(0, 0)], [new(1, 0, 0, Demand: 1), new(2, 0, 0, Demand: 1)])
        {
            Distances = [[0, 4, 6], [4, 0, 3], [6, 3, 0]],
        };
        var solution = _client.SolveVrp(request);
        Assert.Equal(13.0, solution.TotalDistance, 9);
        Assert.True(_client.CheckSolution(request, VrpPlan.Of(solution)).Feasible);

        // One way round is 4 + 3 + 6, the other 6 + 5 + 4: the plan takes the short one.
        var oneWay = _client.SolveVrp(request with { Distances = [[0, 4, 6], [4, 0, 3], [6, 5, 0]], Method = RoutingMethod.Savings });
        Assert.Equal(13.0, oneWay.TotalDistance, 9);

        // Twice as long in time: the route's duration, not its distance, doubles.
        var slow = _client.SolveVrp(request with { Durations = [[0, 8, 12], [8, 0, 6], [12, 6, 0]] });
        Assert.Equal(13.0, slow.TotalDistance, 9);
        Assert.Equal(26.0, slow.RouteSummaries.Sum(r => r.Duration), 9);

        var nan = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(request with { Distances = [[0, 4, 6], [4, 0, double.NaN], [6, 3, 0]] }));
        Assert.Equal("value_not_finite", nan.ErrorCode);
        Assert.Equal("distances[1]", nan.Parameter);
        Assert.Equal(2, nan.Index);
    }

    [Fact]
    public void The_engine_still_names_what_it_refuses()
    {
        var e = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest([new(0, 0)], [new(1, 1, 1), new(1, 2, 2)])));
        Assert.Equal("duplicate_id", e.ErrorCode);
        Assert.Equal(1, e.Details!.Value.GetProperty("id").GetInt64());
    }

    [Fact]
    public void A_missing_key_fails_rather_than_reading_as_zero()
        => Assert.ThrowsAny<JsonException>(() => JsonSerializer.Deserialize(
            """{"routes": [], "total_distance": 0, "num_vehicles": 0, "method_used": "nn", "unassigned": []}""",
            RoutingJson.Default.VrpSolution));

    [Fact]
    public void The_suite_runs_as_a_trimmed_host_does()
    {
        Assert.False(JsonSerializer.IsReflectionEnabledByDefault);
        Assert.Throws<InvalidOperationException>(() => JsonSerializer.Serialize(new { depot_x = 0.0 }, new JsonSerializerOptions()));
    }
}
