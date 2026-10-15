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
        var request = new VrpRequest(0, 0, Customers) { Vehicles = [new(30), new(30)] };
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
        var request = new VrpRequest(0, 0, Customers)
        {
            Vehicles = method == RoutingMethod.NearestNeighbor ? [new(30), new(30)] : [new(30)],
            Method = method,
            Config = method switch
            {
                RoutingMethod.Genetic => new VrpConfig { Seed = 7, MaxGenerations = 20 },
                RoutingMethod.Alns => new VrpConfig { Seed = 7, MaxIterations = 50 },
                _ => null,
            },
        };

        var solution = _client.SolveVrp(request);

        Assert.Equal(method, solution.MethodUsed);
        var written = JsonSerializer.SerializeToNode(solution, RoutingJson.Default.VrpSolution);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw!), written), $"engine: {raw}\nrecord: {written}");
        var served = solution.Routes.SelectMany(r => r).Concat(solution.Unassigned).Order();
        Assert.Equal(Customers.Select(c => c.Id).Order(), served);
    }

    [Fact]
    public void Limits_and_windows_reach_the_engine()
    {
        // The far customer cannot be reached within the vehicle's distance limit.
        Customer[] customers = [new(1, 3, 4), new(2, 300, 400)];
        var solution = _client.SolveVrp(new VrpRequest(0, 0, customers)
        {
            Vehicles = [new(MaxDistance: 50)],
            Method = RoutingMethod.Savings,
        });
        Assert.Equal([2L], solution.Unassigned);

        // A window that closes before the vehicle can arrive.
        var late = _client.SolveVrp(new VrpRequest(0, 0, [new(1, 30, 40, TimeWindow: new(0, 10))]));
        Assert.Equal([1L], late.Unassigned);
    }

    [Fact]
    public void A_fixed_fleet_is_kept()
    {
        var solution = _client.SolveVrp(new VrpRequest(0, 0, Customers)
        {
            Vehicles = [new(20)],
            Method = RoutingMethod.Savings,
            Config = new VrpConfig { MaxVehicles = 2 },
        });
        Assert.True(solution.NumVehicles <= 2);
        Assert.NotEmpty(solution.Unassigned);
    }

    [Fact]
    public void A_setting_the_method_does_not_read_is_refused()
    {
        var x = Assert.Throws<RoutingException>(() => _client.SolveVrp(new VrpRequest(0, 0, Customers)
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
            _client.SolveVrp(new VrpRequest(0, 0, [new(1, 1, 1), new(2, double.NaN, 1)])));
        Assert.Equal("value_not_finite", x.ErrorCode);
        Assert.Equal("customers[1].x", x.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal("customers[1].x", x.Parameter);
        Assert.Null(x.Index);

        var window = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest(0, 0, [new(1, 1, 1, TimeWindow: new(0, double.PositiveInfinity))])));
        Assert.Equal("customers[0].time_window", window.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(1, window.Details!.Value.GetProperty("index").GetInt32());
        Assert.Equal(1, window.Index);
    }

    [Fact]
    public void The_engine_still_names_what_it_refuses()
    {
        var e = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest(0, 0, [new(1, 1, 1), new(1, 2, 2)])));
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
