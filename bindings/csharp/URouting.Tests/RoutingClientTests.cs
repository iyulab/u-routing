using System.Text.Json;
using System.Text.Json.Nodes;
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
            Config = new VrpConfig { Seed = 7, MaxGenerations = 20, MaxIterations = 50 },
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
    public void A_non_finite_value_is_refused_where_it_sits()
    {
        var x = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest(0, 0, [new(1, 1, 1), new(2, double.NaN, 1)])));
        Assert.Equal("value_not_finite", x.Reason);
        Assert.Equal("customers[1].x", x.Details!.Value.GetProperty("parameter").GetString());

        var window = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest(0, 0, [new(1, 1, 1, TimeWindow: new(0, double.PositiveInfinity))])));
        Assert.Equal("customers[0].time_window", window.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(1, window.Details!.Value.GetProperty("index").GetInt32());
    }

    [Fact]
    public void The_engine_still_names_what_it_refuses()
    {
        var e = Assert.Throws<RoutingException>(() =>
            _client.SolveVrp(new VrpRequest(0, 0, [new(1, 1, 1), new(1, 2, 2)])));
        Assert.Equal("duplicate_id", e.Reason);
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
