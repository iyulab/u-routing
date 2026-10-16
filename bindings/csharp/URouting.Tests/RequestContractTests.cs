using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using URouting.Interop;
using Xunit;

namespace URouting.Tests;

/// <summary>
/// The requests the client writes and the ones the engine reads are the same, both ways, against
/// the real native library with reflection-based serialization disabled: every key and name the
/// client can write is one the entry point reads (<c>request_keys.json</c>, which the engine's
/// tests measure from the entry point), every one the entry point reads can be written, and a
/// request with every setting given runs under each method.
/// </summary>
public class RequestContractTests
{
    /// <summary>Places of a request the client does not write, each with why.</summary>
    private static readonly Dictionary<string, string> NotWritten = [];

    private static Vehicle Vehicle() =>
        new(Capacity: 30, MaxDistance: 1000, MaxDuration: 1000, FixedCost: 50, CostPerDistance: 2);

    private static VrpRequest Full(RoutingMethod method) => new(0, 0,
        [
            new Customer(1, 10, 5, Demand: 10, ServiceTime: 1, TimeWindow: new(0, 500)),
            new Customer(2, -4, 8, Demand: 15, ServiceTime: 1, TimeWindow: new(0, 500)),
            new Customer(3, 6, -9, Demand: 20, ServiceTime: 2, TimeWindow: new(0, 500)),
            new Customer(4, -7, -6, Demand: 5, ServiceTime: 2),
        ])
    {
        Vehicles = method == RoutingMethod.NearestNeighbor
            ? [Vehicle(), Vehicle()]
            : [Vehicle()],
        Method = method,
        // Every setting the method reads -- one it does not read is refused, so the
        // settings are spread over the methods and every one is written by some run.
        Config = method switch
        {
            RoutingMethod.Genetic => new VrpConfig
            {
                PopulationSize = 10,
                MaxGenerations = 5,
                MutationRate = 0.1,
                EliteRatio = 0.2,
                Seed = 7,
                MaxVehicles = 3,
            },
            RoutingMethod.Alns => new VrpConfig { MaxIterations = 20, Seed = 7, TimeLimitMs = 60_000, MaxVehicles = 3 },
            _ => new VrpConfig { MaxVehicles = 3 },
        },
    };

    public static TheoryData<RoutingMethod> Methods() => [.. Enum.GetValues<RoutingMethod>()];

    /// <summary>The request under each method: (entry point, body). A refused one is still written.</summary>
    private static readonly Lazy<List<(string Entry, string Body)>> Written = new(() =>
    {
        var written = new List<(string, string)>();
        foreach (var method in Enum.GetValues<RoutingMethod>())
        {
            using var client = new RoutingClient();
            client.RequestObserver = (entry, body) => written.Add((entry, body));
            try
            {
                // The solve, then its plan checked against the same request -- with every
                // key of the plan given.
                var solution = client.SolveVrp(Full(method));
                client.CheckSolution(Full(method), VrpPlan.Of(solution));
            }
            catch (RoutingException)
            {
            }
        }
        return written;
    });

    [Theory]
    [MemberData(nameof(Methods))]
    public void A_request_with_every_setting_is_accepted_and_solved(RoutingMethod method)
    {
        using var client = new RoutingClient();
        var solution = client.SolveVrp(Full(method));
        Assert.Equal(method, solution.MethodUsed);
        Assert.Equal(4, solution.Routes.Sum(r => r.Count) + solution.Unassigned.Count);
    }

    [Fact]
    public void Every_key_and_name_the_client_writes_is_one_the_engine_reads()
    {
        var problems = new List<string>();
        foreach (var (entry, body) in Written.Value)
        {
            var found = new List<string>();
            RequestKeys.Read(RequestKeys.Schema(entry), JsonNode.Parse(body), "$", [], found);
            problems.AddRange(found.Select(p => $"{entry} {p}"));
        }
        Assert.True(problems.Count == 0, string.Join("\n", problems.Distinct()));
    }

    [Fact]
    public void Every_key_and_name_the_engine_reads_is_one_the_client_writes()
    {
        var written = new HashSet<string>();
        foreach (var (entry, body) in Written.Value)
        {
            var places = new HashSet<string>();
            RequestKeys.Read(RequestKeys.Schema(entry), JsonNode.Parse(body), "$", places, []);
            written.UnionWith(places.Select(p => $"{entry} {p}"));
        }
        var engine = RequestKeys.EntryPoints
            .SelectMany(e => RequestKeys.Places(e.Key).Select(p => $"{e.Key} {p}"))
            .ToHashSet();

        var unwritable = engine.Except(written).Except(NotWritten.Keys).Order(StringComparer.Ordinal).ToList();
        Assert.True(unwritable.Count == 0, $"the engine reads what the client cannot write:\n{string.Join("\n", unwritable)}");

        var stale = NotWritten.Keys.Where(p => written.Contains(p) || !engine.Contains(p)).ToList();
        Assert.True(stale.Count == 0, $"stale reasons: {string.Join(", ", stale)}");
    }

    [Fact]
    public void Every_entry_point_the_engine_measures_is_bound()
    {
        var bound = typeof(NativeInterop)
            .GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Select(m => m.Name)
            .Where(n => n is not ("urouting_version" or "urouting_methods" or "urouting_codes" or "urouting_free_string"))
            .Order(StringComparer.Ordinal);
        Assert.Equal(RequestKeys.EntryPoints.Select(e => e.Key).Order(StringComparer.Ordinal), bound);
    }

    [Fact]
    public void Each_method_is_a_name_the_engine_reads()
    {
        var names = Enum.GetValues<RoutingMethod>()
            .Select(m => JsonSerializer.SerializeToNode(m, RoutingJson.Default.RoutingMethod)!.GetValue<string>())
            .Order(StringComparer.Ordinal);
        Assert.Equal(RequestKeys.Names("urouting_solve_vrp", "$.method"), names);
    }
}
