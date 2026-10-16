using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using URouting.Interop;

namespace URouting;

/// <summary>
/// Calls into the u-routing native engine. Each call is one request in and one
/// result or <see cref="RoutingException"/> out; the instance holds no
/// state between calls and no native resources, so disposing it releases nothing;
/// it is <see cref="IDisposable"/> so it can sit in a <c>using</c> block.
/// </summary>
public sealed class RoutingClient : IDisposable
{
    private bool _disposed;

    /// <summary>Sees each raw response body before it is read — the contract tests compare
    /// the two.</summary>
    internal Action<string>? ResponseObserver { get; set; }

    /// <summary>
    /// Sees each request body as it is sent, with the entry point it goes to — the contract
    /// tests check every key and name in it against the ones the engine reads.
    /// </summary>
    internal Action<string, string>? RequestObserver { get; set; }

    /// <summary>
    /// The version of the native engine — the u-routing Rust crate the package
    /// carries — not the URouting package version.
    /// </summary>
    public string GetVersion()
    {
        var ptr = NativeInterop.urouting_version();
        var version = Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        NativeInterop.urouting_free_string(ptr);
        return version;
    }

    /// <summary>
    /// Every method and the <see cref="VrpConfig"/> settings each reads — the table the
    /// engine's refusal of a setting the method does not read comes from, so a form can show
    /// only the settings that change the chosen method's run.
    /// </summary>
    public IReadOnlyList<MethodInfo> GetMethods() =>
        JsonSerializer.Deserialize(MethodsJson(), RoutingJson.Default.IReadOnlyListMethodInfo)
        ?? throw new InvalidOperationException("The native engine returned no method list.");

    /// <summary>
    /// Every refusal <see cref="RoutingException.ErrorCode"/> the engine and this client raise,
    /// and every <see cref="PlanViolation"/> code <see cref="CheckSolution"/> reports, with the
    /// fields beside each — built from the tables the refusals and violations are made from, so
    /// a caller that translates codes can check its list against the engine's.
    /// </summary>
    public Codes GetCodes() =>
        JsonSerializer.Deserialize(CodesJson(), RoutingJson.Default.Codes)
        ?? throw new InvalidOperationException("The native engine returned no code catalog.");

    /// <summary>The code catalog as the native engine returns it.</summary>
    internal static string CodesJson()
    {
        var ptr = NativeInterop.urouting_codes();
        try
        {
            return Marshal.PtrToStringUTF8(ptr)
                ?? throw new InvalidOperationException("The native engine returned no code catalog.");
        }
        finally
        {
            NativeInterop.urouting_free_string(ptr);
        }
    }

    /// <summary>The method list as the native engine returns it.</summary>
    internal static string MethodsJson()
    {
        var ptr = NativeInterop.urouting_methods();
        try
        {
            return Marshal.PtrToStringUTF8(ptr)
                ?? throw new InvalidOperationException("The native engine returned no method list.");
        }
        finally
        {
            NativeInterop.urouting_free_string(ptr);
        }
    }

    /// <summary>
    /// Solves a vehicle routing problem. Every method keeps to capacities, time windows, each
    /// vehicle's distance and duration limits and the fixed fleet; customers it cannot serve
    /// within them are in <see cref="VrpSolution.Unassigned"/>.
    /// </summary>
    /// <exception cref="RoutingException">The engine refused the request.</exception>
    public VrpSolution SolveVrp(VrpRequest request)
    {
        var json = Body(request).ToJsonString();
        RequestObserver?.Invoke(nameof(NativeInterop.urouting_solve_vrp), json);
        var body = Invoke(NativeInterop.urouting_solve_vrp, json);
        ResponseObserver?.Invoke(body);
        return JsonSerializer.Deserialize(body, RoutingJson.Default.VrpSolution)
               ?? throw new RoutingException(-4, "The engine returned null.");
    }

    /// <summary>
    /// Checks a plan against the problem it was solved for, read as <see cref="SolveVrp"/>
    /// reads it: every customer served once or reported unassigned, each route within its
    /// vehicle's capacity, time windows, <see cref="Vehicle.MaxDistance"/> and
    /// <see cref="Vehicle.MaxDuration"/>, <see cref="VrpConfig.MaxVehicles"/> and (for
    /// nearest neighbour) one route per vehicle, and the reported total distance. A plan that
    /// breaks the problem is a result, not an exception.
    /// </summary>
    /// <exception cref="RoutingException">The engine refused the problem, or the plan's
    /// <see cref="VrpPlan.RouteVehicles"/> does not have one entry per route.</exception>
    public PlanCheck CheckSolution(VrpRequest request, VrpPlan plan)
    {
        var routes = new JsonArray();
        foreach (var route in plan.Routes)
            routes.Add((JsonNode)new JsonArray(route.Select(id => (JsonNode)JsonValue.Create(id)).ToArray()));
        var solution = new JsonObject
        {
            ["routes"] = routes,
            ["unassigned"] = new JsonArray(plan.Unassigned.Select(id => (JsonNode)JsonValue.Create(id)).ToArray()),
        };
        if (plan.RouteVehicles is { } vehicles)
            solution["route_vehicles"] = new JsonArray(vehicles.Select(v => (JsonNode)JsonValue.Create(v)).ToArray());
        if (plan.TotalDistance is { } total)
            solution["total_distance"] = Num("solution.total_distance", total);
        var json = new JsonObject { ["problem"] = Body(request), ["solution"] = solution }.ToJsonString();
        RequestObserver?.Invoke(nameof(NativeInterop.urouting_check_solution), json);
        var body = Invoke(NativeInterop.urouting_check_solution, json);
        ResponseObserver?.Invoke(body);
        return JsonSerializer.Deserialize(body, RoutingJson.Default.PlanCheck)
               ?? throw new RoutingException(-4, "The engine returned null.");
    }

    private static JsonObject Body(VrpRequest request)
    {
        var customers = new JsonArray();
        for (var i = 0; i < request.Customers.Count; i++)
        {
            var c = request.Customers[i];
            var at = $"customers[{i}]";
            var customer = new JsonObject
            {
                ["id"] = c.Id,
                ["x"] = Num($"{at}.x", c.X),
                ["y"] = Num($"{at}.y", c.Y),
                ["demand"] = Num($"{at}.demand", c.Demand),
                ["service_time"] = Num($"{at}.service_time", c.ServiceTime),
            };
            if (c.TimeWindow is { } w)
                customer["time_window"] = new JsonArray(Num($"{at}.time_window", w.Ready, 0), Num($"{at}.time_window", w.Due, 1));
            customers.Add((JsonNode)customer);
        }

        var vehicles = new JsonArray();
        for (var i = 0; i < request.Vehicles.Count; i++)
        {
            var v = request.Vehicles[i];
            var at = $"vehicles[{i}]";
            var vehicle = new JsonObject();
            Put(vehicle, "capacity", v.Capacity, $"{at}.capacity");
            Put(vehicle, "max_distance", v.MaxDistance, $"{at}.max_distance");
            Put(vehicle, "max_duration", v.MaxDuration, $"{at}.max_duration");
            vehicles.Add((JsonNode)vehicle);
        }

        var body = new JsonObject
        {
            ["depot_x"] = Num("depot_x", request.DepotX),
            ["depot_y"] = Num("depot_y", request.DepotY),
            ["customers"] = customers,
            ["vehicles"] = vehicles,
            ["method"] = JsonSerializer.SerializeToNode(request.Method, RoutingJson.Default.RoutingMethod),
        };
        if (request.Config is { } c2)
        {
            var config = new JsonObject();
            Put(config, "population_size", c2.PopulationSize);
            Put(config, "max_generations", c2.MaxGenerations);
            Put(config, "mutation_rate", c2.MutationRate, "config.mutation_rate");
            Put(config, "elite_ratio", c2.EliteRatio, "config.elite_ratio");
            Put(config, "max_iterations", c2.MaxIterations);
            if (c2.Seed is { } seed)
                config["seed"] = seed;
            if (c2.TimeLimitMs is { } timeLimit)
                config["time_limit_ms"] = timeLimit;
            Put(config, "max_vehicles", c2.MaxVehicles);
            body["config"] = config;
        }
        return body;
    }

    private static void Put(JsonObject o, string key, double? value, string parameter)
    {
        if (value is { } v)
            o[key] = Num(parameter, v);
    }

    private static void Put(JsonObject o, string key, int? value)
    {
        if (value is { } v)
            o[key] = v;
    }

    /// <summary>
    /// <paramref name="value"/> if it is finite. JSON has no NaN or infinity, so such a value
    /// cannot reach the engine; it is refused here with the engine's own reason and fields —
    /// <c>parameter</c> the path to it, <c>index</c> its position when it sits in an array.
    /// </summary>
    private static JsonNode Num(string parameter, double value, int? index = null)
    {
        if (double.IsFinite(value))
            return JsonValue.Create(value);
        var where = index is { } i ? $"{parameter}[{i}]" : parameter;
        var error = new JsonObject
        {
            ["error"] = $"{where}: expected a finite number, got {value.ToString(CultureInfo.InvariantCulture)}",
            ["code"] = "value_not_finite",
            ["parameter"] = parameter,
            ["index"] = index,
        };
        throw RoutingException.FromErrorBody(-3, error.ToJsonString());
    }

    private delegate int Entry(string requestJson, out IntPtr resultPtr);

    private static string Invoke(Entry entry, string requestJson)
    {
        var code = entry(requestJson, out var resultPtr);

        try
        {
            if (resultPtr == IntPtr.Zero)
                throw new RoutingException(code, "Null result from engine");

            var resultJson = Marshal.PtrToStringUTF8(resultPtr);
            if (string.IsNullOrEmpty(resultJson))
                throw new RoutingException(code, "Empty result from engine");

            if (code != 0)
                throw RoutingException.FromErrorBody(code, resultJson);

            return resultJson;
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
                NativeInterop.urouting_free_string(resultPtr);
        }
    }

    /// <summary>Marks the client disposed; it holds no native resources to release.</summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}

/// <summary>
/// A request the engine refused. <see cref="Exception.Message"/> is readable text; <see
/// cref="ErrorCode"/>,
/// <see cref="Parameter"/>, <see cref="Index"/> and <see cref="Details"/> are for programs —
/// the same members, with the same meaning, on every sibling package's exception.
/// </summary>
public class RoutingException : Exception
{
    /// <summary>Native status: -1 null pointer, -2 malformed request, -3 refused request,
    /// -4 internal panic.</summary>
    public int Status { get; }

    /// <summary>
    /// Stable, machine-readable reason: the <c>code</c> of the engine's error body, the same
    /// name the WebAssembly binding puts on its <c>Error</c> — <c>unknown_option</c>,
    /// <c>duplicate_id</c>, <c>invalid_time_window</c>, <c>not_whole_units</c>,
    /// <c>mixed_fleet</c>, <c>parameter_out_of_range</c>, <c>invalid_option</c>,
    /// <c>value_not_finite</c>, <c>malformed_input</c>, <c>internal</c>, ….
    /// <c>null</c> when the engine returned no readable body.
    /// </summary>
    public string? ErrorCode { get; }

    /// <summary>
    /// The argument or option the refusal is about, as a path into the request
    /// (<c>config.mutation_rate</c>, <c>customers[0].x</c>, …) — the body's
    /// <c>parameter</c>. <c>null</c> when the refusal is not
    /// about one value.
    /// </summary>
    public string? Parameter { get; }

    /// <summary>
    /// Zero-based position of the offending element in the array <see cref="Parameter"/>
    /// names — the body's <c>index</c>. <c>null</c> when the value is not an array element.
    /// </summary>
    public int? Index { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c> and the values behind the reason
    /// (<c>id</c>, <c>index</c>, <c>parameter</c>, <c>min</c>, <c>max</c>,
    /// <c>exclusive_min</c>, <c>exclusive_max</c>, <c>got</c>, <c>expected</c>). For
    /// <c>parameter_out_of_range</c>, <c>min</c> / <c>max</c> are the closed
    /// (included) bounds — <c>null</c> on a side that is open or unbounded — and
    /// <c>exclusive_min</c> / <c>exclusive_max</c> the open (excluded) ones, present only on
    /// an open side: a <c>mutation_rate</c> in (0, 1] is refused with
    /// <c>exclusive_min</c> 0 and <c>max</c> 1. <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

    /// <summary>A refusal with no readable body.</summary>
    /// <param name="status">Native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    public RoutingException(int status, string message) : base(message)
    {
        Status = status;
    }

    /// <summary>
    /// A refusal with the engine's reason and the values behind it; <see cref="Parameter"/>
    /// and <see cref="Index"/> are read from <paramref name="details"/>.
    /// </summary>
    /// <param name="status">Native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    /// <param name="errorCode">The body's <c>code</c>.</param>
    /// <param name="details">The error body.</param>
    public RoutingException(int status, string message, string? errorCode, JsonElement? details) : base(message)
    {
        Status = status;
        ErrorCode = errorCode;
        Details = details;
        Parameter = ParameterIn(details);
        Index = IndexIn(details);
    }

    /// <summary>Reads the engine's <c>{"error", "code", ...}</c> error body.</summary>
    internal static RoutingException FromErrorBody(int status, string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            var message = root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : body;
            string? errorCode = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            return new RoutingException(status, message, errorCode, root.Clone());
        }
        catch (JsonException)
        {
            return new RoutingException(status, body);
        }
    }

    /// <summary>The body's <c>parameter</c>, when it is a string.</summary>
    private static string? ParameterIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("parameter", out var p) && p.ValueKind == JsonValueKind.String
            ? p.GetString()
            : null;

    /// <summary>The body's <c>index</c>, when it is a whole number.</summary>
    private static int? IndexIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number
        && i.TryGetInt32(out var n)
            ? n
            : null;
}
