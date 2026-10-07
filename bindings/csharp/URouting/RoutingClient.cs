using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using URouting.Interop;

namespace URouting;

public sealed class RoutingClient : IDisposable
{
    private bool _disposed;

    /// <summary>Sees each raw response body before it is read — the contract tests compare the two.</summary>
    internal Action<string>? ResponseObserver { get; set; }

    public string GetVersion()
    {
        var ptr = NativeInterop.urouting_version();
        var version = Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        NativeInterop.urouting_free_string(ptr);
        return version;
    }

    /// <summary>
    /// Solves a vehicle routing problem. Every method keeps to capacities, time windows, each
    /// vehicle's distance and duration limits and the fixed fleet; customers it cannot serve
    /// within them are in <see cref="VrpSolution.Unassigned"/>.
    /// </summary>
    /// <exception cref="RoutingException">The engine refused the request.</exception>
    public VrpSolution SolveVrp(VrpRequest request)
    {
        var body = Invoke(Body(request).ToJsonString());
        ResponseObserver?.Invoke(body);
        return JsonSerializer.Deserialize(body, RoutingJson.Default.VrpSolution)
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

    private static string Invoke(string requestJson)
    {
        var code = NativeInterop.urouting_solve_vrp(requestJson, out var resultPtr);

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
/// A request the engine refused. <see cref="Exception.Message"/> is human-readable;
/// <see cref="Reason"/> and <see cref="Details"/> are for programs.
/// </summary>
public class RoutingException : Exception
{
    /// <summary>Native status: -1 null pointer, -2 malformed request, -3 refused request, -4 internal panic.</summary>
    public int Code { get; }

    /// <summary>
    /// Stable, machine-readable reason, e.g. <c>unknown_option</c>, <c>duplicate_id</c>,
    /// <c>invalid_time_window</c>, <c>not_whole_units</c>, <c>mixed_fleet</c>,
    /// <c>invalid_option</c>, <c>malformed_input</c>,
    /// <c>internal</c>. <c>null</c> when the engine returned no readable body.
    /// </summary>
    public string? Reason { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c> and the values behind the reason
    /// (<c>id</c>, <c>index</c>, <c>parameter</c>, ...). <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

    public RoutingException(int code, string message) : base(message)
    {
        Code = code;
    }

    public RoutingException(int code, string message, string? reason, JsonElement? details) : base(message)
    {
        Code = code;
        Reason = reason;
        Details = details;
    }

    /// <summary>Reads the engine's <c>{"error", "code", ...}</c> error body.</summary>
    internal static RoutingException FromErrorBody(int code, string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            var message = root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : body;
            string? reason = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            return new RoutingException(code, message, reason, root.Clone());
        }
        catch (JsonException)
        {
            return new RoutingException(code, body);
        }
    }
}
