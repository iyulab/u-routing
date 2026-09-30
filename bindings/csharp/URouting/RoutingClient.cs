using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using URouting.Interop;

namespace URouting;

public sealed class RoutingClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private bool _disposed;

    public string GetVersion()
    {
        var ptr = NativeInterop.urouting_version();
        var version = Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        NativeInterop.urouting_free_string(ptr);
        return version;
    }

    public JsonElement SolveVrp(object request)
    {
        var requestJson = JsonSerializer.Serialize(request, JsonOptions);
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

            return JsonDocument.Parse(resultJson).RootElement.Clone();
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
    /// <c>routes_exceed_max_vehicles</c>, <c>invalid_option</c>, <c>malformed_input</c>,
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
