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
                throw new RoutingException(code, resultJson);

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

public class RoutingException : Exception
{
    public int Code { get; }
    public RoutingException(int code, string message) : base(message) { Code = code; }
}
