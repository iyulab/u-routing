using System.Runtime.InteropServices;

namespace URouting.Interop;

internal static partial class NativeInterop
{
    private const string DllName = "u_routing";

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int urouting_solve_vrp(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName)]
    public static partial void urouting_free_string(IntPtr ptr);

    [LibraryImport(DllName)]
    public static partial IntPtr urouting_version();
}
