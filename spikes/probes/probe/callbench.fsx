// Micro-benchmark: cost per native call from F# via (a) DllImport, (b) delegate from GetDelegateForFunctionPointer,
// (c) DynamicMethod + EmitCalli (unmanaged stdcall), using kernel32!GetCurrentProcessId (trivial native function).
#nowarn "9"
open System
open System.Diagnostics
open System.Reflection.Emit
open System.Runtime.InteropServices

module K =
    [<DllImport("kernel32.dll")>]
    extern uint32 GetCurrentProcessId()

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type D_U32 = delegate of unit -> uint32

let lib = NativeLibrary.Load "kernel32.dll"
let fp = NativeLibrary.GetExport(lib, "GetCurrentProcessId")
let del = Marshal.GetDelegateForFunctionPointer<D_U32> fp

// calli thunk: Func<nativeint, uint32>
let dm = DynamicMethod("calli_u32", typeof<uint32>, [| typeof<nativeint> |], typeof<D_U32>.Module, true)
let il = dm.GetILGenerator()
il.Emit(OpCodes.Ldarg_0)
il.EmitCalli(OpCodes.Calli, CallingConvention.Winapi, typeof<uint32>, Type.EmptyTypes)
il.Emit(OpCodes.Ret)
let calli = dm.CreateDelegate(typeof<Func<nativeint, uint32>>) :?> Func<nativeint, uint32>

let N = 20_000_000
let bench name (f: unit -> uint32) =
    let mutable acc = 0u
    for _ in 1 .. 1_000_000 do acc <- acc + f ()
    let sw = Stopwatch.StartNew()
    for _ in 1 .. N do acc <- acc + f ()
    sw.Stop()
    printfn "%-28s %6.2f ns/call  (acc %d)" name (float sw.Elapsed.TotalMilliseconds * 1e6 / float N) (acc % 7u)

bench "DllImport" (fun () -> K.GetCurrentProcessId())
bench "GetDelegateForFunctionPtr" (fun () -> del.Invoke())
bench "DynamicMethod calli" (fun () -> calli.Invoke fp)
