/// D1's backend writes `backend = d3d12` on every record it makes (it has one ExecutionBackend, D3D12). On WARP those
/// records belong to the d3d12-warp class, so the spike's backend wrapper makes this facade ambient around every call
/// into D1's code: it forwards each record unchanged except that a `backend` attribute naming d3d12 names d3d12-warp.
/// Values, names, units and every other attribute pass as they are (TelemetryFacade: telemetry changes no value).
module Lrm.Spikes.D3D12.Relabel

open Lrm.Core

/// The backend name D1 writes.
let From = ExecutionBackend.name ExecutionBackend.D3D12

/// The backend name the records get.
let To = ExecutionBackend.name ExecutionBackend.D3D12Warp

/// One attribute relabelled: `backend = d3d12` becomes `backend = d3d12-warp`; every other attribute is unchanged.
let attr (a: Attr) : Attr =
    match a.Value with
    | AttrValue.Str s when a.Key = TelemetryKeys.Backend && s = From -> Attr.str TelemetryKeys.Backend To
    | _ -> a

let attrs (xs: Attr list) =
    if xs |> List.exists (fun a -> a.Key = TelemetryKeys.Backend) then
        List.map attr xs
    else
        xs

[<Sealed>]
type private RelabelSpan(inner: ISpan) =
    interface ISpan with
        member _.Id = inner.Id
        member _.Add(a: Attr) = inner.Add(attr a)
        member _.Dispose() = inner.Dispose()

/// The facade over `inner`.
[<Sealed>]
type RelabelTelemetry(inner: ITelemetry) =
    member _.Inner = inner

    interface ITelemetry with
        member _.Enabled = inner.Enabled

        member _.Span(name, xs) =
            new RelabelSpan(inner.Span(name, attrs xs)) :> ISpan

        member _.Counter(name, delta, unit, xs) = inner.Counter(name, delta, unit, attrs xs)
        member _.Gauge(name, value, unit, xs) = inner.Gauge(name, value, unit, attrs xs)
        member _.Histogram(name, value, unit, xs) = inner.Histogram(name, value, unit, attrs xs)
        member _.Event(name, xs) = inner.Event(name, attrs xs)

        member _.GpuInterval(name, start, duration, xs) =
            inner.GpuInterval(name, start, duration, attrs xs)

        member _.Meter(name, amount, unit, xs) = inner.Meter(name, amount, unit, attrs xs)
        member _.Check(name, passed, xs) = inner.Check(name, passed, attrs xs)

        member _.Child(xs) =
            RelabelTelemetry(inner.Child(attrs xs)) :> ITelemetry

/// Makes the relabelling facade over the current one ambient until the result is disposed (a no-op facade stays a
/// no-op, and a facade that already relabels is not wrapped twice).
let install () : System.IDisposable =
    match Telemetry.current () with
    | :? RelabelTelemetry as t -> Telemetry.install t
    | t when not t.Enabled -> Telemetry.install t
    | t -> Telemetry.install (RelabelTelemetry t)
