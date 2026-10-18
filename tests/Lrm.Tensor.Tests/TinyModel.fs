/// A test-only composition of CpuRef ops into the tiny Qwen3 forward, to check the ops in the chain the model runs
/// (M2a owns the real forward, Lrm.Models.Qwen). It follows ActivationNames and records every tap.
module Lrm.Tensor.Tests.TinyModel

open System
open System.Collections.Generic
open System.Numerics
open System.Text.Json
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Tensor.Tests.Common

type Config =
    {
        Layers: int
        Hidden: int
        Heads: int
        KvHeads: int
        HeadDim: int
        Intermediate: int
        Vocab: int
        Eps: float
        MaxPos: int
    }

type LayerWeights =
    {
        InputNorm: HostTensor<uint16>
        Q: HostTensor<uint16>
        K: HostTensor<uint16>
        V: HostTensor<uint16>
        O: HostTensor<uint16>
        QNorm: HostTensor<uint16>
        KNorm: HostTensor<uint16>
        PostNorm: HostTensor<uint16>
        Gate: HostTensor<uint16>
        Up: HostTensor<uint16>
        Down: HostTensor<uint16>
    }

type Weights =
    {
        Embed: HostTensor<uint16>
        Layers: LayerWeights[]
        Norm: HostTensor<uint16>
    }

type Model =
    {
        Config: Config
        Weights: Weights
        Rope: RopeTable
        /// The pinned inverse frequencies from rope.safetensors.
        InvFreqBits: uint32[]
    }

let private loadModel () =
    let set = Common.tiny.Force()
    let config = JsonDocument.Parse(set.ReadAllBytes "model/config.json").RootElement
    let int' (k: string) = config.GetProperty(k).GetInt32()

    let cfg =
        {
            Layers = int' "num_hidden_layers"
            Hidden = int' "hidden_size"
            Heads = int' "num_attention_heads"
            KvHeads = int' "num_key_value_heads"
            HeadDim = int' "head_dim"
            Intermediate = int' "intermediate_size"
            Vocab = int' "vocab_size"
            Eps = config.GetProperty("rms_norm_eps").GetDouble()
            MaxPos = int' "max_position_embeddings"
        }

    use st = set.OpenSafetensors "model/model.safetensors"
    let w (name: string) = st.ReadBf16 name

    let layer i =
        let p = Invariant.format $"model.layers.{i}."

        {
            InputNorm = w (p + "input_layernorm.weight")
            Q = w (p + "self_attn.q_proj.weight")
            K = w (p + "self_attn.k_proj.weight")
            V = w (p + "self_attn.v_proj.weight")
            O = w (p + "self_attn.o_proj.weight")
            QNorm = w (p + "self_attn.q_norm.weight")
            KNorm = w (p + "self_attn.k_norm.weight")
            PostNorm = w (p + "post_attention_layernorm.weight")
            Gate = w (p + "mlp.gate_proj.weight")
            Up = w (p + "mlp.up_proj.weight")
            Down = w (p + "mlp.down_proj.weight")
        }

    let weights =
        {
            Embed = w "model.embed_tokens.weight"
            Layers = Array.init cfg.Layers layer
            Norm = w "model.norm.weight"
        }

    use rope = set.OpenSafetensors "rope.safetensors"
    let bits = (rope.ReadU32 "inv_freq.bits").Data

    {
        Config = cfg
        Weights = weights
        Rope = CpuRef.ropeTable bits cfg.MaxPos
        InvFreqBits = bits
    }

/// The tiny model, loaded once (weights and RoPE table verified against the manifest).
let model = lazy (loadModel ())

/// K (after norm and RoPE) and V (raw) per layer, [MaxPos, KvHeads * HeadDim] in the mode's dtype.
type Cache<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType> =
    {
        Keys: HostTensor<'T>[]
        Values: HostTensor<'T>[]
    }

let newCache<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (m: Model)
    : Cache<'T> =
    let c = m.Config

    let dtype =
        if typeof<'T> = typeof<float32> then
            DType.F32
        else
            DType.F64

    let shape = Shape.matrix c.MaxPos (c.KvHeads * c.HeadDim)

    {
        Keys = Array.init c.Layers (fun _ -> HostTensor.zeros<'T> dtype shape)
        Values = Array.init c.Layers (fun _ -> HostTensor.zeros<'T> dtype shape)
    }

/// One forward over `ids` at positions pos0 .. pos0 + T - 1: writes the new K and V rows into the cache, records every
/// ActivationNames tap into `taps` (when given) and returns the logits [T, vocab].
let forward<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (exec: CpuExec)
    (m: Model)
    (cache: Cache<'T>)
    (pos0: int)
    (ids: int[])
    (taps: Dictionary<string, HostTensor<'T>> option)
    : HostTensor<'T> =
    let c = m.Config
    let w = m.Weights
    let eps = 'T.CreateTruncating c.Eps
    let scale = CpuRef.attentionScale<'T> c.HeadDim

    let shape =
        {
            QHeads = c.Heads
            KvHeads = c.KvHeads
            HeadDim = c.HeadDim
        }

    let tap (name: string) (t: HostTensor<'T>) =
        taps |> Option.iter (fun d -> d[name] <- t)
        t

    let mutable x = tap "embed.out" (CpuRef.embed<'T> exec w.Embed ids)

    for i in 0 .. c.Layers - 1 do
        let lw = w.Layers[i]
        let p = Invariant.format $"L{i}."
        let h = tap (p + "input_layernorm.out") (CpuRef.rmsNorm exec x lw.InputNorm eps)
        let q = tap (p + "attn.q_proj.out") (CpuRef.matmulBf16W exec h lw.Q)
        let k = tap (p + "attn.k_proj.out") (CpuRef.matmulBf16W exec h lw.K)
        let v = tap (p + "attn.v_proj.out") (CpuRef.matmulBf16W exec h lw.V)
        let qn = tap (p + "attn.q_norm.out") (CpuRef.rmsNorm exec q lw.QNorm eps)
        let kn = tap (p + "attn.k_norm.out") (CpuRef.rmsNorm exec k lw.KNorm eps)
        let qr = tap (p + "attn.q_rope") (CpuRef.ropeApply exec m.Rope qn c.Heads pos0)
        let kr = tap (p + "attn.k_rope") (CpuRef.ropeApply exec m.Rope kn c.KvHeads pos0)
        CpuRef.copyRows kr 0 cache.Keys[i] pos0 ids.Length
        CpuRef.copyRows v 0 cache.Values[i] pos0 ids.Length

        tap (p + "attn.v_cache") (rowsOf cache.Values[i] pos0 (pos0 + ids.Length))
        |> ignore

        let o =
            tap (p + "attn.o_proj.in") (CpuRef.attention exec qr cache.Keys[i] cache.Values[i] shape pos0 scale)

        let oo = tap (p + "attn.o_proj.out") (CpuRef.matmulBf16W exec o lw.O)
        let r = tap (p + "resid_mid") (CpuRef.add exec x oo)

        let h2 =
            tap (p + "post_attention_layernorm.out") (CpuRef.rmsNorm exec r lw.PostNorm eps)

        let g = tap (p + "mlp.gate.out") (CpuRef.matmulBf16W exec h2 lw.Gate)
        let u = tap (p + "mlp.up.out") (CpuRef.matmulBf16W exec h2 lw.Up)
        let d = tap (p + "mlp.down.in") (CpuRef.siluMul exec g u)
        let dd = tap (p + "mlp.down.out") (CpuRef.matmulBf16W exec d lw.Down)
        x <- tap (p + "out") (CpuRef.add exec r dd)

    let f = tap "final_norm.out" (CpuRef.rmsNorm exec x w.Norm eps)
    tap "logits" (CpuRef.matmulBf16W exec f w.Embed)

/// A prefill of `prompt` with every tap recorded.
let prefillTaps<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (exec: CpuExec)
    (prompt: int[])
    =
    let m = model.Force()
    let taps = Dictionary<string, HostTensor<'T>>(StringComparer.Ordinal)
    let cache = newCache<'T> m
    forward exec m cache 0 prompt (Some taps) |> ignore
    taps, cache

/// The canonical greedy schedule: one prefill, then single-token decode steps; argmax of each step's last row.
let greedy<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (exec: CpuExec)
    (prompt: int[])
    (steps: int)
    =
    let m = model.Force()
    let cache = newCache<'T> m
    let logits = forward exec m cache 0 prompt None
    let mutable next = (CpuRef.argmax (rowOf logits (logits.Rows - 1)))[0]
    let ids = ResizeArray [ next ]

    for s in 1 .. steps - 1 do
        let l = forward exec m cache (prompt.Length + s - 1) [| next |] None
        next <- (CpuRef.argmax l)[0]
        ids.Add next

    ids.ToArray()
