let torch = System.IO.File.ReadAllLines(__SOURCE_DIRECTORY__ + "/inv_torch.txt") |> Array.map uint32
let bits (f: float32) = System.BitConverter.SingleToUInt32Bits f
let a = [| for i in 0..63 -> 1.0f / System.MathF.Pow(1e6f, float32 (2*i) / 128.0f) |]
let b = [| for i in 0..63 -> float32 (1.0 / System.Math.Pow(1e6, float (2*i) / 128.0)) |]
let c = [| for i in 0..63 -> 1.0f / float32 (System.Math.Pow(1e6, float (float32 (2*i) / 128.0f))) |]
let diff (x: float32[]) = Array.zip x torch |> Array.filter (fun (v, t) -> bits v <> t) |> Array.length
printfn "System.MathF.Pow f32 path mismatches: %d; f64-then-round mismatches: %d; f64 pow, round, f32 divide: %d" (diff a) (diff b) (diff c)
