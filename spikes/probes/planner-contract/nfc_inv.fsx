let s = "é Å ẛ̣ 각"
let t = try s.Normalize(System.Text.NormalizationForm.FormC) |> Seq.map (fun c -> sprintf "%04X" (int c)) |> String.concat " " with e -> "ERR " + e.GetType().Name + ": " + e.Message
printfn "env=%s nfc=%s" (System.Environment.GetEnvironmentVariable "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT") t
