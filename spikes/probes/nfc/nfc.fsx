let s = "é Å ﬁ ẛ̣ 𝒜"
try
  let n = s.Normalize(System.Text.NormalizationForm.FormC)
  printfn "invariant=%s normalized=%A len %d->%d" (System.Environment.GetEnvironmentVariable "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT") (n |> Seq.map int |> Seq.toList) s.Length n.Length
with e -> printfn "EXC %s: %s" (e.GetType().Name) e.Message
printfn "ICU version: %A" (try (typeof<System.Globalization.CultureInfo>.Assembly.GetType("System.Globalization.GlobalizationMode")) |> string with _ -> "?")
