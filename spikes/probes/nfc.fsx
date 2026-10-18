open System
open System.Text
printfn "invariant env=%s" (Environment.GetEnvironmentVariable "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT")
let s = "é Å ﬁ ẛ̣"
try
  let n = s.Normalize(NormalizationForm.FormC)
  printfn "NFC ok: %A" [ for c in n -> sprintf "%04X" (int c) ]
with e -> printfn "NFC failed: %s" (e.GetType().Name)
let r = Rune.GetRuneAt("\U0001D49C", 0)
printfn "rune cat=%A" (Rune.GetUnicodeCategory r)
printfn "unicode ver hint: U+1FAE9 cat=%A" (Rune.GetUnicodeCategory(Rune(0x1FAE9)))
printfn "U+31350 cat=%A" (Rune.GetUnicodeCategory(Rune(0x31350)))
