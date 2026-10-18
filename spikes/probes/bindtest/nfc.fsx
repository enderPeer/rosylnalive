let s = "é"
let n = s.Normalize(System.Text.NormalizationForm.FormC)
printfn "len in=%d out=%d culture=%s" s.Length n.Length (System.Globalization.CultureInfo.CurrentCulture.Name)
