# HostTensorAndSafetensors contract

Owner: O2. Status: v1, frozen at the contracts-v1 tag. Consumers: M2 (`Qwen3Weights.open`, CpuOps), X1 (weight streaming, taps), E1 (CpuDevice buffers), C2 (stage boundaries), I1 (`lrm bisect`) and every reader of goldens. Code: `src/Lrm.Tensor/Shape.fs`, `HostTensor.fs` and `Safetensors.fs`. Tests: `tests/Lrm.Tensor.Tests/TensorTests.fs` and `SafetensorsTests.fs` (cpu tier), `ModelTests.fs` (model tier).

After contracts-v1, this text and its vectors change only through a `contract-change` PR, with a CHANGELOG entry and updated vectors. Element types and their codes are LrmCore's `DType` (`lrm-core.md`). The golden-file conventions that build on this format are GoldenSet (`golden-set.md`).

## Shape

A `Shape` is a list of dimensions, outermost first.

- **Dimensions** are `int64` and at least 0. A negative dimension, or an element count beyond `int64`, raises `ArgumentException`.
- **Rank 0** is a scalar with one element (`Shape.Scalar`). A dimension of 0 gives an empty tensor.
- **Elements** is the product of the dimensions. `ElementCount` is the same as an `int` for host arrays, and raises when it does not fit.
- **RowMajorStrides** are the packed strides in elements: the last dimension has stride 1, and each earlier one the product of the later dimensions.
- **Rows and Cols.** `Rows` is the first dimension (1 for a scalar). `Cols` is the product of the others (1 for rank 0 and 1). Row-major ops treat a tensor as `[Rows, Cols]`.
- **Text** is `[13, 64]`, with invariant digits.
- **Value semantics.** A shape keeps its own copy of the dimensions; `Dims` returns a copy. Two shapes are equal when their dimensions are.

## TensorDesc

A `TensorDesc` is a dtype, a shape and one stride per dimension, in elements (not bytes).

- `TensorDesc.Contiguous(dtype, shape)` uses the row-major strides. `TensorDesc.Strided` takes explicit strides, each at least 0.
- **`IsContiguous`** holds when every stride equals the packed stride, ignoring dimensions of size 0 or 1, which never move.
- **`ExtentElements`** is `1 + Σ (dims[i] − 1)·strides[i]`, or 0 for an empty tensor: the elements a tensor spans from its first one. This is the extent rule of HeapLayoutAndTensorTable (R4a), whose rows hold at most 4 dimensions.
- **`ByteSize`** is elements times the element size.

## HostTensor

`HostTensor<'T>` is a contiguous row-major tensor in host memory: a dtype, a shape and a `'T[]` of exactly `shape.Elements` values.

| DType | Element carrier `'T` |
| --- | --- |
| F32 | `float32` |
| F64 | `float` |
| BF16, F16 | `uint16`: the raw bits |
| I32 | `int32` |
| U32 | `uint32` |
| U8 | `byte` |
| I64 | `int64` |

- **Checks.** Construction raises `ArgumentException` when `'T` does not carry the dtype, or the element count does not match the shape.
- **Ownership.** The array is not copied. Code that keeps it must not change it while the tensor is in use. `Clone` copies.
- **Access.** `Data`, `Span`, `Row r` (a span of `Cols` elements) and `Reshape` (same elements, another shape with the same count).
- **Bytes** are little-endian. `HostTensor.ofBytes` copies elements from bytes and checks the byte count; `ToBytes` returns them.
- **`IHostTensor`** is the dtype-erased view that comparators and writers take: `Desc`, `ToBytes` and `ToFloat64`.
  - `ToFloat64` widens every element exactly: BF16 and F16 through `Bf16.widen` and `F16.widen`, integers by conversion.
  - An I64 element beyond ±2^53 has no exact float64, so `ToFloat64` raises `InvalidOperationException` for it.

## Safetensors files

A safetensors file is an 8-byte little-endian header length N, then N bytes of header JSON, then the data section. The header maps each tensor name to `{"dtype", "shape", "data_offsets"}`, with offsets in bytes from the start of the data section, end exclusive. The optional key `__metadata__` maps strings to strings.

### The reader

`SafetensorsFile.Open path` memory-maps the whole file read-only and validates it before returning. Every violation raises `SafetensorsFormatException` (a `FormatException`) naming the file and the rule.

1. The file has at least 8 bytes.
2. N is at most 100,000,000 (the reference implementation's limit) and at most the file length minus 8.
3. The header starts with `{` and is a UTF-8 JSON object.
4. Keys are unique, at the top level, in `__metadata__` and inside each entry.
5. `__metadata__`, when present, is an object of string values.
6. Each tensor name is non-empty. Its entry is an object with exactly the fields `dtype`, `shape` and `data_offsets`.
7. `dtype` is one of the eight LrmCore safetensors names: `F32`, `BF16`, `F16`, `I32`, `U32`, `U8`, `F64`, `I64`. Other safetensors types, such as `BOOL` or `F8_E4M3`, are refused.
8. `shape` is an array of non-negative integers. `data_offsets` is a pair of non-negative integers `[b, e]` with `b <= e`.
9. `e − b` equals the elements times the dtype size, computed without overflow, and `e` is at most the data length.
10. **The tensors tile the data section.** Sorted by offset, each tensor begins where the previous one ends, the first at 0: no holes and no overlaps. The last one ends exactly at the end of the file: no trailing bytes.

**Leniency.** The reader accepts any N, including one that is not a multiple of 8, and any order of the header keys. `IsAligned` reports whether the data section starts at a multiple of 8 and every tensor at a multiple of its element size, which is the GoldenSet alignment rule. Every golden file and the Qwen3-0.6B checkpoint are aligned.

**What it returns.**
- `HeaderLength` (N), `DataOffset` (8 + N), `Length`;
- `Metadata` and `Entries`, both sorted in code point order of the keys and names;
- `Entry`, `TryEntry` and `Contains` by name.

**Reading tensors.**
- **`Span name`** is the tensor's bytes, straight from the mapping, with no copy. It is valid until `Dispose`, after which reads raise `ObjectDisposedException`. A tensor over 2 GiB is read with `OpenRead`, a stream over the same bytes.
- **Typed copies.** `Read<'T>`, `ReadF32`, `ReadF64`, `ReadBf16`, `ReadI32`, `ReadU32`, `ReadU8` and `ReadI64` return a HostTensor. They raise `ArgumentException` when the dtype differs. `Tensor name` returns any tensor as an `IHostTensor`.
- **`Sha256()`** is the raw SHA-256 of the whole file, read from the mapping. It is a raw-content id (LrmCore keyspaces), for `SHA256SUMS` and golden manifests.
- **`HeaderBytes()`** is the file's own first 8 + N bytes. **`RebuildHeader()`** is the header the writer below builds from the parsed metadata and entries; it equals `HeaderBytes()` exactly when the file follows the writer's convention.

**Sharing and integrity.**
- The file is opened with `FileShare.Read`, so nobody can write it while it is open. Bytes a caller has verified therefore stay the bytes it reads.
- **`SafetensorsFile.OpenVerified(path, bytes, sha256)`** pins a file. It checks the size before mapping and the SHA-256 over the mapped bytes, both before the header is parsed, and raises `IntegrityException` on a mismatch. `Goldens.Open` (ComparatorsAndGoldenReader) opens every golden safetensors file this way.
- `SafetensorsFile.Validate bytes` runs the same validation on bytes in memory.
- Reads are thread-safe.

### The writer

`Safetensors.toBytes`, `Safetensors.writeTo stream` and `Safetensors.writeFile path` take metadata pairs and `SafetensorsTensor` values (name, dtype, shape, little-endian bytes; `SafetensorsTensor.ofHost` builds one from a HostTensor). The output is deterministic:

1. **Inputs are checked.** It raises `ArgumentException` for:
   - duplicate tensor names or metadata keys;
   - an empty name, or the name `__metadata__`;
   - text that is not valid Unicode (a lone surrogate);
   - a byte count other than elements times the dtype size.
2. **The header is compact ASCII JSON.**
   - `"__metadata__"` comes first, with its keys in code point order. It is omitted when there is no metadata.
   - Then each tensor, in code point order of the names, as `{"dtype":"F32","shape":[13,64],"data_offsets":[b,e]}`, with the fields in that order and no spaces.
3. **Code point order** is the order of the UTF-8 bytes. It is also the order of Python's `sorted()` over `str`, and it differs from .NET's UTF-16 ordinal order for astral characters.
4. **Strings are escaped exactly as Python's `json.dumps(s, ensure_ascii=True)`:**

   | Character | Written as |
   | --- | --- |
   | `"` and `\` | `\"` and `\\` |
   | LF, CR, TAB, BS, FF | `\n`, `\r`, `\t`, `\b`, `\f` |
   | Other characters below U+0020, and DEL (U+007F) | `\u00xx`, lowercase hex |
   | U+0020 to U+007E, apart from `"` and `\` | the character itself |
   | Everything above U+007E | `\uxxxx` per UTF-16 unit, lowercase hex, so an astral character becomes a surrogate pair |

5. **Padding.** Spaces pad the header to a multiple of 8 bytes, so the data section starts at a multiple of 8.
6. **Data.** The tensors' bytes follow in the same name order, with no gaps and nothing after the last one.
7. **Order independence.** The same tensors and metadata, given in any order, produce the same bytes.
8. **Atomic files.** `writeFile` writes a temporary file in the target folder and moves it over the target only when complete.

**One convention, two implementations.** This is the convention of `tools/goldens/common.py` (`safetensors_bytes`, O1) and of the Qwen3-0.6B checkpoint itself. The F# writer rebuilds every tiny golden file byte for byte from the parsed file, and rebuilds the real checkpoint's 35,552-byte header byte for byte.

## Vectors

**S1, the writer.** The bytes below were produced by a standalone Python 3 script using only `json` and `struct` with the GoldenSet convention, independently of the F# code. The cpu tier checks the F# writer against them.

- Metadata: `lrm.test` = `saïtensors!`, and `a` = the 7 characters `"`, `q`, `"`, `\`, LF, DEL, TAB.
- Tensors:

  | Name | dtype | shape | values |
  | --- | --- | --- | --- |
  | `x` | F32 | [2] | 1.0, −2.0 |
  | `é😀` (U+00E9 U+1F600) | BF16 | [1, 1] | bits `3f80` |
  | `b` | I64 | [] | 7 |
  | `e` | U8 | [0] | none |

- Header (N = 312, the last 7 bytes are spaces):

  ```
  {"__metadata__":{"a":"\"q\"\\\n\u007f\t","lrm.test":"sa\u00eftensors!"},"b":{"dtype":"I64","shape":[],"data_offsets":[0,8]},"e":{"dtype":"U8","shape":[0],"data_offsets":[8,8]},"x":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},"\u00e9\ud83d\ude00":{"dtype":"BF16","shape":[1,1],"data_offsets":[16,18]}}
  ```

- The file is 338 bytes: 8 + 312 + 18, with data `0700000000000000 0000803f 000000c0 803f`. Its SHA-256 is `49fc00c6aa63a8f23a5488fe2bd49a8768779e8f5b943a166013f4b2140666ce`.

**Escapes.** The string U+0000 U+001F U+007F U+0080 U+FFFF U+10FFFF is written as `"\u0000\u001f\u007f\u0080\uffff\udbff\udfff"`.

**The tiny model** (`goldens/tiny/g1/model/model.safetensors`, 232,760 bytes):
- 24 BF16 tensors and `__metadata__` `{"format":"pt"}`, with N = 2,480;
- SHA-256 `7ade5616bee29bf1008e4a2902cd7aa6b334e3571aa978356d7adfe08e058b2e`, as in its SHA256SUMS.

**Qwen3-0.6B** (`LRM_DATA/models/Qwen3-0.6B/model.safetensors`, revision `c1899de289a04d12100db370d81485cdf75e47ca`):
- 311 BF16 tensors, `lm_head.weight` among them as [151936, 1024];
- `__metadata__` `{"format":"pt"}`;
- N = 35,552, so the data section starts at byte 35,560;
- aligned, and the writer rebuilds its header byte for byte;
- its SHA-256 equals the `SHA256SUMS` line.

## Verification

The cpu tier of `Lrm.Tensor.Tests` covers:
- **Shapes and host tensors:** element counts, strides, contiguity and extents; the carrier and count checks; exact widening, including the I64 limit.
- **The writer:**
  - vector S1 byte for byte;
  - order independence;
  - the escape table;
  - every input check;
  - atomic `writeFile`;
  - every tiny golden safetensors file rebuilt byte for byte, with its manifest SHA-256.
- **The reader:**
  - S1 read back, with `RebuildHeader` = `HeaderBytes`;
  - one test for each of 19 malformed headers, plus a short file, an N beyond the file and an N above the cap;
  - an N that is not a multiple of 8 accepted, with `IsAligned` false;
  - writers refused while a file is open;
  - spans invalid after `Dispose`;
  - `OpenVerified` refusing a wrong size or hash before parsing.
- **A property** (fixed FsCheck seed): random sets of tensors of every dtype and rank, with escaped and astral names, round-trip through write, read and rewrite to the same bytes, with N a multiple of 8.

The model tier opens the real Qwen3-0.6B checkpoint (311 BF16 entries, data at 35,560, header rebuilt) and hashes it against `SHA256SUMS`.
