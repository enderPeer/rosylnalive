namespace Lrm.Gpu.Core

open System
open System.Buffers.Binary
open System.Text

/// The SPIR-V side of the binding model (ShaderBuildAndSourceConvention v1.1, from VulkanHeapRecipe): where `LRM_BUF`
/// and `LRM_ARGS` sit in every SPIR-V blob, and the mapping values a Vulkan backend passes so that those bindings
/// resolve into the one resource heap. The SPIR-V inspector checks every blob against these values.
[<RequireQualifiedAccess>]
module SpirvBinding =
    /// The descriptor set of `g_buf[]`, the one resource variable (`[[vk::binding(0, 0)]]` in lrm.hlsli).
    [<Literal>]
    let DescriptorSet = 0u

    /// The binding of `g_buf[]` within its set.
    [<Literal>]
    let Binding = 0u

    /// `VkDescriptorSetAndBindingMappingEXT.firstBinding`: the mapping starts at `g_buf[]`'s binding.
    [<Literal>]
    let FirstBinding = 0u

    /// `VkDescriptorSetAndBindingMappingEXT.bindingCount`: one binding, the runtime array.
    [<Literal>]
    let BindingCount = 1u

    /// `VkDescriptorSetAndBindingMappingEXT.resourceMask`: READ_ONLY_STORAGE_BUFFER (0x40) | READ_WRITE_STORAGE_BUFFER
    /// (0x80). The read-only bit keeps the mapping valid if a later DXC marks the array NonWritable.
    [<Literal>]
    let ResourceMask = 0xC0u

    /// `VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT`.
    [<Literal>]
    let MappingSourceHeapWithConstantOffset = 0u

    /// `VkDescriptorMappingSourceConstantOffsetEXT.heapOffset`: slot 0 is the first descriptor of the heap.
    [<Literal>]
    let HeapOffset = 0u

    /// The push-data offset of the ArgBlock (`vkCmdPushDataEXT`).
    [<Literal>]
    let PushDataOffset = 0u

    /// The bytes of push data per dispatch: the whole 128-byte ArgBlock, whatever the kernel's block declares.
    [<Literal>]
    let PushDataBytes = 128u

    /// `VkDescriptorMappingSourceConstantOffsetEXT.heapArrayStride`: the device's `bufferDescriptorSize`.
    let heapArrayStride (bufferDescriptorSize: uint32) =
        if bufferDescriptorSize = 0u then
            Fail.contract "bufferDescriptorSize must be positive"

        bufferDescriptorSize

    /// The heap byte that `g_buf[slot]` resolves to: heapOffset + (binding - firstBinding + slot) * heapArrayStride.
    let resolve (bufferDescriptorSize: uint32) (slot: int) : int64 =
        if slot < 0 || slot >= Heap.SlotCount then
            Fail.contract "slot %d is outside 0..%d" slot (Heap.SlotCount - 1)

        int64 HeapOffset
        + (int64 Binding - int64 FirstBinding + int64 slot)
          * int64 (heapArrayStride bufferDescriptorSize)

    /// The SPIR-V DXC arguments that must be present: the SPIR-V backend for the Vulkan 1.3 environment.
    let requiredSpirvFlags = [ "-spirv"; "-fspv-target-env=vulkan1.3" ]

    /// DXC arguments the SPIR-V build must never use, with the reason. The binding is decided in lrm.hlsli, not by
    /// an emulation flag; descriptor-heap SPIR-V (SPV_EXT_descriptor_heap) does not support UAVs in the pinned DXC;
    /// 16-bit types are forbidden; a denormal mode needs shaderDenormPreserveFloat32; DXC layout rules would move
    /// the push-block offsets.
    let forbiddenSpirvFlags =
        [
            "-fvk-bind-resource-heap", "the binding is the explicit g_buf[] array of lrm.hlsli (VulkanHeapRecipe)"
            "-fspv-use-descriptor-heap", "direct heap indexing is not supported for UAVs by the pinned DXC"
            "-enable-16bit-types", "kernels use no 16-bit types"
            "-denorm", "a SPIR-V denormal mode needs shaderDenormPreserveFloat32"
            "-fvk-use-dx-layout", "the push block uses DXC's default Vulkan layout"
            "-fvk-use-scalar-layout", "the push block uses DXC's default Vulkan layout"
            "-fvk-use-gl-layout", "the push block uses DXC's default Vulkan layout"
            "-fspv-reflect", "no reflection decorations"
        ]

    /// What is wrong with a SPIR-V argument list: a required flag missing, or a forbidden flag present.
    let spirvFlagProblems (flags: string list) : string list =
        [
            for f in requiredSpirvFlags do
                if not (List.contains f flags) then
                    sprintf "the SPIR-V flags lack %s" f

            for f, why in forbiddenSpirvFlags do
                if List.contains f flags then
                    sprintf "the SPIR-V flags must not contain %s: %s" f why
        ]

/// One SPIR-V instruction.
type SpirvInstruction =
    {
        /// The word index of the instruction in the module (the header is words 0 to 4).
        Word: int
        Opcode: int
        /// The operand words: the instruction without its first word.
        Operands: uint32[]
    }

/// A parsed SPIR-V module.
type SpirvModule =
    {
        /// The version word, for example 0x00010600 for SPIR-V 1.6.
        Version: uint32
        Generator: uint32
        /// Every result id is below the bound.
        Bound: uint32
        Schema: uint32
        Instructions: SpirvInstruction list
    }

/// A minimal reader and writer of SPIR-V modules (little-endian words, SPIR-V 1.6 unified specification).
[<RequireQualifiedAccess>]
module Spirv =
    /// The SPIR-V magic number.
    [<Literal>]
    let Magic = 0x07230203u

    /// SPIR-V 1.6, the version DXC emits for -fspv-target-env=vulkan1.3.
    [<Literal>]
    let Version16 = 0x00010600u

    /// Opcodes the inspector reads.
    [<RequireQualifiedAccess>]
    module Op =
        [<Literal>]
        let Name = 5

        [<Literal>]
        let MemberName = 6

        [<Literal>]
        let Extension = 10

        [<Literal>]
        let ExtInstImport = 11

        [<Literal>]
        let MemoryModel = 14

        [<Literal>]
        let EntryPoint = 15

        [<Literal>]
        let ExecutionMode = 16

        [<Literal>]
        let Capability = 17

        [<Literal>]
        let TypeVoid = 19

        [<Literal>]
        let TypeBool = 20

        [<Literal>]
        let TypeInt = 21

        [<Literal>]
        let TypeFloat = 22

        [<Literal>]
        let TypeVector = 23

        [<Literal>]
        let TypeMatrix = 24

        [<Literal>]
        let TypeImage = 25

        [<Literal>]
        let TypeSampler = 26

        [<Literal>]
        let TypeSampledImage = 27

        [<Literal>]
        let TypeArray = 28

        [<Literal>]
        let TypeRuntimeArray = 29

        [<Literal>]
        let TypeStruct = 30

        [<Literal>]
        let TypePointer = 32

        [<Literal>]
        let Constant = 43

        [<Literal>]
        let SpecConstant = 50

        [<Literal>]
        let Function = 54

        [<Literal>]
        let FunctionParameter = 55

        [<Literal>]
        let FunctionEnd = 56

        [<Literal>]
        let Variable = 59

        [<Literal>]
        let Load = 61

        [<Literal>]
        let AccessChain = 65

        [<Literal>]
        let InBoundsAccessChain = 66

        [<Literal>]
        let PtrAccessChain = 67

        [<Literal>]
        let Decorate = 71

        [<Literal>]
        let MemberDecorate = 72

        [<Literal>]
        let CopyObject = 83

        [<Literal>]
        let AtomicLoad = 227

        [<Literal>]
        let AtomicStore = 228

        [<Literal>]
        let AtomicExchange = 229

        [<Literal>]
        let AtomicIAdd = 234

        [<Literal>]
        let AtomicXor = 242

        [<Literal>]
        let AtomicFlagTestAndSet = 318

        [<Literal>]
        let AtomicFlagClear = 319

        [<Literal>]
        let GroupNonUniformElect = 333

        [<Literal>]
        let GroupNonUniformFAdd = 350

        [<Literal>]
        let GroupNonUniformQuadSwap = 366

        [<Literal>]
        let ExecutionModeId = 331

        [<Literal>]
        let AtomicFMinEXT = 5614

        [<Literal>]
        let AtomicFMaxEXT = 5615

        [<Literal>]
        let AtomicFAddEXT = 6035

    /// Storage classes.
    [<RequireQualifiedAccess>]
    module StorageClass =
        [<Literal>]
        let UniformConstant = 0u

        [<Literal>]
        let Input = 1u

        [<Literal>]
        let Uniform = 2u

        [<Literal>]
        let Output = 3u

        [<Literal>]
        let Workgroup = 4u

        [<Literal>]
        let Private = 6u

        [<Literal>]
        let Function = 7u

        [<Literal>]
        let PushConstant = 9u

        [<Literal>]
        let StorageBuffer = 12u

        let name (sc: uint32) =
            match sc with
            | 0u -> "UniformConstant"
            | 1u -> "Input"
            | 2u -> "Uniform"
            | 3u -> "Output"
            | 4u -> "Workgroup"
            | 5u -> "CrossWorkgroup"
            | 6u -> "Private"
            | 7u -> "Function"
            | 8u -> "Generic"
            | 9u -> "PushConstant"
            | 10u -> "AtomicCounter"
            | 11u -> "Image"
            | 12u -> "StorageBuffer"
            | 5349u -> "PhysicalStorageBuffer"
            | n -> sprintf "storage class %d" n

    /// Decorations.
    [<RequireQualifiedAccess>]
    module Decoration =
        [<Literal>]
        let Block = 2u

        [<Literal>]
        let ArrayStride = 6u

        [<Literal>]
        let BuiltIn = 11u

        [<Literal>]
        let NonWritable = 24u

        [<Literal>]
        let Binding = 33u

        [<Literal>]
        let DescriptorSet = 34u

        [<Literal>]
        let Offset = 35u

    /// Capabilities, with names for the ones the inspector reports.
    [<RequireQualifiedAccess>]
    module Capability =
        [<Literal>]
        let Shader = 1u

        [<Literal>]
        let StorageBufferArrayDynamicIndexing = 30u

        [<Literal>]
        let RuntimeDescriptorArray = 5302u

        let name (c: uint32) =
            match c with
            | 0u -> "Matrix"
            | 1u -> "Shader"
            | 8u -> "Float16Buffer"
            | 9u -> "Float16"
            | 10u -> "Float64"
            | 11u -> "Int64"
            | 12u -> "Int64Atomics"
            | 18u -> "Groups"
            | 22u -> "Int16"
            | 30u -> "StorageBufferArrayDynamicIndexing"
            | 39u -> "Int8"
            | 43u -> "Sampled1D"
            | 44u -> "Image1D"
            | 46u -> "SampledBuffer"
            | 47u -> "ImageBuffer"
            | 55u -> "StorageImageReadWithoutFormat"
            | 56u -> "StorageImageWriteWithoutFormat"
            | 61u -> "GroupNonUniform"
            | 62u -> "GroupNonUniformVote"
            | 63u -> "GroupNonUniformArithmetic"
            | 64u -> "GroupNonUniformBallot"
            | 65u -> "GroupNonUniformShuffle"
            | 66u -> "GroupNonUniformShuffleRelative"
            | 67u -> "GroupNonUniformClustered"
            | 68u -> "GroupNonUniformQuad"
            | 4433u -> "StorageBuffer16BitAccess"
            | 4434u -> "UniformAndStorageBuffer16BitAccess"
            | 4435u -> "StoragePushConstant16"
            | 4436u -> "StorageInputOutput16"
            | 4448u -> "StorageBuffer8BitAccess"
            | 4449u -> "UniformAndStorageBuffer8BitAccess"
            | 4450u -> "StoragePushConstant8"
            | 4464u -> "DenormPreserve"
            | 4465u -> "DenormFlushToZero"
            | 4466u -> "SignedZeroInfNanPreserve"
            | 4467u -> "RoundingModeRTE"
            | 4468u -> "RoundingModeRTZ"
            | 5301u -> "ShaderNonUniform"
            | 5302u -> "RuntimeDescriptorArray"
            | 5345u -> "VulkanMemoryModel"
            | 5347u -> "PhysicalStorageBufferAddresses"
            | 5612u -> "AtomicFloat32MinMaxEXT"
            | 5613u -> "AtomicFloat64MinMaxEXT"
            | 5616u -> "AtomicFloat16MinMaxEXT"
            | 6033u -> "AtomicFloat32AddEXT"
            | 6034u -> "AtomicFloat64AddEXT"
            | 6095u -> "AtomicFloat16AddEXT"
            | n -> sprintf "%d" n

        /// "Name (n)", or "n" for a capability without a name here.
        let label (c: uint32) =
            match name c with
            | n when n = sprintf "%d" c -> n
            | n -> sprintf "%s (%d)" n c

    /// Built-in variables.
    [<RequireQualifiedAccess>]
    module BuiltIn =
        [<Literal>]
        let WorkgroupId = 26u

        [<Literal>]
        let LocalInvocationId = 27u

        [<Literal>]
        let GlobalInvocationId = 28u

        [<Literal>]
        let LocalInvocationIndex = 29u

        let name (b: uint32) =
            match b with
            | 24u -> "NumWorkgroups"
            | 25u -> "WorkgroupSize"
            | 26u -> "WorkgroupId"
            | 27u -> "LocalInvocationId"
            | 28u -> "GlobalInvocationId"
            | 29u -> "LocalInvocationIndex"
            | 36u -> "SubgroupSize"
            | 38u -> "NumSubgroups"
            | 40u -> "SubgroupId"
            | 41u -> "SubgroupLocalInvocationId"
            | n -> sprintf "built-in %d" n

    /// Execution modes.
    [<RequireQualifiedAccess>]
    module ExecutionMode =
        [<Literal>]
        let LocalSize = 17u

        let name (m: uint32) =
            match m with
            | 17u -> "LocalSize"
            | 38u -> "LocalSizeId"
            | 4459u -> "DenormPreserve"
            | 4460u -> "DenormFlushToZero"
            | 4461u -> "SignedZeroInfNanPreserve"
            | 4462u -> "RoundingModeRTE"
            | 4463u -> "RoundingModeRTZ"
            | n -> sprintf "execution mode %d" n

    /// Names of the atomic and wave opcodes the inspector reports.
    let opName (op: int) =
        let nonUniform =
            [|
                "Elect"
                "All"
                "Any"
                "AllEqual"
                "Broadcast"
                "BroadcastFirst"
                "Ballot"
                "InverseBallot"
                "BallotBitExtract"
                "BallotBitCount"
                "BallotFindLSB"
                "BallotFindMSB"
                "Shuffle"
                "ShuffleXor"
                "ShuffleUp"
                "ShuffleDown"
                "IAdd"
                "FAdd"
                "IMul"
                "FMul"
                "SMin"
                "UMin"
                "FMin"
                "SMax"
                "UMax"
                "FMax"
                "BitwiseAnd"
                "BitwiseOr"
                "BitwiseXor"
                "LogicalAnd"
                "LogicalOr"
                "LogicalXor"
                "QuadBroadcast"
                "QuadSwap"
            |]

        let atomics =
            [|
                "Load"
                "Store"
                "Exchange"
                "CompareExchange"
                "CompareExchangeWeak"
                "IIncrement"
                "IDecrement"
                "IAdd"
                "ISub"
                "SMin"
                "UMin"
                "SMax"
                "UMax"
                "And"
                "Or"
                "Xor"
            |]

        match op with
        | _ when op >= Op.AtomicLoad && op <= Op.AtomicXor -> "OpAtomic" + atomics[op - Op.AtomicLoad]
        | Op.AtomicFlagTestAndSet -> "OpAtomicFlagTestAndSet"
        | Op.AtomicFlagClear -> "OpAtomicFlagClear"
        | Op.AtomicFMinEXT -> "OpAtomicFMinEXT"
        | Op.AtomicFMaxEXT -> "OpAtomicFMaxEXT"
        | Op.AtomicFAddEXT -> "OpAtomicFAddEXT"
        | _ when op >= Op.GroupNonUniformElect && op <= Op.GroupNonUniformQuadSwap ->
            "OpGroupNonUniform" + nonUniform[op - Op.GroupNonUniformElect]
        | _ -> sprintf "opcode %d" op

    /// The GLCompute execution model.
    [<Literal>]
    let GLCompute = 5u

    /// True for a whole number of words starting with the magic number.
    let isModule (b: byte[]) =
        not (isNull b)
        && b.Length >= 20
        && b.Length % 4 = 0
        && BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(b, 0, 4)) = Magic

    /// The little-endian words of `b` (a whole number of words).
    let words (b: byte[]) : uint32[] =
        Array.init (b.Length / 4) (fun i -> BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(b, 4 * i, 4)))

    /// Parses a module: the 5-word header, then instructions whose word counts tile the rest exactly.
    let parse (b: byte[]) : Result<SpirvModule, string> =
        if isNull b then
            Error "no bytes"
        elif b.Length % 4 <> 0 then
            Error(sprintf "%d bytes is not a whole number of words" b.Length)
        elif b.Length < 20 then
            Error "shorter than the 5-word header"
        else
            let w = words b

            if w[0] <> Magic then
                if BinaryPrimitives.ReverseEndianness w[0] = Magic then
                    Error "big-endian modules are not supported"
                else
                    Error(sprintf "no SPIR-V magic (word 0 is 0x%08x)" w[0])
            else
                let out = ResizeArray<SpirvInstruction>()
                let mutable i = 5
                let mutable err = None

                while err.IsNone && i < w.Length do
                    let count = int (w[i] >>> 16)

                    if count = 0 then
                        err <- Some(sprintf "word %d: an instruction with word count 0" i)
                    elif i + count > w.Length then
                        err <- Some(sprintf "word %d: an instruction of %d words runs past the end" i count)
                    else
                        out.Add
                            {
                                Word = i
                                Opcode = int (w[i] &&& 0xFFFFu)
                                Operands = Array.sub w (i + 1) (count - 1)
                            }

                        i <- i + count

                match err with
                | Some e -> Error e
                | None ->
                    Ok
                        {
                            Version = w[1]
                            Generator = w[2]
                            Bound = w[3]
                            Schema = w[4]
                            Instructions = List.ofSeq out
                        }

    /// The bytes of a module: its header, then every instruction with its word count recomputed.
    let toBytes (m: SpirvModule) : byte[] =
        let total = 5 + (m.Instructions |> List.sumBy (fun i -> 1 + i.Operands.Length))

        let out = Array.zeroCreate<byte>(4 * total)
        let mutable o = 0

        let put (v: uint32) =
            BinaryPrimitives.WriteUInt32LittleEndian(Span(out, 4 * o, 4), v)
            o <- o + 1

        put Magic
        put m.Version
        put m.Generator
        put m.Bound
        put m.Schema

        for ins in m.Instructions do
            if ins.Opcode < 0 || ins.Opcode > 0xFFFF || ins.Operands.Length + 1 > 0xFFFF then
                invalidArg "m" (sprintf "opcode %d with %d operands cannot be encoded" ins.Opcode ins.Operands.Length)

            put ((uint32 (ins.Operands.Length + 1) <<< 16) ||| uint32 ins.Opcode)

            for v in ins.Operands do
                put v

        out

    /// A new instruction (Word is -1 until it is written).
    let instruction (opcode: int) (operands: uint32 list) =
        {
            Word = -1
            Opcode = opcode
            Operands = Array.ofList operands
        }

    /// The literal string at operand `start`: UTF-8, nul-terminated, padded to whole words. Returns the string and
    /// the number of words it takes, or None when it has no terminator.
    let literalString (operands: uint32[]) (start: int) : (string * int) option =
        if start < 0 || start >= operands.Length then
            None
        else
            let bytes = ResizeArray<byte>()
            let mutable k = start
            let mutable fin = false

            while not fin && k < operands.Length do
                let v = operands[k]

                for s in 0..3 do
                    let c = byte ((v >>> (8 * s)) &&& 0xFFu)

                    if not fin then
                        if c = 0uy then fin <- true else bytes.Add c

                k <- k + 1

            if fin then
                Some(Encoding.UTF8.GetString(bytes.ToArray()), k - start)
            else
                None

    /// The words of a literal string (UTF-8, nul-terminated, zero-padded to whole words).
    let stringOperands (s: string) : uint32 list =
        let b = Encoding.UTF8.GetBytes s
        let n = b.Length / 4 + 1
        let padded = Array.zeroCreate<byte>(4 * n)
        Array.blit b 0 padded 0 b.Length

        [
            for i in 0 .. n - 1 do
                BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(padded, 4 * i, 4))
        ]

    /// The LocalSize execution mode (OpExecutionMode, mode 17), if there is exactly one.
    let localSize (b: byte[]) : (int * int * int) option =
        match (if isModule b then parse b else Error "not a module") with
        | Error _ -> None
        | Ok m ->
            let found =
                m.Instructions
                |> List.filter (fun i ->
                    i.Opcode = Op.ExecutionMode
                    && i.Operands.Length >= 5
                    && i.Operands[1] = ExecutionMode.LocalSize)

            match found with
            | [ i ] -> Some(int i.Operands[2], int i.Operands[3], int i.Operands[4])
            | _ -> None
