namespace Lrm.Gpu.Core

open System
open Lrm.Core

/// One token argument of a dispatch, resolved through the table.
type TokenUse =
    {
        Field: string
        Access: Access
        Token: TensorToken
        Row: TensorRow
        /// The page slot of the tensor.
        Slot: int
        /// The byte range [Start, End) the tensor spans in its page.
        Start: int64
        End: int64
    }

/// A dispatch ready to record: the catalog entry, the logical and physical grids, the argument block with word 0
/// (lrm_groups_x) written, and the resolved tokens.
type PreparedDispatch =
    {
        Kernel: KernelId
        Spec: KernelSpec
        Grid: Grid
        Physical: PhysicalGrid
        Args: ArgBlock
        Tokens: TokenUse list
    }

/// The shared dispatch rules of the KernelAbiAndCatalog contract. Every backend calls Dispatch.prepare, so a
/// violation raises the same GpuContractException everywhere.
[<RequireQualifiedAccess>]
module Dispatch =
    /// The catalog entry of a kernel id.
    let spec (kernel: KernelId) =
        if kernel.Value < 0 || kernel.Value >= Kernels.all.Length then
            Fail.contract "%O is not in the catalog (%d kernels)" kernel Kernels.all.Length

        Kernels.all[kernel.Value]

    /// The id of a kernel name.
    let tryFind (name: string) =
        Kernels.all
        |> Array.tryFindIndex (fun k -> k.Name = name)
        |> Option.map KernelId

    /// Resolves and checks the token fields of `args`:
    /// - every token is below 16,384;
    /// - a meta token is the null token or an assigned row;
    /// - any other token is an assigned row whose dtype the field accepts, with the Contiguous flag, and, when the
    ///   kernel writes it, without the ReadOnly flag.
    let tokens (table: SealedTable) (k: KernelSpec) (args: ArgBlock) : TokenUse list =
        k.Fields
        |> List.mapi (fun i f -> i, f)
        |> List.choose (fun (i, f) ->
            match f.Kind with
            | FieldKind.Token(access, dtypes, _) ->
                let token = TensorToken args[i + 1]
                let row = table.Row token

                if access = Access.Meta then
                    if not token.IsNull && not (table.IsAssigned token) then
                        Fail.contract "%s.%s: token %d is not in the table" k.Name f.Name token.Value
                else
                    if token.IsNull then
                        Fail.contract "%s.%s: the null token is not allowed here" k.Name f.Name

                    if not (table.IsAssigned token) then
                        Fail.contract "%s.%s: token %d is not in the table" k.Name f.Name token.Value

                    match row.DType with
                    | Some t when List.contains t dtypes -> ()
                    | t ->
                        Fail.contract
                            "%s.%s: tensor '%s' is %s, the kernel accepts %s"
                            k.Name
                            f.Name
                            (table.Name token)
                            (t |> Option.map DType.name |> Option.defaultValue "invalid")
                            (dtypes |> List.map DType.name |> String.concat ", ")

                    if not (TensorFlags.has TensorFlags.Contiguous row.Flags) then
                        Fail.contract "%s.%s: tensor '%s' is not contiguous" k.Name f.Name (table.Name token)

                    if Access.writes access && TensorFlags.has TensorFlags.ReadOnly row.Flags then
                        Fail.contract "%s.%s: tensor '%s' is read only" k.Name f.Name (table.Name token)

                let start = int64 row.ByteOffset

                Some
                    {
                        Field = f.Name
                        Access = access
                        Token = token
                        Row = row
                        Slot = row.Slot
                        Start = start
                        End = start + TensorRow.extentBytes row
                    }
            | _ -> None)

    /// Checks aliasing: when two token arguments overlap in memory and one of them is written, the written field must
    /// list the other in its catalog alias, and the two must span exactly the same bytes. Two written arguments
    /// never overlap. Meta tokens are ignored.
    let checkAliasing (k: KernelSpec) (uses: TokenUse list) =
        let data = uses |> List.filter (fun u -> u.Access <> Access.Meta)

        let aliasOf (field: string) =
            match k.Fields |> List.tryFind (fun f -> f.Name = field) with
            | Some { Kind = FieldKind.Token(_, _, alias) } -> alias
            | _ -> []

        for i in 0 .. data.Length - 1 do
            for j in i + 1 .. data.Length - 1 do
                let a = data[i]
                let b = data[j]

                let overlap = a.Slot = b.Slot && a.Start < b.End && b.Start < a.End

                if overlap && (Access.writes a.Access || Access.writes b.Access) then
                    let same = a.Start = b.Start && a.End = b.End

                    let declared =
                        (Access.writes a.Access
                         && not (Access.writes b.Access)
                         && List.contains b.Field (aliasOf a.Field))
                        || (Access.writes b.Access
                            && not (Access.writes a.Access)
                            && List.contains a.Field (aliasOf b.Field))

                    if not (declared && same) then
                        Fail.contract
                            "%s: %s and %s overlap in slot %d, and %s"
                            k.Name
                            a.Field
                            b.Field
                            a.Slot
                            (if declared then
                                 "a declared alias must span exactly the same bytes"
                             else
                                 "the catalog does not let them alias")

    /// Checks an argument block for `kernel` against the sealed table and computes the grids:
    /// - word 0 is 0 (Dispatch writes it) and every word after the last field is 0;
    /// - the token rules of `tokens` and the aliasing rules of `checkAliasing`;
    /// - the catalog grid, folded by Grid.physical; word 0 of the result is the physical X.
    let prepare (table: SealedTable) (kernel: KernelId) (args: ArgBlock) : PreparedDispatch =
        let k = spec kernel

        if args[0] <> 0u then
            Fail.contract "%s: argument word 0 is lrm_groups_x, which Dispatch writes; pass 0" k.Name

        for w in k.Fields.Length + 1 .. ArgBlock.WordCount - 1 do
            if args[w] <> 0u then
                Fail.contract "%s: argument word %d is past the last field and must be 0" k.Name w

        let uses = tokens table k args
        checkAliasing k uses
        let grid = KernelSpec.gridOf k args

        match Grid.physical k.Linear grid with
        | Error e -> Fail.contract "%s: %s" k.Name e
        | Ok p ->
            {
                Kernel = kernel
                Spec = k
                Grid = grid
                Physical = p
                Args = args.WithGroupsX(uint32 p.X)
                Tokens = uses
            }

    /// Checks a copy: both ranges in pages (slots 1..63) of the layout, positive equal sizes, offsets and sizes
    /// multiples of 4, and no overlap.
    let checkCopy (layout: HeapLayout) (source: BufferRange) (destination: BufferRange) =
        let check (what: string) (r: BufferRange) =
            if r.Slot < 1 || r.Slot >= Heap.SlotCount then
                Fail.contract "copy %s: slot %d is not a page slot (1..63)" what r.Slot

            let page = layout.Page r.Slot

            if r.Offset < 0L || r.Bytes <= 0L || r.Offset % 4L <> 0L || r.Bytes % 4L <> 0L then
                Fail.contract "copy %s: offset %d and size %d must be multiples of 4, size > 0" what r.Offset r.Bytes

            if r.Offset + r.Bytes > page.Bytes then
                Fail.contract
                    "copy %s: bytes %d..%d are past the %d-byte page"
                    what
                    r.Offset
                    (r.Offset + r.Bytes)
                    page.Bytes

        check "source" source
        check "destination" destination

        if source.Bytes <> destination.Bytes then
            Fail.contract "copy: source has %d bytes, destination %d" source.Bytes destination.Bytes

        if
            source.Slot = destination.Slot
            && source.Offset < destination.Offset + destination.Bytes
            && destination.Offset < source.Offset + source.Bytes
        then
            Fail.contract "copy: source and destination overlap in slot %d" source.Slot

    /// Checks a host transfer: a page slot (1..63; slot 0 only for reads when `allowTable`), offset and size multiples
    /// of 4, inside the page.
    let checkTransfer (layout: HeapLayout) (allowTable: bool) (slot: int) (offset: int64) (bytes: int64) =
        let size =
            if slot = Heap.TableSlot && allowTable then
                Heap.TableBytes
            elif slot = Heap.TableSlot then
                Fail.contract "slot 0 holds the tensor table; write it with WriteTable"
            elif slot < 1 || slot >= Heap.SlotCount then
                Fail.contract "slot %d is outside 0..63" slot
            else
                (layout.Page slot).Bytes

        if offset < 0L || bytes < 0L || offset % 4L <> 0L || bytes % 4L <> 0L then
            Fail.contract
                "transfer in slot %d: offset %d and size %d must be non-negative multiples of 4"
                slot
                offset
                bytes

        if offset + bytes > size then
            Fail.contract
                "transfer in slot %d: bytes %d..%d are past the %d-byte buffer"
                slot
                offset
                (offset + bytes)
                size
