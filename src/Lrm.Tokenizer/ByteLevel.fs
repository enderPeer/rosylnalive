/// The byte-to-unicode map of the ByteLevel pre-tokenizer and decoder (TokenizerApi contract, "Byte level").
///
/// GPT-2's map: the 188 printable bytes 0x21..0x7E, 0xA1..0xAC and 0xAE..0xFF stand for themselves, and the other 68
/// bytes, in increasing order, become U+0100, U+0101, ... U+0143. Every vocabulary token of tokenizer.json is a string
/// over these 256 characters, one character per byte.
module Lrm.Tokenizer.ByteLevel

open System

let private printable (b: int) =
    (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)

/// The character that stands for each byte.
let private byteChars: char[] =
    let map = Array.zeroCreate<char> 256
    let mutable next = 0x100

    for b in 0..255 do
        if printable b then
            map[b] <- char b
        else
            map[b] <- char next
            next <- next + 1

    map

/// The highest character of the map (U+0143).
let private maxChar = 0x143

/// The byte each map character stands for, or -1.
let private charBytes: int[] =
    let back = Array.create (maxChar + 1) -1

    for b in 0..255 do
        back[int byteChars[b]] <- b

    back

/// The character that stands for byte `b`.
let charOf (b: byte) = byteChars[int b]

/// The byte that character `c` stands for, or None when `c` is not in the map.
let tryByteOf (c: char) =
    let i = int c

    if i <= maxChar && charBytes[i] >= 0 then
        Some(byte charBytes[i])
    else
        None

/// The map string of `bytes`: one character per byte.
let encode (bytes: byte[]) =
    String(Array.map (fun (b: byte) -> byteChars[int b]) bytes)

/// The bytes of a map string, or None when a character is not in the map.
let tryDecode (token: string) : byte[] option =
    let out = Array.zeroCreate<byte> token.Length
    let mutable ok = true
    let mutable i = 0

    while ok && i < token.Length do
        let c = int token[i]

        if c <= maxChar && charBytes[c] >= 0 then
            out[i] <- byte charBytes[c]
        else
            ok <- false

        i <- i + 1

    if ok then Some out else None

/// The bytes that the ByteLevel decoder gives for one token string: its map bytes when every character is in the map,
/// else the token's own UTF-8 bytes (tokenizers' ByteLevel decode_chain does the same).
let decoderBytes (token: string) : byte[] =
    match tryDecode token with
    | Some bytes -> bytes
    | None -> Text.Encoding.UTF8.GetBytes token
