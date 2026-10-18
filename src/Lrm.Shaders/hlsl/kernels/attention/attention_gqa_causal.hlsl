// attention_gqa_causal (K3): causal grouped-query attention over the fp32 KV cache (KernelAbiAndCatalog, kernel 13;
// BatchInvariance, "Attention"). The algorithm, its order, its error and its timings are ADR K3-0001
// (docs/decisions/K3-0001-attention-kernel.md).
//
// One thread group per (query row i, query head h), never folded (the kernel is not linear): group gid computes head
// h = gid.y of row i = rows - 1 - gid.x, so the rows with the most keys start first and the short ones fill the tail.
// Query row i reads q row q_row0 + i, writes y row q_row0 + i, has position p = past + i and attends key positions
// 0 .. p; query head h reads KV head h / (q_heads / kv_heads). Element layouts (flat tokens, sizes from the fields):
//   q, y  [tokens, q_heads * head_dim]    element (row * q_heads + h) * head_dim + d
//   k, v  [max_pos, kv_heads * head_dim]  element (j * kv_heads + kvh) * head_dim + d   (K after norm and RoPE, V raw)
//
// The order, which depends only on head_dim and the position p (never on rows, q_row0, past or the group):
//   - keys are visited in blocks of 64 anchored at position 0 (block b holds keys 64b .. 64b + 63), in increasing b;
//     the causal mask is the loop bound: the last block holds keys 64b .. p, and no key or value after p is read;
//   - thread t scores key j = 64b + t: s_j = (the sum of mad(q[d], k_j[d], .) in increasing d from +0) * scale,
//     the order of CpuRef's dot;
//   - the block maximum M_b of the block's scores (exact, so its order does not matter); the running maximum is
//     m = M_0 after block 0 and m' = max(m, M_b) after block b, and alpha = exp(m - m');
//   - the weights e_j = exp(s_j - m'); the block sum is the sum of e_j in increasing j from +0, and the block value
//     is the sum of mad(e_j, v_j[d], .) in increasing j from +0, per output dimension d;
//   - the online-softmax state is updated once per block, from l = o = +0: l = mad(l, alpha, block sum) and
//     o[d] = mad(o[d], alpha, block value[d]); after the last block y[d] = o[d] / l.
// There is no -inf (the running maximum starts at the first block's maximum) and no NaN from finite inputs: the key
// that holds the maximum adds exp(0) = 1 to l, and no later block rescales it (alpha = exp(0) once m is reached).
// DXIL arithmetic is fast-math (ADR int-0001), so the compiler may contract or reassociate what is written here; it
// compiles one instruction sequence that every (row, head) runs, so the result still depends only on p and the data.
//
// Memory. The keys of a block are staged in groupshared 32 dimensions at a time, read coalesced: thread t reads
// dimension t % 32 of keys t / 32, t / 32 + 2, ..., so warp w reads 128 contiguous bytes of key 2n + w per step n,
// 16 loads issued before their 16 stores. Thread t then reads its own key's chunk from groupshared (row stride 33
// floats, so the 32 threads of a warp hit 32 banks). A last chunk narrower than 32 dimensions (head_dim not a multiple
// of 32) is staged element by element: element e = t + 64n is dimension e % width of key e / width. The values are
// read straight from the cache, 16 keys of loads issued before their 16 mads; for one key the threads of a warp read
// consecutive dimensions, 128 contiguous bytes.
//
// Output dimensions are handled in passes of 128: in pass c, thread t owns d = 128c + t and d = 128c + 64 + t. A pass
// recomputes the scores and weights of the previous pass exactly (the same instructions on the same inputs), so a
// head_dim above 128 costs one pass per 128 dimensions and changes no order. Qwen3-0.6B (head_dim 128) runs one pass.
//
// groupshared: the staged key chunk (64 x 33 floats), the query chunk (32), the scores and the weights (64 each):
// 9,088 bytes. Barriers, all reached by the whole group: one after each staged chunk, one before each later chunk
// (its readers are done), one after the scores and one after the weights; the next block's first writes come after a
// barrier that every reader of this block's data has passed.
#include "lrm.hlsli"

LRM_ARGS(attention_gqa_causal)

// Keys per block: the catalog's thread count, one key per thread.
static const uint KeysPerBlock = 64u;

// Output dimensions per pass: two per thread, d0 + t and d0 + HalfPass + t.
static const uint DimsPerPass = 128u;
static const uint HalfPass = 64u;

// Key dimensions staged at a time, the padded row stride of the staged keys, and the keys a staging step covers.
static const uint ChunkDims = 32u;
static const uint ChunkStride = 33u;
static const uint KeysPerStep = 2u;

// Loads issued together: keys per thread in a staging batch, and keys per value batch.
static const uint Batch = 16u;

groupshared float g_keys[64 * 33];
groupshared float g_query[32];
groupshared float g_scores[64];
groupshared float g_weights[64];

LRM_NUMTHREADS(attention_gqa_causal)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // The grid is exactly (rows, q_heads), so this never returns; it is the kernel rule that a group with no work
    // returns, and it returns for the whole group, before any barrier.
    if (gid.x >= lrm_args.rows || gid.y >= lrm_args.q_heads)
        return;

    uint i = lrm_args.rows - 1u - gid.x;
    uint h = gid.y;
    uint t = tid.x;
    uint hd = lrm_args.head_dim;
    uint kvh = h / (lrm_args.q_heads / lrm_args.kv_heads);
    uint keys = lrm_args.past + i + 1u;
    uint blocks = (keys + KeysPerBlock - 1u) / KeysPerBlock;
    uint kvStride = lrm_args.kv_heads * hd;
    uint kvHead = kvh * hd;
    uint qBase = ((lrm_args.q_row0 + i) * lrm_args.q_heads + h) * hd;
    float scale = lrm_args.scale;

    lrm_tensor_t qt = lrm_tensor(lrm_args.q);
    lrm_tensor_t kt = lrm_tensor(lrm_args.k);
    lrm_tensor_t vt = lrm_tensor(lrm_args.v);
    lrm_tensor_t yt = lrm_tensor(lrm_args.y);

    // Staging of a full chunk: dimension stageDim of keys stageKey, stageKey + 2, ...
    uint stageDim = t % ChunkDims;
    uint stageKey = t / ChunkDims;
    // Staging of the narrower last chunk (`tail` dimensions, 0 when 32 divides head_dim): thread t starts at key
    // t / width, dimension t % width, and each step of 64 elements adds tailStep keys and tailStepDim dimensions, with
    // a carry into the key.
    uint tail = hd % ChunkDims;
    uint tailWidth = max(tail, 1u);
    uint tailStep = KeysPerBlock / tailWidth;
    uint tailStepDim = KeysPerBlock - tailStep * tailWidth;
    uint tailKey = t / tailWidth;
    uint tailDim = t - tailKey * tailWidth;

    for (uint d0 = 0u; d0 < hd; d0 += DimsPerPass)
    {
        uint da = d0 + t;
        uint db = d0 + HalfPass + t;
        bool hasA = da < hd;
        bool hasB = db < hd;

        float m = 0.0f;
        float l = 0.0f;
        float oa = 0.0f;
        float ob = 0.0f;

        for (uint b = 0u; b < blocks; ++b)
        {
            uint j0 = b * KeysPerBlock;
            uint count = min(KeysPerBlock, keys - j0);
            uint blockRow = j0 * kvStride + kvHead;

            // The score of key j0 + t (thread t < count), its dot accumulated chunk by chunk in increasing d.
            float acc = 0.0f;
            for (uint c0 = 0u; c0 < hd; c0 += ChunkDims)
            {
                uint width = min(ChunkDims, hd - c0);
                uint kAt = blockRow + c0;
                if (c0 != 0u)
                    GroupMemoryBarrierWithGroupSync();
                if (width == ChunkDims)
                {
                    uint r = stageKey;
                    for (; r + (Batch - 1u) * KeysPerStep < count; r += Batch * KeysPerStep)
                    {
                        float staged[16];
                        [unroll]
                        for (uint u = 0u; u < Batch; ++u)
                            staged[u] = lrm_load_f32(kt, kAt + (r + KeysPerStep * u) * kvStride + stageDim);
                        [unroll]
                        for (uint w = 0u; w < Batch; ++w)
                            g_keys[(r + KeysPerStep * w) * ChunkStride + stageDim] = staged[w];
                    }
                    for (; r < count; r += KeysPerStep)
                        g_keys[r * ChunkStride + stageDim] = lrm_load_f32(kt, kAt + r * kvStride + stageDim);
                }
                else
                {
                    uint key = tailKey;
                    uint dim = tailDim;
                    for (; key < count; key += tailStep)
                    {
                        g_keys[key * ChunkStride + dim] = lrm_load_f32(kt, kAt + key * kvStride + dim);
                        dim += tailStepDim;
                        if (dim >= width)
                        {
                            dim -= width;
                            key += 1u;
                        }
                    }
                }
                if (t < width)
                    g_query[t] = lrm_load_f32(qt, qBase + c0 + t);
                GroupMemoryBarrierWithGroupSync();
                if (t < count)
                {
                    uint row = t * ChunkStride;
                    for (uint dd = 0u; dd < width; ++dd)
                        acc = mad(g_query[dd], g_keys[row + dd], acc);
                }
            }
            float s = acc * scale;
            g_scores[t] = s;
            GroupMemoryBarrierWithGroupSync();

            float blockMax = g_scores[0];
            for (uint jm = 1u; jm < count; ++jm)
                blockMax = max(blockMax, g_scores[jm]);
            if (b == 0u)
                m = blockMax;
            float mNew = max(m, blockMax);
            float alpha = exp(m - mNew);
            m = mNew;

            g_weights[t] = t < count ? exp(s - mNew) : 0.0f;
            GroupMemoryBarrierWithGroupSync();

            float blockSum = 0.0f;
            float accA = 0.0f;
            float accB = 0.0f;
            uint j = 0u;
            for (; j + Batch <= count; j += Batch)
            {
                float wts[16];
                float valA[16];
                float valB[16];
                uint vAt = blockRow + j * kvStride;
                [unroll]
                for (uint u = 0u; u < Batch; ++u)
                {
                    wts[u] = g_weights[j + u];
                    valA[u] = 0.0f;
                    valB[u] = 0.0f;
                    if (hasA)
                        valA[u] = lrm_load_f32(vt, vAt + u * kvStride + da);
                    if (hasB)
                        valB[u] = lrm_load_f32(vt, vAt + u * kvStride + db);
                }
                [unroll]
                for (uint w = 0u; w < Batch; ++w)
                {
                    blockSum += wts[w];
                    accA = mad(wts[w], valA[w], accA);
                    accB = mad(wts[w], valB[w], accB);
                }
            }
            for (; j < count; ++j)
            {
                float e = g_weights[j];
                uint vAt = blockRow + j * kvStride;
                blockSum += e;
                if (hasA)
                    accA = mad(e, lrm_load_f32(vt, vAt + da), accA);
                if (hasB)
                    accB = mad(e, lrm_load_f32(vt, vAt + db), accB);
            }

            l = mad(l, alpha, blockSum);
            oa = mad(oa, alpha, accA);
            ob = mad(ob, alpha, accB);
        }

        if (hasA)
            lrm_store_f32(yt, qBase + da, oa / l);
        if (hasB)
            lrm_store_f32(yt, qBase + db, ob / l);
    }
}
