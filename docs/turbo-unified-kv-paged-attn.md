# Turbo Experiment: Unified-KV Paged Attention for `-kvu`

## Status

This is an R&D experiment, not an optimization claim.

Current state:

- Design mapped against the current `-kvu` KV-cache and SYCL FATTN paths.
- `LLAMA_KV_PAGE_PROBE=1` instrumentation exists and is off by default.
- `LLAMA_KV_COMPACT_ATTN=1` enables a compact prepack prototype that gathers
  active physical KV rows into compact K/V tensors before attention.
- `LLAMA_KV_COMPACT_ATTN=2` additionally logs compact row bounds for debugging.
- `LLAMA_KV_INDEXED_FATTN=1` implies compact row maps and enables an opt-in
  SYCL vector flash-attention decode path that reads physical K/V rows directly
  through the row map when the live rows are fragmented and K/V is not f16.
  Dense-prefix decode bypasses both compact gather and row indexing. Fragmented
  f16 decode keeps compact gather + TILE until there is an indexed TILE path.
  `LLAMA_KV_INDEXED_FATTN=2` forces indexed decode for kernel smoke.
- `scripts/turbo-kv-page-ablate.py` can run baseline/probe/compact/indexed
  ablations and parse KV geometry from server logs.

Not done:

- No fused dense prepack kernel exists yet; the current prototype uses ggml row
  gathers and the existing attention path.
- The indexed SYCL path is decode-only and row-index based; it is not yet a full
  page-table kernel with eviction, compaction, or prefix sharing metadata.
- No throughput, stability, or quality claim is valid without the ablation set
  below.

## Goal

Treat the unified KV cache as a GPU page cache instead of a single dense row
range that attention must scan from row 0 to `used_max_p1()`.

For the current serving shape:

```text
Qwen3.6-35B-A3B Q5_K_XL
Intel Arc Pro B70
SYCL flash attention
-kvu
-np 12
-c 262144
```

the win is not "262k context" by itself. The win would be keeping that capacity
while making fragmented, reused, multi-slot long-context service behave like a
compact hot working set.

## Current Code Shape

Relevant files:

- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `src/llama-graph.cpp`
- `ggml/src/ggml-sycl/fattn-vec.hpp`
- `ggml/src/ggml-sycl/fattn-common.hpp`

Important current mechanics:

- `llama_kv_cache` allocates K and V tensors as `[n_embd_gqa, kv_size, n_stream]`.
- With `-kvu`, `n_stream == 1`, so every slot shares one physical KV row pool.
- `llama_kv_cells` already tracks logical occupancy, sequence ids, positions,
  and holes at the cell level.
- `find_slot(..., cont=false)` can assign non-contiguous physical rows.
- `get_n_kv()` still pads attention up to a dense prefix based on
  `cells.used_max_p1()`.
- `get_k()` and `get_v()` expose dense ggml views. They do not expose a page
  table or logical-to-physical indirection.
- SYCL flash attention receives K/V base pointers plus dense strides. The inner
  vector path advances with `K += stride` and `V += stride`; it cannot currently
  follow pages.

Practical consequence: once a high physical row has been touched, attention can
keep scanning a large dense prefix even if much of it is empty, stale, masked,
or fragmented.

## Proposed Page Model

Use fixed-size KV pages inside the existing unified physical pool.

Initial page size candidates:

```text
128 tokens: lower internal waste, more page-table traffic
256 tokens: matches current n_pad floor and keeps tables small
512 tokens: lower metadata overhead, worse reuse granularity
```

Start with 256 because `get_n_kv()` already pads to at least 256 and the
current 262144-token pool becomes exactly 1024 pages.

Core metadata:

```text
physical_page_id -> rows [page_id * page_size, (page_id + 1) * page_size)
logical_page_id  -> page order visible to a sequence or slot
page_table[logical_page_id] = physical_page_id
page_used[physical_page_id] = live token count or bitmap
page_owner/refcount        = slot/sequence/prefix sharing state
```

The key distinction is that logical attention order stays compact even when
physical rows are reused out of order.

## Two Viable Paths

### Path A: Dense Prepack MVP

Before attention, gather active logical pages into a compact scratch K/V buffer,
then call the existing dense flash-attention kernel unchanged.

Pros:

- Smallest correctness step.
- Does not require changing `GGML_OP_FLASH_ATTN_EXT`.
- Lets us measure whether fragmentation is actually hurting the 12-slot service.
- Can be guarded behind a Turbo-only env flag.

Cons:

- Adds a K/V copy before attention.
- Likely loses for full 262k scans unless the active/hot set is much smaller
  than the dense prefix.
- Scratch memory can be large if every slot is genuinely full.

Best use:

- Fragmented multi-agent service.
- Prefix reuse.
- Long-running server where `used_max_p1()` drifts high while active pages are
  sparse.

### Path B: True Paged SYCL Flash Attention

Add a page-table-aware attention path. The logical K loop remains ordered from
0 to `n_logical_kv`, but each logical row maps to a physical row:

```text
logical_row = k
page        = logical_row / page_size
offset      = logical_row % page_size
phys_row    = page_table[page] * page_size + offset
K_row       = K_base + phys_row * nb11
V_row       = V_base + phys_row * nb21
mask_row    = mask_base + logical_row
```

Pros:

- Avoids dense compaction copies.
- Preserves compact logical attention even with fragmented physical pages.
- Opens the door to page eviction, prefix sharing, and slot remapping.

Cons:

- Requires a new ggml/SYCL ABI surface or a Turbo-only op variant.
- Page-table reads sit inside the hottest attention loop.
- Needs careful specialization so the B70 path does not regress dense,
  non-fragmented decode.

Best use:

- Real long-context service where slots churn and share a huge pool.
- Future single-agent breakout/recombination work that needs cheap branch pages.

## Kernel ABI Problem

The current SYCL vector attention kernel only receives:

```text
Q, K, V, mask, sinks, KV_max, dst, dst_meta
scale, bias, dimensions, dense strides
```

There is no source slot for:

- page table
- per-page lengths
- per-page logical position base
- compact logical row count independent of physical `kv_size`

So the first serious code decision is ABI shape:

```text
Option 1: add GGML_OP_FLASH_ATTN_EXT_PAGED
Option 2: extend GGML_OP_FLASH_ATTN_EXT with extra optional src tensors
Option 3: keep ggml op unchanged and prepack into dense scratch
```

For Turbo R&D, start with option 3, then move to option 1 if the probe shows
enough headroom.

Current Turbo branch state:

- Option 3 exists as `LLAMA_KV_COMPACT_ATTN=1`.
- A narrow option 2 exists as `LLAMA_KV_INDEXED_FATTN=1`: `GGML_OP_FLASH_ATTN_EXT`
  carries an optional I32 row-index source at `src[5]`, and the SYCL vector
  kernel uses it for fragmented non-f16 decode K/V loads.
  `LLAMA_KV_INDEXED_FATTN=2` force-enables the same kernel path for validation
  on f16 and dense smoke cases.
- The row index is physical-row based, not a page table. It is a stepping stone
  toward page-aware FATTN, not the final design.

## Implementation Milestones

### Phase 0: Measure the Pain

Add instrumentation around:

- `llama_kv_cache::get_n_kv()`
- `llama_kv_cache::find_slot()`
- `llama_kv_cache::apply_ubatch()`

Metrics:

```text
kv_size
used_cells
used_max_p1
hole_count_below_used_max
largest_free_run
allocated_pages
live_pages
pages_touched_by_current_ubatch
```

Success condition:

The 12-slot 262k service shows a meaningful gap between dense prefix scanned
and live/hot KV rows under realistic agent churn.

### Phase 1: Page Directory on Host

Introduce a Turbo-only page directory beside `llama_kv_cells`.

No kernel changes yet. It should be able to answer:

```text
sequence/slot -> ordered logical page ids
logical page  -> physical page
physical page -> live cells/refcount
```

This phase should not change model output.

### Phase 2: Dense Prepack Experiment

When `-kvu` and the Turbo flag are enabled:

1. Build a compact logical page list for the current ubatch.
2. Gather K/V pages into a scratch dense K/V tensor.
3. Build the KQ mask against compact logical rows.
4. Call existing `ggml_flash_attn_ext`.

Accept only if correctness matches the normal path and fragmented workloads
recover more time than the gather copy costs.

### Phase 3: Paged SYCL FATTN

Create a page-aware vector flash-attention path for the B70 case.

Targets:

- `Q->ne[1] == 1` decode first.
- f16 K/V first, because current deployed max-context service uses
  `-ctk f16 -ctv f16`.
- Preserve existing dense path for non-paged and non-fragmented cases.

Address calculation changes inside the K/V loop only. Mask indexing should
remain logical so causality and sequence separation do not depend on physical
row layout.

Current prototype:

```text
LLAMA_KV_INDEXED_FATTN=1
```

For `Q->ne[1] == 1` decode, the graph keeps K/V as cache views and passes the
compact logical-to-physical row map into SYCL FATTN. The vector kernel resolves
each logical K/V column to `phys_row = kv_idxs[sequence, logical_row]` before
loading K and V. Prefill and unsupported FATTN shapes fall back to compact
gather.

Mode details:

```text
LLAMA_KV_INDEXED_FATTN=1: auto, only fragmented non-f16 decode uses row indexing
LLAMA_KV_INDEXED_FATTN=2: force indexed decode for kernel validation
```

## Risks

- Page-table indirection can cost more than it saves for dense, sequential
  contexts.
- Host mask construction may become the bottleneck if we rebuild compact masks
  naively for every token.
- Prepack can burn memory bandwidth and scratch memory.
- Full state save/restore must preserve page metadata, not just cells.
- Prefix sharing/refcounts make eviction correctness harder.

## Non-Goals

- This is not intended for the upstream fused-top-k PR lane.
- Do not mix this with the llama.cpp review branch.
- Do not change default llama.cpp behavior.
- Do not advertise a context-length win. This is a serving efficiency and
  stability experiment.

## First Code Step

The safest first code step is instrumentation, not a kernel edit:

```text
LLAMA_KV_PAGE_PROBE=1
```

should print compact one-line KV geometry at decode boundaries, enough to answer
whether a paged cache is worth taking into SYCL.

## Ablation Standard

Do not claim this experiment is beneficial until the following matrix has been
run and archived as JSONL plus logs.

Required variants:

```text
baseline: same binary, LLAMA_KV_PAGE_PROBE unset
probe:    same binary, LLAMA_KV_PAGE_PROBE=1
compact:  same binary, LLAMA_KV_COMPACT_ATTN=1
compact-probe: same binary, LLAMA_KV_COMPACT_ATTN=1, LLAMA_KV_PAGE_PROBE=1
indexed:  same binary, LLAMA_KV_INDEXED_FATTN=1
indexed-probe: same binary, LLAMA_KV_INDEXED_FATTN=1, LLAMA_KV_PAGE_PROBE=1
indexed-force: same binary, LLAMA_KV_INDEXED_FATTN=2
indexed-force-probe: same binary, LLAMA_KV_INDEXED_FATTN=2, LLAMA_KV_PAGE_PROBE=1
```

Required service shapes:

```text
np=1,  ctx=32768
np=4,  ctx=65536
np=12, ctx=262144
```

Required workload shapes:

```text
short prompts:  256-ish prompt tokens
mixed prompts:  256, 2048, 8192-ish prompt tokens
long prompts:   8192+ prompt tokens
generation:     64, 96, or 256 tokens
waves:          at least 3
repeats:        at least 5 for any claim
```

First harness:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
python3 scripts/turbo-kv-page-ablate.py \
  --bin /home/frosty40/builds/turbo-experimental-build/bin/llama-server \
  --model /home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  --ctx 262144 \
  --parallel 12 \
  --batch 8192 \
  --ubatch 1024 \
  --waves 3 \
  --repeats 5 \
  --prompt-lens 256,2048,8192 \
  --gen-tokens 96 \
  --out-dir /tmp/turbo-kv-page-ablate
```

The probe is useful only if it shows persistent waste such as:

```text
max_hole_ratio materially above zero
dense_pages materially greater than live_pages
n_kv materially greater than used
largest_free_run large enough to justify page remapping or compaction
```

Acceptance bar for moving to Phase 2:

- The probe overhead itself is measured by baseline vs probe and is small enough
  to trust the geometry.
- Fragmentation/dense-prefix waste repeats across at least five runs.
- The worst-case service shape is the real target shape: `np=12`, `ctx=262144`,
  `-kvu`, f16 K/V.
- The data identifies a concrete expected win for dense prepack or paged FATTN.

Rejection bar:

- If `used_max_p1`, `n_kv`, and live rows stay close under churn, paged KV is
  not the next kernel to build.
- If probe logging itself perturbs throughput enough to hide the signal, the
  probe must be made cheaper before continuing.
