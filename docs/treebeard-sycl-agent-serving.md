# Treebeard SYCL agent-serving dispatch

This note describes the two decode-path changes in commit `0424f677f`. The
target is the fixed 262144-context, 12-slot Qwen3.6 35B A3B serving profile on
an Intel Arc Pro B70. The optimization does not change quantization, routing
IDs, accumulation order within a subgroup, or output tensor layout.

## Runtime flowchart

```mermaid
flowchart TD
    A[Decode graph submits matrix operation] --> B{Operation kind}

    B -->|MUL_MAT| C{Reorder optimization allowed}
    C -->|No| D[Use existing non-reordered path]
    C -->|Yes| E{Right-hand columns}
    E -->|1 through 8| F[Use existing reordered MMVQ kernel]
    E -->|12| G{Left tensor is Q8_0 and 12-column gate enabled}
    E -->|Other| H[Use wide fallback in chunks of at most 8]
    G -->|Yes| I[Dispatch one reordered ncols=12 MMVQ kernel]
    G -->|No| H

    B -->|MUL_MAT_ID| J[Read output rows nrows]
    J --> K{Valid subgroup override set}
    K -->|Yes| L[Use GGML_SYCL_MMID_WG_SUBGROUPS]
    K -->|No| M{nrows at least 1024}
    M -->|Yes| N[Use 4 subgroups per workgroup]
    M -->|No| O[Use 1 subgroup per workgroup]
    L --> P[Launch token x expert x row-block grid]
    N --> P
    O --> P
    P --> Q[Each subgroup owns one independent output row]
    Q --> R[Subgroup lanes compute and reduce the row dot product]
    R --> S[Subgroup leader stores the row result]

    D --> T[Continue decode]
    F --> T
    H --> T
    I --> T
    S --> T
```

Text fallback:

```text
decode operation
|-- MUL_MAT
|   |-- reorder unavailable ----------------------> existing path
|   |-- ncols 1..8 -------------------------------> existing reordered MMVQ
|   |-- ncols 12 + Q8_0 + gate enabled ----------> one 12-column MMVQ launch
|   `-- otherwise --------------------------------> wide 8-column chunks
`-- MUL_MAT_ID
    |-- valid subgroup override ------------------> requested subgroup count
    |-- nrows >= 1024 ----------------------------> 4 subgroups/workgroup
    `-- nrows < 1024 -----------------------------> 1 subgroup/workgroup
        `-- each subgroup computes one output row -> subgroup leader stores
```

## Work mapping

For fused MoE matrix-vector work, let `S` be the selected number of subgroups
per workgroup and let the hardware subgroup width be 32. The launch uses:

```text
global groups = (tokens, experts-or-groups, ceil(nrows / S))
local range   = (1, 1, S * 32)
row           = group_z * S + subgroup_linear_id
```

The old launch represented independent output rows through a second local
dimension. The new launch packs those rows as independent subgroups in the
third local dimension. All lane-local block traversal, XOR reduction, and
leader-only store behavior remain subgroup-scoped. The mapping is shared by
the plain, reordered, and grouped-reordered `MUL_MAT_ID` kernels.

The automatic policy keeps a single subgroup for smaller projections and uses
four for projections with at least 1024 output rows. Four was selected by
matched parent/candidate controls at the release serving shape. Larger values
remain available for controlled experiments but are not the automatic path.

## Control gates

| Variable | Meaning |
| --- | --- |
| `GGML_SYCL_DISABLE_MMVQ_12COL=1` | Disable direct Q8_0 12-column reorder and use the wide fallback. |
| `GGML_SYCL_MMID_WG_SUBGROUPS=1` | Reproduce the prior one-row-per-workgroup MMID geometry. |
| `GGML_SYCL_MMID_WG_SUBGROUPS=2,4,8,16,32` | Force an experimental packed subgroup count. |

Unset or zero `GGML_SYCL_MMID_WG_SUBGROUPS` selects the automatic 1-or-4
policy. Any unsupported value also falls back to automatic selection.

## Correctness invariants

- The 12-column fast path is restricted to reordered Q8_0 weights.
- Shapes outside the exact eligibility checks retain the existing fallback.
- Each MMID subgroup owns one output row; no row is shared by subgroups.
- Routing IDs and token/expert addressing are unchanged.
- Bounds checks still suppress the tail rows in the last row block.
- Reduction and storage remain subgroup-local and leader-only.
- Both optimizations have environment controls for same-binary attribution.

## Release-gate flowchart

```mermaid
flowchart LR
    A[Committed source] --> B[Build relocatable runtime]
    B --> C[Backend-op CPU-oracle checks]
    C --> D[StateTree and tool-boundary tests]
    D --> E[Native pp4096 and tg128]
    E --> F[262K 12-slot packaged server]
    F --> G[1,2,4,6,8,12 agent Pareto]
    G --> H[Standard API matrix]
    H --> I[Full 69-case 12-agent tool bench]
    I --> J[12-agent poison gate]
    J --> K[GPU fault scan]
    K --> L[Archive and checksum validation]
    L --> M[Private Hub staging]
    M --> N[Immutable-revision clean download]
    N --> O[Public RC3]

    C -->|Fail| X[Reject and restore RC2]
    D -->|Fail| X
    E -->|Fail| X
    F -->|Fail| X
    G -->|Fail| X
    H -->|Fail| X
    I -->|Fail| X
    J -->|Fail| X
    K -->|Fail| X
    L -->|Fail| X
    N -->|Fail| X
```

The benchmark service uses isolated port 8098. Production RC2 on port 8093 is
never used for inference during the gate and is restored and attested on every
exit path.
