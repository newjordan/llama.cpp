# Turbo Unified-KV Slot Fork - 2026-07-09

## Status

The isolated `turbo-combined` worktree now has a queued, in-memory
`POST /slots/{source}?action=fork` primitive and a breakout harness backend that
uses it. A controlled 35B/B70 gate was run with the candidate on `:8098`; the
exact `a2edfe66f` production command is restored and healthy on `:8093`, and its
full Hydra surface was health-promoted and route-probed after restoration.

The fork primitive passes its state-transfer contract. After fixing stale
server checkpoint metadata, the clean rerun made cloning 28.9x faster, kept
equal-token 12-way decode effectively flat, and reduced complete flow wall time.
This is an R&D result, not yet a production rollout decision.

## Direct Evidence

Rebuilt `llama-server` checks used the local Qwen 3.5 0.8B Q8 model, whose
hybrid/recurrent memory path matches the deployed model family
(`general.architecture=qwen35moe`, SSM metadata, full-attention interval 4).

- Canonical ID validation rejected five alias/out-of-range fork cases without
  changing source or destination state.
- A valid source plus two-destination fork reused all six prefix tokens on all
  three sequences.
- Post-fork checkpoints preserved repeated hybrid prefix reuse on source and
  both siblings (`cache_n=9`) after cross-sequence checkpoint restores.
- With every slot reserved, an automatic request reached
  `requests_deferred=1`; erasing slot 1 woke it on slot 1 and returned the
  deferred metric to zero.
- Reserved state survived the idle prompt-cache sweep; source and both branches
  reused six prefix tokens.
- A failed restore released only its destination reservation.
- Reservations survived a 1.5 second wait past a 1 second idle-sleep threshold
  while a request was deferred.
- Non-unified fork returned HTTP 501.

The first harness smoke artifact is:

```text
/home/frosty40/turbo/results/slot-fork-harness-smoke/20260709T192859Z-3350744.task000.json
```

It records a 92-token prefix forked to three destinations in 1.295 ms, four
successful branch requests, one fanout wave, and zero cleanup errors. All four
slots were unreserved and empty after the run.

## First 35B B70 Gate

Artifacts:

```text
/home/frosty40/turbo/results/slot-fork-35b-gate/20260709T195555Z
```

The candidate used Qwen3.6-35B-A3B Q5_K_XL on the B70 with the production
`-kvu -np 12 -c 262144 -fa on -ctk f16 -ctv f16` shape. It also used
`--slots --metrics --slot-save-path ... --no-cache-idle-slots`. An initial file
sample without the last flag was discarded: the server log proved that idle
restored slots were moved into prompt RAM cache and cleared.

The deterministic lane used one fixed task, greedy sampling, no baseline,
verifier, scorer, or repair, and a verified erase of all 12 slots before every
invocation. It ran two self-controls per backend and five order-balanced
file/fork pairs (`n=7` per backend).

Clone-only signals remain usable because `clone_wall_s` excludes prefix and
branch evaluation:

| Signal | File save/restore | In-memory fork | Result |
|---|---:|---:|---:|
| Client clone wall, mean | 200.692 ms | 61.559 ms | fork 3.260x faster |
| Server clone time, mean | 195.003 ms | 60.901 ms | fork 3.202x faster |

All 14 clone contracts passed: source and destination token counts matched,
destination IDs were exact, and every branch reported `cache_n=124`. Each file
clone wrote 68,407,184 bytes and read 752,479,024 bytes; fork performed neither
operation.

The first branch comparison is invalid. `action=erase` removed sequence state
and prompt tokens but left `server_prompt.checkpoints`; successful file restore
also left those checkpoints in place. File samples therefore inherited hidden
62.8 MiB recurrent checkpoints while fork samples replaced or cleared them.
The server now clears checkpoint/data metadata on erase and restore and exposes
`n_prompt_checkpoints` through `/slots`; the local hybrid regression passes.

K/K self-controls still showed branch-token differences under 12-way greedy
execution, so exact output stability remains a separate gate. A longer-output
lane was rejected because generation lengths diverged; `--ignore-eos` is now
available for the equal-length rerun. No branch throughput or full-flow value
from this first gate should be used as an optimization claim.

## Clean 35B B70 Rerun

Artifacts and generated summary:

```text
/home/frosty40/turbo/results/slot-fork-35b-gate/20260709T201929Z-checkpoint-fix
/home/frosty40/turbo/results/slot-fork-35b-gate/20260709T201929Z-checkpoint-fix/rerun-summary.json
```

The rerun used the same model and serving shape, with prompt RAM cache disabled.
All 18 pre-run resets proved zero processing, reservations, fork owners, prompt
tokens, and prompt checkpoints. All 18 file/fork clone contracts passed.

The short lane used two self-controls and five order-balanced pairs per backend
(`n=7`):

| Signal | File save/restore | In-memory fork | Result |
|---|---:|---:|---:|
| Client clone wall, mean | 200.398 ms | 6.926 ms | fork 28.935x faster |
| Server clone time, mean | 193.880 ms | 6.384 ms | fork 30.370x faster |
| Branch-wave wall, mean | 3.014 s | 3.037 s | effectively flat |
| Complete flow wall, mean | 4.907 s | 4.707 s | fork 4.26% faster |

The forced lane used `--ignore-eos --no-stop-strings`; every one of the 48
branch requests generated exactly 128 tokens. Across two order-balanced pairs:

| Signal | File save/restore | In-memory fork | Result |
|---|---:|---:|---:|
| Branch aggregate rate | 158.482 tok/s | 158.048 tok/s | fork -0.27% |
| Branch-wave wall, mean | 9.693 s | 9.719 s | effectively flat |
| Complete flow wall, mean | 12.827 s | 12.554 s | fork 2.17% faster |

Each file clone still wrote 68,407,184 bytes and read 752,479,024 bytes. Fork
avoided both. The stale-checkpoint fix also removed hidden cleanup work from the
fork transaction, reducing its measured mean from the contaminated 61.559 ms
to 6.926 ms.

Exact branch-token output is not stable enough for backend attribution: both
F/F and K/K self-controls differed in at least one branch. All five cross-backend
pairs produced identical final tokens, but their intermediate branch arrays did
not. Deterministic objective validation and final-answer parity remain required
quality gates.

## Validation

- `cmake --build build --target llama-server -j 16`
- Direct unified-KV fork, branch, erase, restore-failure, metrics, deferred,
  idle-cache, sleep, and strict-ID checks
- `pytest --noconftest -v tools/server/tests/unit/test_slot_fork.py` with the
  local Qwen 3.5 0.8B Q8 model (`9 passed`)
- `pytest --noconftest -q tools/server/tests/unit/test_ignore_eos.py` with the
  same local model (`3 passed`), including explicit empty-stop override
- `python3 tests/test_turbo_speculative_breakout.py` (`16 passed`)
- `python3 tests/test_turbo_kv_page_ablate.py` (`16 passed`)
- Python compilation and `git diff --check`

The standard repository pytest session fixture could not bootstrap because
this local build was configured without HTTPS support and the fixture downloads
its model set at session start. The focused server test supports an explicit
local model and ran all nine scenarios against the rebuilt binary.

## Next Gate

Repeat forced-token measurements at 1k, 8k, and 32k-plus shared-prefix lengths
and with at least five pairs. Add sparse/fragmented slot layouts and validate a
fork after bounded recurrent rollback, including the recurrent-state index.
Then rerun the deterministic objective suite with final-answer parity and
quality gates. Add busy-slot, context-limit, LoRA, speculative, and multimodal
rejection tests before considering a dedicated canary endpoint. Do not replace
`:8093` from this worktree yet.
