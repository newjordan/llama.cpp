# Turbo Host and Kernel Baseline - 2026-07-10

## Decision

A custom OS kernel is not the highest-value next improvement. The measured
server-runtime StateTree retention work has direct correctness and memory
benefits and is ready for its final matched B70 gate. The largest host-side
reliability defect is operational: the current B61 production unit is
runtime-only, while an older build remains enabled for the next user-manager
start.

No production service, governor, boot setting, IRQ, sysctl, or kernel was
changed while collecting this baseline. Production remained healthy on
`:8093`.

Raw machine-readable evidence:

```text
/home/frosty40/turbo/results/host-baseline/20260710-current.json
```

Reproduce the capture with:

```bash
python3 scripts/turbo-host-baseline.py \
  --energy-sample-seconds 5 \
  --sched-pipe-repeats 5 \
  --output /home/frosty40/turbo/results/host-baseline/20260710-current.json
```

The collector is read-only apart from writing the requested JSON artifact and
the optional transient scheduler workload. Its unit tests cover memory-field
parsing, swap-owner ranking, energy deltas, scheduler result parsing, and PCIe
state capture.

Compare two captures with:

```bash
python3 scripts/turbo-host-compare.py REFERENCE.json CANDIDATE.json
```

The comparator rejects hardware or production-workload mismatches, unhealthy
captures, a host PCIe-link downgrade, insufficient scheduler samples, more
than 10% scheduler median regression, more than 5% Xe card/package energy
regression, missing energy data, and nonzero candidate memory PSI. A canonical
self-comparison passes all 14 default gates:

```text
/home/frosty40/turbo/results/host-baseline/20260710-self-comparison.json
```

## Baseline

| Area | Observed state |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | `6.17.0-35-generic`, dynamic preemption build, 1000 Hz, high-resolution timers |
| CPU | Ryzen 9 5950X, 16 cores / 32 threads, one NUMA node |
| CPU policy | `amd-pstate-epp`, `performance` governor and EPP, boost enabled |
| Idle policy | only `POLL` and `C1` exposed; `processor.max_cstate=1` is on the kernel command line |
| Firmware | ASUS ROG Crosshair VIII Dark Hero, BIOS 5302 dated 2025-10-03 |
| Memory | 62.7 GiB total, 58.5-58.7 GiB available, zero current memory PSI |
| Production cgroup | zero swap, zero memory events, zero OOM/high/max events |
| THP | `madvise`; no static huge pages; zswap disabled |
| Root NVMe | scheduler `none`, 128 KiB read-ahead, `rq_affinity=1` |
| B70 host link | PCIe 4.0 x16 active from the AMD root port through the Intel bridge |
| B70 driver | in-tree `xe`; runtime active while the model is loaded |
| Xe IRQ | IRQ 159, effective CPU 27, all CPUs allowed |
| Production | B61 `a2edfe66f`, 12 slots, 262144 context, health `ok` |

The GPU and audio functions expose 2.5 GT/s x1 links behind the B70 bridge.
Those are internal endpoint links; the host-facing bridge and AMD root port are
both active at 16.0 GT/s x16. There is no evidence of a host PCIe-width
bottleneck.

Four five-second loaded-idle energy windows were stable:

| Counter | Observed range |
| --- | ---: |
| Xe card | 44.868-45.157 W |
| Xe package | 26.132-26.208 W |

The five `perf bench sched pipe -l 100000` samples embedded in the canonical
JSON measured 3.311-3.726 us/op, with a 3.551 us/op median. This is a baseline,
not an A/B result.

System-wide swap use was about 1.9 GiB, but current memory pressure was zero and
the production Turbo cgroup had never swapped. The largest swapped residents
were long-lived Letta and CPU embedding processes. Lowering swappiness or
forcing swap-in is therefore not justified by current evidence.

## Ranked Improvement Paths

| Rank | Path | Evidence and disposition |
| ---: | --- | --- |
| 1 | StateTree bounded retention | Exact admission, eviction, expiry, race, churn, and matched CPU gates pass. Run the already-prepared matched B70 windows after explicit downtime approval. |
| 2 | Persist the current production unit | Direct reboot-reliability defect. Stage is ready below; install only with approval because it changes next-boot behavior. |
| 3 | Restore deeper CPU idle states | Plausible idle-efficiency win because the host is forced to C1. Requires a controlled reboot A/B and whole-system power measurement before adoption. |
| 4 | Re-enable PCIe ASPM | Plausible idle-power win, but higher B70 stability risk. Test separately only after the C-state A/B. |
| 5 | IRQ affinity or `irqbalance` | No imbalance or throughput evidence. Leave CPU 27 placement unchanged until a B70 load trace shows an IRQ-bound lane. |
| 6 | Swap, THP, static huge pages | No active pressure and production swap is zero. Leave current policy unchanged. |
| 7 | Custom kernel | No missing kernel capability, driver fault, scheduler failure, or workload regression supports the maintenance burden. Do not pursue now. |

## Reboot-Safe Production Stage

The current live unit is:

```text
/run/user/1000/systemd/user/turbo-head-a2edfe66f-rollback-8093.service
```

It is static, runtime-only, and has `Restart=no`. User lingering is enabled.
At the same time, this older unit is enabled for `default.target`:

```text
~/.config/systemd/user/turbo-cd395a152.service
```

Consequently, a reboot or fresh user-manager start would lose the current B61
definition and select the older `cd395a152` deployment.

An install-ready, rate-limited `Restart=on-failure` definition for the exact
current B61 command is staged at:

```text
docs/ops/turbo-head-a2edfe66f-rollback-8093.service
```

The staged unit intentionally preserves the current binary, model, model
alias, SYCL environment, context, slot count, batch sizes, and port. It adds
only persistence, bounded failure restart, a stop timeout, and an install
target. It has not been copied into `~/.config`, enabled, or loaded.

Run the read-only promotion preflight immediately before seeking or using
authorization:

```bash
python3 scripts/turbo-production-preflight.py \
  --output /home/frosty40/turbo/results/host-baseline/production-preflight.json
```

The current preflight passes all 14 checks and explicitly reports that it made
no mutation:

```text
/home/frosty40/turbo/results/host-baseline/20260710-production-promotion-preflight.json
```

It verifies the exact live/staged `ExecStart`, systemd syntax, bounded restart
contract, live executable SHA256, model/build/slot/context identity, health,
old enabled rollback unit, absent persistent destination, and stable production
PID. Any drift makes it return nonzero.

The later promotion must:

1. Recheck the live binary/model hashes and `/props` identity.
2. Verify the staged unit with `systemd-analyze --user verify`.
3. Copy it to the user unit directory without replacing or restarting the live
   process.
4. Enable the B61 unit and disable the inactive older unit.
5. Reload the user manager and prove the production `MainPID` did not change.
6. Recheck health, properties, enabled-unit state, and exact rollback commands.

Rollback is to disable the staged B61 unit, re-enable the untouched older unit,
remove the staged persistent copy, and reload the user manager. None of these
steps should stop the currently running process; nevertheless, the next-boot
selection is production state and requires explicit authorization.

## Boot-Policy Experiment Gate

Do not combine boot changes. The first reboot A/B should remove only
`processor.max_cstate=1`, retain the performance governor and all other command
line settings, then compare:

- wall power at loaded idle and unloaded idle;
- the scheduler microbaseline above;
- matched parent/candidate server throughput and latency;
- B70 health, Xe errors, PCIe link state, and StateTree correctness.

Only if that passes should a separate A/B restore the default PCIe ASPM policy.
`idle=nomwait` should be evaluated in its own later lane. Every boot-policy
experiment needs an exact boot-entry rollback and a timed automatic reboot
fallback. A custom kernel should be reconsidered only if these stock-kernel
experiments reveal a specific capability or driver limitation that cannot be
fixed more narrowly.

For the first C-state experiment, use the stricter comparison form and supply
wall power measured over the same stable interval on both boots:

```bash
python3 scripts/turbo-host-compare.py REFERENCE.json CANDIDATE.json \
  --require-different-boot \
  --require-cmdline-change \
  --expect-removed-cmdline-token processor.max_cstate=1 \
  --require-wall-power \
  --reference-wall-watts REFERENCE_WATTS \
  --candidate-wall-watts CANDIDATE_WATTS \
  --min-wall-power-improvement-pct 2
```

Passing proves identity and the configured non-regression gates plus at least
2% measured wall-power improvement. It does not replace the matched server
workload or Xe error checks listed above.
