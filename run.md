# Run status

Snapshot: **2026-08-13 23:25 UTC**. Target: all rollouts finished before **Aug 20**.

## SWE-Gym 4-model rollout

All four models run the same 1500-instance selection
(`eval_outputs/swegym_select_1500.txt`), so results are directly comparable.
Progress is **critic-passing** trajectories, not recorded rows: a row is written
whenever the run did not raise, including runs that ended with an empty patch or no
finish action, and those stay eligible for retry.

| model | run | workers | maxiter | critic-passing | left | rate/worker-hr | ETA |
|---|---|---|---|---|---|---|---|
| gpt-5-mini | — | — | 60 | **1500 / 1500 COMPLETE** | 0 | 4.98 | **done Aug 14 00:07** |
| qwen3-next-80b | — | — | 60 | **1497 / 1500 CONVERGED** | 3 | 5.85 | **done Aug 14 01:14** |
| kimi-k2.5 | `swegym-kimi25` | 5 | 100 | **547** / 1500 | 953 | 2.00 | Aug 18 |
| deepseek-v4-pro | `swegym-dv4pro` | 3 | 100 | **230** / 1500 | 1270 | **1.15 (degraded)** | **at risk — see below** |

**Both fast models are done.** gpt-5-mini finished clean at 1500/1500; qwen80b converged
at 1497/1500 after `MAX_STALLED=2` fruitless retry passes on 3 instances it could never
solve — which is exactly what that stall detector exists to terminate.

The handoff completed as designed: kimi25 3→4→5 and dv4pro 2→3, total 8 = budget.

**dv4pro is the deadline risk now.** Its throughput collapsed from 3.97 to ~1.15 per
worker-hour over the last few hours — it is stuck in a run of `python__mypy-*`
instances that fail and retry (452 -> 1201 -> 2034 s/it). At 1.15/wh even 3 workers
would not finish 1270 instances by Aug 20. If the rate does not recover once it clears
the mypy block, the options are to cap its target below 1500 or drop it.

**dv4pro was killed and restarted.** At 00:16 a kimi25 top-up coincided with dv4pro's
supervisor dying — log stopped mid-instance with no exit line, session gone 18s later.
All the kill paths in `restart_with` are note-scoped so the mechanism is unconfirmed.
Restarted at 00:27, and `restart_with` now snapshots live sessions and relaunches any
that vanish during a restart, so this class of failure self-heals rather than costing a
day unnoticed.

Rates are measured over the last 19h, so they include retry overhead to date. They stay
mildly optimistic for the tail — what's left at the end is what already failed once.
The previous snapshot's ETA for the top two was ~2h early for exactly that reason.

### The handoff, which is load-bearing for the deadline

qwen80b and gpt5mini both converge **within the next couple of hours**, freeing 5 of
the 10 workers. `rebalance.sh` hands them to the two runs still going, targeting
kimi25=5 and dv4pro=3 (8 total, the gateway budget):

| model | remaining at handoff | ETA after handoff |
|---|---|---|
| kimi-k2.5 | ~947 | **Aug 18 ~00:00** |
| deepseek-v4-pro | ~1264 | **Aug 18 ~11:00** |

Both land with ~2 days of margin. At their *current* worker counts neither would
(Aug 20 14:00 and 15:00), so the rebalancer is doing real work here, not bookkeeping.

### Health

- **Gateway**: 10 workers, no 429 storm (cumulative 429s over 3 days: 24/17/34/0).
  The observed-safe ceiling is the *sum* of workers across every live job on this box
  (per-IP WAF); 14 caused a storm on 2026-08-05.
- **Disk**: 244G free of 14T — tighter than yesterday's 611G. `disk_guard.sh` sweeps
  every ~10min and is holding a stable 230–250G band. Our own footprint is ~200G; the
  pressure is other tenants (1.29TB of images, mostly not ours).
- **dv4pro is confirmed good**: 228 passing / 229 recorded, 1 unusable row. It is the
  working replacement for deepseek-v4-flash, which is dead for agent work (81% of its
  rows were critic-unusable — its endpoint stops emitting tool calls under long
  contexts, and a short-context probe does *not* reproduce it).
- **kimi-k2.5 sped up**: 1.52 → 2.00 per worker-hour since the last snapshot.

## Upload / evaluation ledger (2026-08-14 19:15 UTC)

**SWE-Gym — nothing evaluated, nothing uploaded.** The harness works (vendored
`SWE-Gym/SWE-Bench-Fork` + `__`->`_s_` retag) but has never been run past a 2-instance
validation, and no `swegym_*` rollout dataset exists on HF.

| model | instances | critic-passing | status |
|---|---|---|---|
| gpt-5-mini | 1500 | **1500** | COMPLETE — ready to eval+push |
| qwen3-next-80b | 1500 | **1497** | CONVERGED — ready to eval+push |
| kimi-k2.5 | 593 | 590 | running @5 |
| deepseek-v4-pro | 291 | 291 | running @3 |
| deepseek-v4-flash | 189 | 98 | abandoned (81% rows unusable) |

**R2E-Gym uploaded (6):** `r2egym_opus45_1502i`, `r2egym_converted_1054i`,
`r2egym_converted_3231i`, `r2egym_deepseek_v4_flash_1390i` (982/1390, 70.7%),
`r2egym_gpt5mini_1500i` (926/1500, 61.7%), `r2egym_qwen3next80b_1500i` (663/1500, 44.2%).

**R2E-Gym unuploaded (4, all stopped, none evaluated):**

| model | instances | critic-passing | note |
|---|---|---|---|
| qwen3.5-35b-a3b | 524 | 165 | 69% of rows unusable — low value |
| kimi-k2.6 (azure) | 89 | 89 | clean but tiny |
| kimi-k2.5 | 89 | 89 | clean but tiny |
| kimi-k2.6 (nvidia) | 48 | — | tiny |

## Backups (2026-08-15 03:35) — everything irreplaceable is off the full disk

Verified byte-identical, with the last JSON line of each file confirmed parseable
(a torn append was the specific risk of writing to a full disk):

| what | where | size |
|---|---|---|
| gpt5mini 1500 / qwen80b 1500 / kimi25 662 rows | `/mnt/data1/gaokaizhang/swegym_backup_20260815` | 2.7G |
| rubric-discovery annotations + all 9 pairwise results | `/mnt/data1/gaokaizhang/rubric_discovery_data_20260815` | 7.7M |

`/mnt/data1` has 252G free. The rubric-discovery `data/` dir is gitignored, so those
annotations were NOT in the GitHub push and existed only on the failing disk.

## BLOCKER: the box is out of disk, and it is not our doing

124GB free of 14T (100%). `docker system df` shows **784 images / 1.35TB, of which 21
are ours**. The disk guard now reclaims **0** per sweep — there is nothing of ours left.

Effect: agent-server containers die mid-run (~300 recent `Remote conversation ended with
error` failures on kimi25 alone), so each instance costs several attempts. Throughput on
both live rollouts collapsed to **~1.1 critic-passing per worker-hour** (kimi25 was 2.00,
dv4pro 3.97). At that rate neither finishes by Aug 20: kimi25 needs ~16 days, dv4pro ~15.

Not fixable by tuning workers, and must not be fixed by pruning — the other 763 images
belong to other tenants with live containers.

## rubric-discovery (launched 2026-08-13 23:23)

Runs the three r2egym student comparisons. tmux `rubric-discovery`, serial over pairs,
each pair fanning out its two categories in parallel.

| pair | shared instances | status |
|---|---|---|
| `gpt5mini:qwen80b` | 1498 | running |
| `gpt5mini:converted` | 681 | queued |
| `qwen80b:converted` | 682 | queued |

Serial over pairs **on purpose**: the WAF ceiling counts every concurrent job on this
box, and the SWE-Gym rollouts already hold 10 workers on a hard deadline. This adds 2
concurrent callers, not 6. Re-entrant via stamp files in `data/.stamps`, so a crash or
a deliberate stop resumes the queue.

Settings: 3 iterations, 60 samples/student, 20 judge pairs, agent+judge both
`claude-sonnet-4-5` over the NVIDIA gateway. No extra API key was needed — the gateway
serves Claude. Budget is not a constraint ($190 spent of $15,000).

## Guards running

| tmux session | what it does |
|---|---|
| `swegym-diskguard` | sweeps images below 450GB free; first-seen ledger with 1h grace |
| `swegym-rebalance` | keeps the pool at budget; hands converged models' workers to the runs still going |

Never `docker system prune -a` on this box — it also runs other users' swesmith /
susvibes / opsd-eval / cvat / airflow containers. Filter by our own name prefixes only.
