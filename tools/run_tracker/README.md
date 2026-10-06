# run_tracker

Status snapshots for the long-running benchmark jobs on this box. Rollouts run for
days, so the useful question is rarely "did it start" but "is it still making
progress, and is anything shared running out".

## Usage

```bash
bash tools/run_tracker/status.sh          # print one snapshot

tmux new -s run-tracker "bash tools/run_tracker/track.sh"
tail -40 eval_outputs/run_tracker.log     # read the latest
```

`track.sh` appends a snapshot every `INTERVAL` seconds (default 1800) and trims the
log once it passes `MAX_LOG_MB` (default 20). It must live in tmux — a background
shell tied to an agent session gets SIGTERMed when that session ends.

## What a snapshot shows

```
RUN                                MODEL        OK    ERR  TOTAL     IDLE  STATE
kimi-k26-azure-r2egym-1502         w=4          52    252   1502      11m  ok
gpt5-mini-azure-r2egym-1502        w=4         243    178   1502       0m  ok

judge verdicts:
  func_localize_claude45_1457i                  63 judged     0 errored  (idle 0m)

resources:
  disk /home 2.5T free, /mnt/data 115G free
  docker: 42 containers, 14 agent-server images
  gateway WAF: 0/5 probes rate-limited
```

Rollouts are discovered from the environment of live `run_supervisor.sh` processes,
so a newly launched run appears without editing anything — and a run that has exited
disappears, which is itself the signal that it finished or died.

Three things in here are worth understanding, because each corresponds to a failure
that has actually cost time on this box:

- **OK/ERR count distinct instance ids, not lines.** `output_errors.jsonl` appends one
  line per failed attempt and a resumed run re-attempts errored instances, so `wc -l`
  once reported a run as ~300 instances further along than it was.
- **IDLE is the age of `output.jsonl`.** Every stall so far — upstream 529s, WAF 429s,
  a crashed child — showed up as a frozen mtime well before the supervisor's own
  counters noticed. An hours-old mtime on a live run is flagged `STALLED?`.
- **The gateway WAF limit is per source IP** (`x-amzn-waf-rule: CPE_RateLimit_IP`),
  shared by every job here. It throttles on *total* concurrency across all rollouts,
  not per model or per key, so the fix for a throttled run is usually to reduce the
  box's aggregate worker count rather than to switch models. It is bursty, so the
  probe samples five times instead of trusting one response.
