# Recent coding-agent models: base model + before → after performance

Survey of recent (2024–2026) SWE-agent training works: which **base model** each starts from, and the **before → after** resolve rate (base checkpoint zero-shot vs. after training).

All numbers are **SWE-bench Verified, pass@1 single-attempt** unless noted (mostly OpenHands / OpenHands-style scaffold). "Before" = the base checkpoint evaluated zero-shot in the same scaffold. Numbers pulled from each paper's results tables.

## Base / foundation coder models (the checkpoints everyone fine-tunes *from*)

| Model | Sizes | Paper | arxiv |
|---|---|---|---|
| Qwen2.5-Coder | 0.5 / 1.5 / 3 / **7** / 14 / 32B | Qwen2.5-Coder Technical Report | [2409.12186](https://arxiv.org/abs/2409.12186) |
| Qwen2.5 (general) | 0.5–72B | Qwen2.5 Technical Report | [2412.15115](https://arxiv.org/abs/2412.15115) |
| Qwen3-Coder-Next | 80B-A3B (MoE) | Qwen3-Coder-Next Technical Report | [2603.00729](https://arxiv.org/abs/2603.00729) |

## Agent-training works: base model, size, before → after

| System | Base model | Size | Before (base, 0-shot) | After training | Paper | arxiv |
|---|---|---|---|---|---|---|
| SWE-Gym | Qwen2.5-Coder-Instruct | 7B | 1.8% | 10.6% | SWE-Gym | [2412.21139](https://arxiv.org/abs/2412.21139) |
| SWE-Gym | Qwen2.5-Coder-Instruct | 14B | 4.0% | 16.4% | SWE-Gym | [2412.21139](https://arxiv.org/abs/2412.21139) |
| SWE-Gym | Qwen2.5-Coder-Instruct | 32B | 7.0% | 20.6% | SWE-Gym | [2412.21139](https://arxiv.org/abs/2412.21139) |
| R2E-Gym | Qwen2.5-Coder | 7B | 1.8% | 19.0% | R2E-Gym | [2504.07164](https://arxiv.org/abs/2504.07164) |
| R2E-Gym | Qwen2.5-Coder | 14B | 4.0% | 26.8% | R2E-Gym | [2504.07164](https://arxiv.org/abs/2504.07164) |
| R2E-Gym | Qwen2.5-Coder | 32B | 7.0% | 34.4% (51.0% best-of-n w/ verifier) | R2E-Gym | [2504.07164](https://arxiv.org/abs/2504.07164) |
| SWE-Dev | Qwen2.5-Coder-Instruct | 7B | ~1.6% | 23.4% (22.8% @ standard 30 rounds) | SWE-Dev | [2506.07636](https://arxiv.org/abs/2506.07636) |
| SWE-Dev | Qwen2.5-Coder-Instruct | 32B | 6.6% | 36.6% (34.0% @ standard 30 rounds) | SWE-Dev | [2506.07636](https://arxiv.org/abs/2506.07636) |
| Skywork-SWE | Qwen2.5-Coder-32B-Instruct | 32B | 6.4% | 38.0% (47.0% w/ TTS best-of-8) | Skywork-SWE | [2506.19290](https://arxiv.org/abs/2506.19290) |
| SWE-World | Qwen2.5-Coder-32B | 32B | 6.2% | 52.0 / 55.0 / 68.2% (SFT / RL / RL+TTS@8) | SWE-World | [2602.03419](https://arxiv.org/abs/2602.03419) |
| SWE-Master | Qwen2.5-Coder-32B-Instruct | 32B | 6.2% | 57.8 / 61.4% (SFT / SFT+RL) | SWE-Master | [2602.03411](https://arxiv.org/abs/2602.03411) |
| SWE-Master | Qwen3-4B-Instruct-2507 | 4B | not reported | 27.6 / 33.4% (SFT / SFT+RL) | SWE-Master | [2602.03411](https://arxiv.org/abs/2602.03411) |
| SWE-RL | Llama-3.3-70B-Instruct | 70B | not reported\* | 41.0% | SWE-RL | [2502.18449](https://arxiv.org/abs/2502.18449) |
| Lingma SWE-GPT | Qwen2.5-Coder-7B | 7B | not reported | 18.2% (Lite 12.0%) | Lingma SWE-GPT | [2411.00622](https://arxiv.org/abs/2411.00622) |
| Lingma SWE-GPT | Qwen2.5-72B-Instruct | 72B | 25.4% | 30.2% (Lite 18.0% → 22.0%) | Lingma SWE-GPT | [2411.00622](https://arxiv.org/abs/2411.00622) |
| SWE-Fixer | Qwen2.5-7B (retriever) + Qwen2.5-72B (editor) | 7B+72B | not reported | 32.8% (Lite 24.7%) | SWE-Fixer | [2501.05040](https://arxiv.org/abs/2501.05040) |

\* SWE-RL reports only an isolated oracle-file-repair subtask baseline (5.4% greedy / 16.6% majority-vote), not a comparable end-to-end SWE-bench Verified number.

## Takeaways

- **Base coder models are near-useless zero-shot as agents**, and training is the entire story: 7B ≈ 1.8% → 19–23%; 32B ≈ 6–7% → 34–68% on full Verified. So for student models, the *fine-tuning* matters far more than the base's out-of-box score.
- **Qwen2.5-Coder-7B is the near-universal small SWE-agent base.** The only reported **4B** base anyone trains is **Qwen3-4B-Instruct-2507** (SWE-Master: → 33.4%). Both are HF-only (not on the NVIDIA gateway) — fine for fine-tuning, just not for a zero-shot gateway probe.
- **Cleanest same-scaffold before→after pairs to anchor on:** SWE-Gym, R2E-Gym, SWE-Dev, Skywork-SWE, SWE-World, SWE-Master.
- **Do not compare these to our qwen3.5-9b base = 76%** — that was on **easy50** (the 50 easiest instances); every number above is **full SWE-bench Verified (500)**. Easy50 runs far higher, so the comparison is apples-to-oranges.

## Gateway reachability (our NVIDIA API, verified 2026-07)

The literature bases (Qwen2.5-Coder-7B, Qwen3-4B-Instruct-2507) are **not** on our gateway. Small models our API *can* reach (live-tested): `phi-4-mini-instruct` (~3.8B), `llama-3.1-8b-instruct` (8B), `nemotron-nano-9b-v2` (9B), `gemma-2-9b-it` (9B), `mistral-7b-instruct-v0.3` (7B), `qwen3.5-9b` (done, easy50 76%), plus small-active MoEs `gpt-oss-20b` (A3.6B), `qwen3.5-35b-a3b` (A3B). Reachability only matters for zero-shot gateway probes; fine-tuning pulls weights from HF and is not gateway-gated.
