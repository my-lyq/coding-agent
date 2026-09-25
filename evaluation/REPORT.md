# Coding Agent Evaluation Report

## Setup

- Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- LoRA adapter: /home/lyq/coding-agent-training/outputs/qwen2.5-coder-agent-lora
- Tasks: 3 (one seen during SFT, two held out)
- Decoding: greedy (temperature 0), fixed seed, isolated workspace per run
- Repair success: non-empty source patch and evaluator-owned final tests pass
- Test pass: final test suite exits successfully
- Agent steps: model decision turns; evaluator test is excluded
- Token usage: tokenizer input tokens plus generated tokens

## Aggregate results

| Condition | Repair success | Held-out repair | Test pass | Avg steps | Total tokens | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|
| base | 100.0% | 100.0% | 100.0% | 1.00 | 497 | 165.7 |
| agent | 33.3% | 50.0% | 33.3% | 5.33 | 11091 | 3697.0 |
| agent_sft | 33.3% | 50.0% | 33.3% | 6.00 | 10687 | 3562.3 |

## Per-task results

| Condition | Task | SFT seen | Repaired | Tests | Steps | Tokens |
|---|---|---:|---:|---:|---:|---:|
| base | calculator_subtract | True | True | True | 1 | 168 |
| base | normalize_palindrome | False | True | True | 1 | 159 |
| base | stable_unique | False | True | True | 1 | 170 |
| agent | calculator_subtract | True | False | False | 6 | 3760 |
| agent | normalize_palindrome | False | True | True | 4 | 1588 |
| agent | stable_unique | False | False | False | 6 | 5743 |
| agent_sft | calculator_subtract | True | False | False | 6 | 4062 |
| agent_sft | normalize_palindrome | False | True | True | 6 | 2887 |
| agent_sft | stable_unique | False | False | False | 6 | 3738 |

## Findings

1. **Direct generation is strongest on these toy tasks.** Base solves all three because the complete source is in context and each repair is local. The Agent protocol adds no information advantage here.
2. **Tool-use reliability is the bottleneck.** Agent failures are dominated by unsupported actions, placeholder writes, and repeated reads/tests rather than inability to describe the code fix.
3. **One-trajectory SFT does not improve success.** It slightly reduces total Agent tokens (11,091 to 10,687) but does not improve repair rate. The adapter also fails on its training-seen task, showing that low teacher-forced loss does not guarantee stable autoregressive rollout.
4. **The SFT data is too narrow.** Its only example contains a fixed calculator path and one successful action order. More diverse paths, failures, recovery steps, and step-level next-action samples are needed.

## Threats to validity

- Three synthetic tasks are insufficient for statistical conclusions.
- One task is present in SFT; it is marked explicitly and excluded from the held-out column.
- Repair success and test pass coincide here because all accepted patches changed source and passed the entire tiny suite.
- Baseline sees complete source while Agent must acquire it through tools; this intentionally measures the cost and reliability of agentization, not context retrieval ability.

## Next experiments

- Generate 50-100 diverse trajectories and split by repository before SFT.
- Add invalid-action and failed-test recovery demonstrations.
- Evaluate on multi-file tasks where iterative search and testing can outperform one-shot generation.
- Report confidence intervals and pass@k across multiple decoding seeds.
