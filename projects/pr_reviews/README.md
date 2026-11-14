# pr_reviews: fine-tuning a code reviewer on real review findings

The first project built on couch's fine-tuning layer (`finetune/`). Goal: take a 27B open model and make it better at finding bugs in pull requests, using an internal export of code reviews as the training signal. Paused 2026-11-08 with a clear picture of what the data can and cannot teach. Everything here reproduces from the commands below given the data.

## Data

Work data, not in this repository and never to be published. The code holds paths only.

- An export of pull requests from an internal organisation, one directory per PR with the PR metadata and the diff of each reviewed commit, under `~/training-data/code/pr-reviews`.
- A golden set of review findings with human or rule verdicts (`valid`, `invalid`, `unclear`) under `~/code/reviewdo/bench/golden`.
- Clones of the repositories under `~/code/remundo`, read at the reviewed commit with `git show`.
- Everything derived (jsonl splits, adapters, generated answers, judge verdicts, reader packets and verdicts) under `~/training-data/code/pr-reviews-*`.

Languages: C#, TypeScript, Svelte. Functions are located with tree-sitter; findings on files without a grammar (yaml, markdown, csproj, razor) are dropped.

## Tasks and files

| file | what it does |
|---|---|
| `dataset.py` | joins findings to the export, locates the enclosing function at the reviewed commit, renders it, splits by PR and time. Also builds the verifier examples (function + finding → `valid` / `invalid`). |
| `generator.py` | the main task: function → the reviewer's finding, or `no break found`. Real findings from the golden set; clean functions sampled from files the reviewer raised nothing on. |
| `generate_eval.py` | scores a base or adapter on val: loss, probability of the refusal token at the first answer position, AUC between real and clean, refusal rates, and greedy answers. `--match` picks the adapter threshold that answers as many real functions as the base did so hit counts compare at equal volume. `--mode verdict` instead reads the decision at the token after `Verdict:` in a reason-first answer (`finetune/verdict.py`). |
| `judge.py` | asks the local 27B through Ollama whether a generated finding matches the reviewer's. |
| `verifier_eval.py` | per-class accuracy and AUC for the verifier task. |
| `contained.py` | labels each finding as verifiable from the function alone or not, and builds the contained-only subset. |
| `context.py` | adds the file's diff hunks from the reviewed commit to the prompt, rebuilds the split, relabels containment, and writes the contained-with-diff subset (`subset`). |
| `headroom.py` | samples the base several times per val function and reports any-of-k hits. |
| `packets.py` | writes one markdown packet per function (prompt, gold, each model's finding, clone path and commit) for a human or LLM reader with repository access. |
| `verdicts.py` | merges the readers' verdict files (latest line per function and subject wins) and prints the per-model and per-label counts. |
| `verified.py` | applies the readers' labels: drops invalid findings and clean functions with a defect, tags each kept finding `contained`, writes the label sidecars and the defects-on-clean lists. |
| `traces.py` | the teacher-trace run: writes one packet per verified function for a stronger model to write a reasoning trace, bundles the traces into check assignments, and assembles the trace-format train and val splits from traces that passed the check. |

Run everything from the repository root as modules, e.g. `uv run python -m projects.pr_reviews.generator`.

## Recipe

QLoRA on `Qwen/Qwen3.8-27B`: NF4 frozen base via bitsandbytes, LoRA rank 16 alpha 32 on every attention and MLP projection (400 modules, 109M trainable, 0.4%), gradient checkpointing, token-budget batches of 3,072, accumulation 4, one epoch, lr 1e-4 cosine to 1e-5, AdamW without weight decay, loss summed over labelled tokens and divided by the labelled tokens in the optimiser step. About 23 GB peak and 440 to 500 tokens/s on one 32 GB card. The decision to speak is read as the probability of the first refusal token; generation is greedy with that token suppressed.

```
uv run python -m projects.pr_reviews.generator
uv run python -m projects.pr_reviews.context            # diff-context prompts and containment labels
uv run python -m projects.pr_reviews.context subset     # contained-with-diff train/val
uv run python -m projects.pr_reviews.generate_eval --data <dir> --model Qwen/Qwen3.8-27B --bits 4
uv run python -m finetune.sft --data <dir> --output-dir <runs> --model Qwen/Qwen3.8-27B --bits 4 --checkpoint --epochs 1 --eval-interval 12
uv run python -m projects.pr_reviews.generate_eval --data <dir> --run-dir <runs>/<ts> --model Qwen/Qwen3.8-27B --bits 4 --match
uv run python -m projects.pr_reviews.judge <runs>/<ts>/answers-adapter.jsonl
uv run python -m projects.pr_reviews.packets --answers base=<answers> adapter=<answers>   # then readers append verdicts-*.jsonl
uv run python -m projects.pr_reviews.verdicts
uv run python -m projects.pr_reviews.verified
```

`generate_eval` also takes `--split`, `--kind` and `--max-new-tokens`, and writes answers as it goes, so a long run can be read before it ends.

## Results

Val is the newest 15% of PRs by review date, split by PR. Judge hits are "the generated finding matches the reviewer's finding" as decided by the local 27B; it is a ceiling on both models, since a correct finding the reviewer did not raise scores zero. AUC is the share of real-clean pairs where the real function scored riskier (1.0 perfect, 0.5 chance).

### Verifier, 1.5B (`Qwen2.5-Coder-1.5B-Instruct`), 2026-11-06

Val 127 findings, 13 invalid. The adapter learnt the class prior and nothing that ranks.

| model | accuracy | recall invalid | recall valid | balanced | AUC |
|---|---|---|---|---|---|
| base | 0.850 | 2/13 | 106/114 | 0.542 | 0.322 |
| adapter | 0.898 | 0/13 | 114/114 | 0.500 | 0.463 |

### Generator, function only

Val 110 real, 105 clean. Compared at equal answered counts (91 real answered).

| model | AUC | judge hits | answer length |
|---|---|---|---|
| 1.5B adapter | 0.542 | 3/53 | |
| 27B base | 0.493 | 10/91 | ~800 chars |
| 27B adapter, one epoch | 0.645 | 5/91 | ~340 chars |
| 27B adapter, contained-only subset (27 val) | 0.704 | 0/24 vs base 1/24 | 278 chars |

Every function-only adapter settled on a refusal probability near 0.68 on both classes regardless of the data mix, and wrote shorter, reviewer-style comments that matched the reviewer less often than the untrained base.

### Generator, function plus file diff, contained subset, 2026-11-08

Train 354 real + 354 clean, val 44 + 44. Run `20261108-094029`, val loss 2.543 → 1.922 nats per token.

| | base | adapter |
|---|---|---|
| AUC | 0.558 | 0.858 |
| mean p(refuse) real / clean | 0.011 / 0.040 | 0.615 / 0.845 |
| correct refusals on clean at 0.5 | 1/44 | 41/44 |
| misses on real at 0.5 | 0/44 | 32/44 |
| balanced accuracy, best val threshold | 0.51 | 0.81 (threshold 0.7: 33 answered, 6 clean flagged) |
| judge hits, both answering all 44 | 17 | 13 |
| answer length | 838 chars | 352 chars (gold 322) |

Gate-plus-base (adapter decides, base writes) at threshold 0.7: 33/44 answered, 15 hits, 6/44 clean flagged, against the base's 17 hits with 43/44 clean flagged.

Headroom: the base sampled four times per val function at temperature 0.7 reaches 21/44 (any of four), 23 with greedy included; 21 functions are never matched. Ten of those are one finding pattern repeated across one PR that needs context from another file; four are logic bugs; seven are conventions or numeric checks.

### Validation by a stronger reader, 2026-11-09

The judge above asks "does this match the reviewer". To check what that measured, every saved val answer was re-read by Claude subagents with the repository checked out at the reviewed commit, recording whether each finding names a genuine defect with the correct mechanism. All answers were generated with the refusal token suppressed, so both models were forced to name a break on every function.

| | base, 200-token answers | base, 600-token answers | adapter |
|---|---|---|---|
| genuine defect on real functions (44) | 0 | 1 | 12 |
| genuine defect on clean functions (44) | 0 | 0 | 1 |
| answers cut off before naming a defect | most | 29/88 | 0 |
| Ollama judge hits confirmed | 0 of 17 | | 11 of 13 |

The base writes long reasoning before committing and was cut at 200 tokens in every earlier table; at 600 tokens it still runs out on a third of the functions. The judge's 17 base hits were reasoning that circled the right area. Of the adapter's 12, 9 are one finding pattern repeated across repositories by a templated change, which also appears many times in train; outside that cluster the adapter named 3 genuine defects in 35 functions and the base none.

Label quality from the same pass, which is the first human-grade read of the data:

| | val | train |
|---|---|---|
| clean functions with a genuine defect | 1/44 | 29/354 |
| real findings verifiable from the prompt | 21/44 | 259/354 |
| real findings true but needing other files | 23/44 | 84/354 |
| real findings wrong | 0/44 | 11/354 |
| second defects the reviewer did not raise | | 37 |

The 29 train defects come from two independent reads: an open-ended read of each clean function found 17, and checking the base's forced 600-token finding on each function found 16, with only 4 in common. Two passes with that little overlap mean neither is a complete audit and the true defect rate on "clean" functions is above 8%. The base's forced findings on clean train functions were a genuine defect 16 times in 354, wrong 202 times and cut off 136 times. The Ollama containment labeller had passed all 44 val findings as contained. The corrected set (`pr-reviews-generator-context-verified`, 343 + 325 train and 44 + 43 val, each kept finding tagged `contained`, with label sidecars and the defects-on-clean lists) is what any resumed run should train and evaluate on.

### Teacher traces with a verdict line, 2026-11-09

Trained on reasoning traces written by a stronger model (Claude Fable for real functions, Claude Opus for clean ones) from the student's prompt plus the known answer, each trace checked by a second Opus pass; 29 of 648 traces were rejected and 84 real findings had no trace because they need other files. The answer format changed to a three-to-five sentence trace, a line `Verdict: break` or `Verdict: no break`, then the finding. The decision is read as the probability of `no` against `break` at the token after `Verdict:` (`--mode verdict`), so the model reasons before it decides. Train 250 real + 306 clean, val 44 + 43 (the verified val set, including the 23 real findings that need context). Run `20261109-111239`, 90 steps, val loss 1.410 → 1.002 nats per token. Both models generated with 400 tokens and refusal not suppressed; every `Verdict: break` finding was read by an Opus subagent with the repository.

| | base | trace adapter |
|---|---|---|
| reached a verdict line | 32/87 | 87/87 |
| said `break` on real (44) | 7 | 5 |
| genuine defect among those | 2 | 3 (2 in the templated cluster) |
| said `break` on clean (43) | 5 | 1 |
| genuine defect among those | 0 | 0 |
| AUC on p(no break) at the verdict position | 0.41 | 0.76 |
| answer length | 1,700 chars | 590 chars |

The adapter learnt the format and the prose register of its teacher, and its findings are more often right than wrong (3 of 6 against the base's 2 of 12). But it declines on 39 of 44 real functions, and its verdict probability is 0.0 or 1.0 on nearly every item, so there is no threshold that answers more. Last night's first-token adapter, forced to answer, named 12 genuine defects on the same 44 (3 outside the cluster); this one names 3 (1 outside) and cannot be pushed higher. Two of its three wrong findings are the same claim, that a handler which no longer catches exceptions loses dead-lettering, which the messaging library it hands off to makes false; both the base and the teacher-trained adapter make it.


## What was learnt

1. **The function alone does not carry the signal.** Reviewer findings cite the diff, callers and sibling files. Containment labelling put 23 to 50% of findings as verifiable from the function; with the file's diff, 52 to 62%. Only the diff-context run separated the classes (AUC 0.86 against 0.65 to 0.70 for every function-only run).
2. **Fine-tuning on the reviewer's comments taught when to speak, not what to say.** The decision lives in the first answer token and learnt from the data. The prose imitated the short comment style and matched the reviewer less often than the base's long answers. This matches the literature: supervised fine-tuning on raw review comments degrades code models; what works is filtered teacher traces, rejection sampling, or preference training on judge-ranked samples.
3. **A stronger reviewer's final comments are not teacher data.** They were produced from more context than the student sees and they are conclusions, not reasoning. A teacher transfers when it writes from the student's input with its reasoning shown, or when it acts as the filter on the student's own samples.
4. **The val set is too small and too clustered to show a content win.** 44 items with a 10-item cluster from one PR; a 5-hit difference is noise. Human labelling of the val set and a larger or public test set come before more training.
5. **An LLM judge with a match criterion rewards reasoning that circles the answer.** Seventeen base "hits" were answers cut off mid-reasoning or wrong on the mechanism; none survived a reader with repository access. Generation budgets change judged results for verbose models, so a validator needs a "truncated" verdict and the saved text must reach a conclusion before it is scored.
6. **Templated changes across repositories leak.** One finding pattern raised on the same test in nine repositories supplied most of the adapter's confirmed hits in val. It does not appear in train at all, so the adapter solved it from the prompt, not from memory; it still counts as one item, and clusters like it should be collapsed to one per split before reading any number.
7. **A verdict after the trace is a committed decision, not a score.** With the decision token first, its probability is a soft estimate from the prompt alone and a threshold trades volume for precision (AUC 0.86 on the diff-context run). After a trace, the trace has already argued one way and the verdict token follows it at 0.0 or 1.0, so AUC falls (0.76) and the threshold has nothing to move. Reason-first answers need sampling several traces per function, or a separate gate, to recover a ranking.
8. **Mechanics that mattered on a 32 GB card:** gradient checkpointing for the hybrid linear-attention layers, computing logits only at labelled positions (the 248k vocabulary made the full logits tensor the peak), `expandable_segments` in the allocator, and a per-optimiser-step loss denominator so short clean answers do not outweigh long real ones.

## If resumed

The trace run leaves 20 wrong adapter samples and 250 checked teacher traces that pair directly into DPO data (student's wrong trace rejected, teacher's trace chosen). Otherwise: rejection-sampling fine-tune with the base as its own teacher (ceiling about 21 to 23 of 44 from the headroom measurement), or teacher traces from a stronger model (costed at roughly $25 for four samples over 354 functions with Sonnet or batch pricing). Either way, start from the verified set, collapse the templated cluster to one item per split, and give the base enough tokens to finish before judging it. Serving (merge, GGUF, Ollama) waits for a model that beats the base on content.
