# pr_reviews: fine-tuning a code reviewer on real review findings

The first project built on couch's fine-tuning layer (`finetune/`). Goal: take a 27B open model and make it better at finding bugs in pull requests, using an internal export of code reviews as the training signal. Paused 2026-11-08 with a clear picture of what the data can and cannot teach. Everything here reproduces from the commands below given the data.

## Data

Work data, not in this repository and never to be published. The code holds paths only.

- An export of pull requests from an internal organisation, one directory per PR with the PR metadata and the diff of each reviewed commit, under `~/training-data/code/pr-reviews`.
- A golden set of review findings with human or rule verdicts (`valid`, `invalid`, `unclear`) under `~/code/reviewdo/bench/golden`.
- Clones of the repositories under `~/code/remundo`, read at the reviewed commit with `git show`.
- Everything derived (jsonl splits, adapters, generated answers, judge verdicts) under `~/training-data/code/pr-reviews-*`.

Languages: C#, TypeScript, Svelte. Functions are located with tree-sitter; findings on files without a grammar (yaml, markdown, csproj, razor) are dropped.

## Tasks and files

| file | what it does |
|---|---|
| `dataset.py` | joins findings to the export, locates the enclosing function at the reviewed commit, renders it, splits by PR and time. Also builds the verifier examples (function + finding → `valid` / `invalid`). |
| `generator.py` | the main task: function → the reviewer's finding, or `no break found`. Real findings from the golden set; clean functions sampled from files the reviewer raised nothing on. |
| `generate_eval.py` | scores a base or adapter on val: loss, probability of the refusal token at the first answer position, AUC between real and clean, refusal rates, and greedy answers. `--match` picks the adapter threshold that answers as many real functions as the base did so hit counts compare at equal volume. |
| `judge.py` | asks the local 27B through Ollama whether a generated finding matches the reviewer's. |
| `verifier_eval.py` | per-class accuracy and AUC for the verifier task. |
| `contained.py` | labels each finding as verifiable from the function alone or not, and builds the contained-only subset. |
| `context.py` | adds the file's diff hunks from the reviewed commit to the prompt, rebuilds the split, relabels containment, and writes the contained-with-diff subset (`subset`). |
| `headroom.py` | samples the base several times per val function and reports any-of-k hits. |

Run everything from the repository root as modules, e.g. `uv run python -m projects.pr_reviews.generator`.

## Recipe

QLoRA on `Qwen/Qwen3.8-27B`: NF4 frozen base via bitsandbytes, LoRA rank 16 alpha 32 on every attention and MLP projection (400 modules, 109M trainable, 0.4%), gradient checkpointing, token-budget batches of 3,072, accumulation 4, one epoch, lr 1e-4 cosine to 1e-5, AdamW without weight decay, loss summed over labelled tokens and divided by the labelled tokens in the optimiser step. About 23 GB peak and 440 to 500 tokens/s on one 32 GB card. The decision to speak is read as the probability of the first refusal token; generation is greedy with that token suppressed.

```
uv run python -m projects.pr_reviews.generator
uv run python -m projects.pr_reviews.context            # diff-context prompts and containment labels
uv run python -m projects.pr_reviews.context subset     # contained-with-diff train/val
uv run python -m projects.pr_reviews.generate_eval --data <dir> --model Qwen/Qwen3.8-27B --bits 4
uv run python -m finetune.sft --data <dir> --model Qwen/Qwen3.8-27B --bits 4 --checkpoint --epochs 1 --eval-interval 12
uv run python -m projects.pr_reviews.generate_eval --data <dir> --run <ts> --model Qwen/Qwen3.8-27B --bits 4 --match
uv run python -m projects.pr_reviews.judge <dir>/runs/<ts>/answers-adapter.jsonl
```

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

## What was learnt

1. **The function alone does not carry the signal.** Reviewer findings cite the diff, callers and sibling files. Containment labelling put 23 to 50% of findings as verifiable from the function; with the file's diff, 52 to 62%. Only the diff-context run separated the classes (AUC 0.86 against 0.65 to 0.70 for every function-only run).
2. **Fine-tuning on the reviewer's comments taught when to speak, not what to say.** The decision lives in the first answer token and learnt from the data. The prose imitated the short comment style and matched the reviewer less often than the base's long answers. This matches the literature: supervised fine-tuning on raw review comments degrades code models; what works is filtered teacher traces, rejection sampling, or preference training on judge-ranked samples.
3. **A stronger reviewer's final comments are not teacher data.** They were produced from more context than the student sees and they are conclusions, not reasoning. A teacher transfers when it writes from the student's input with its reasoning shown, or when it acts as the filter on the student's own samples.
4. **The val set is too small and too clustered to show a content win.** 44 items with a 10-item cluster from one PR; a 5-hit difference is noise. Human labelling of the val set and a larger or public test set come before more training.
5. **Mechanics that mattered on a 32 GB card:** gradient checkpointing for the hybrid linear-attention layers, computing logits only at labelled positions (the 248k vocabulary made the full logits tensor the peak), `expandable_segments` in the allocator, and a per-optimiser-step loss denominator so short clean answers do not outweigh long real ones.

## If resumed

Rejection-sampling fine-tune with the base as its own teacher (ceiling about 21 to 23 of 44 from the headroom measurement), or teacher traces from a stronger model (costed at roughly $25 for four samples over 354 functions with Sonnet or batch pricing). Either way, first fix the val set: collapse the duplicate cluster, drop findings not verifiable from the prompt, tag conventions, and widen the split. Serving (merge, GGUF, Ollama) waits for a model that beats the base on content.
