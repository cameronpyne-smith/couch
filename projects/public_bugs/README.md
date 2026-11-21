# public_bugs: teacher traces on open-source bug fixes

The second project on couch's fine-tuning layer, started 2026-11-09 after `pr_reviews` showed that teacher traces transfer but that the work data could not be published and had too few real examples. Everything here is built from permissively licensed open-source repositories, so the dataset and any adapter trained on it can be released.

## Idea

A merged pull request that fixes a bug tells you where a defect was. Blaming the lines it changed, at the fix's parent, names the commit that introduced them. That introducing commit is the review moment: the function as it was then, plus that commit's diff of the file, is what a reviewer saw. The fix diff and its issue are the ground truth a teacher reasons towards.

- **real**: a function at its introducing commit, labelled with the later fix. The teacher writes a trace that reaches the defect from the shown code, or marks it unreachable when the defect needs code the prompt does not show.
- **clean**: a function changed by an ordinary merged pull request, in a file no later fix touched. A stronger negative than "the reviewer said nothing", though still not a proof.

The answer format is the one from `pr_reviews/traces.py`: a three to five sentence trace, a line `Verdict: break` or `Verdict: no break`, then the finding. The decision is read at the verdict position, averaged over sampled traces (`generate_eval --mode verdict --samples 4`).

## Data

Under `~/training-data/code/public-bugs/`, not in this repository because of size, not secrecy.

- `repos.jsonl`, `fixes.jsonl`, `merges.jsonl`: the mined pull requests (a Claude subagent working from `MINING.md`, using the GitHub CLI and full clones under `clones/`).
- `dataset/train.jsonl`, `dataset/val.jsonl`: the built examples. Val is three whole repositories (zod, Humanizer, Quartz.NET) so nothing templated can leak across the split.
- `dataset/packets/`, `traces-*.jsonl`, `checks-*.jsonl`: teacher packets, written traces and the second-pass checks, same protocol as the `pr_reviews` trace run (`BRIEF.md`, `BRIEF-CHECK.md`).
- `dataset/train-traces.jsonl`, `val-full.jsonl`: assembled traces and the full val with gold; `dataset/sft/`: the training split with a small holdout for the loss curve.

Repositories: C# Polly, FluentValidation (dropped, too few fixes), Humanizer, Npgsql, refit, RestSharp, Serilog (dropped), Spectre.Console, StackExchange.Redis, Ocelot, NLog, NUnit, Quartz.NET, YamlDotNet, MassTransit; TypeScript zod, date-fns, drizzle-orm, hono, NestJS, valibot, zustand, TanStack Query, tRPC, Vitest, Vue core. Licences MIT, Apache-2.0, BSD-3-Clause, PostgreSQL. Each row carries its repository, commits and licence.

## Files

| file | what it does |
|---|---|
| `dataset.py` | from the mined fixes: blame to the introducing commit, locate the enclosing function there with tree-sitter, render the prompt with that commit's diff; from the merges: clean functions in files no later fix touched; split by repository with a per-repository cap. `stats` summarises the mined rows. |
| `traces.py` | packets for the teacher, writer assignments, check bundles, assembly (drops unreachable, disagree, rejected and generated or docs-site paths), and the `sft` split at a chosen real share. |
| `pairs.py` | chosen and rejected answer pairs for `finetune.dpo` from sampled answers: by verdict (optionally balanced by kind) or by reader verdicts on which answers name the fixed defect, with a holdout by function. |

```
uv run python -m projects.public_bugs.dataset build --val-repos colinhacks/zod Humanizr/Humanizer quartznet/quartznet
uv run python -m projects.public_bugs.traces packets && uv run python -m projects.public_bugs.traces assign --size 75
# writers append traces-*.jsonl
uv run python -m projects.public_bugs.traces checks --size 75
# checkers append checks-*.jsonl
uv run python -m projects.public_bugs.traces assemble && uv run python -m projects.public_bugs.traces sft
uv run python -m finetune.sft --data <dataset>/sft --output-dir <runs>/sft --model Qwen/Qwen3.8-27B --bits 4 --checkpoint --epochs 1 --eval-interval 12
uv run python -m projects.pr_reviews.generate_eval --data <dataset> --run-dir <runs>/sft/<ts> --model Qwen/Qwen3.8-27B --bits 4 --split val-full --mode verdict --max-new-tokens 400 --samples 4
```

## Yield

| stage | count |
|---|---|
| fix pull requests mined (20 repositories kept by the miner) | 3,740 |
| with an introducing commit from blame | 3,211 |
| introducing commit owns at least half the fixed lines and touched at most 10 files | 1,569 |
| built as real examples (per-repository cap 60; the rest lost to no function at the blamed line, or a prompt over 16k chars) | 566 |
| train real packets written by Opus teachers | 501 |
| marked unreachable by the writer | 207 (41%) |
| rejected by the checker among the rest | 6 |
| real traces kept for training | 286 |
| clean traces kept (150 written, 4 disagreed, 6 rejected) | 141 |

The unreachable share is the honest cost of the method: in two cases out of five the fixing commit landed in a different function from the one blame pointed at, or in a much later version of the code. Checkers and writers agreed closely, 73 of the 78 rejections in the first wave were traces the writer had already flagged.

## Results, 2026-11-09

Run `sft/20261109-152620`: 259 real + 125 clean traces, one epoch, 55 optimiser steps, 1,037 s at 485 tokens/s, peak 25.1 GB, held-out loss 1.300 → 0.905 nats per token. Both models generated 400 tokens with refusal allowed; the adapter's decision score is p(no break) at the verdict position averaged over the greedy answer and four samples at temperature 0.7; the base's is from its greedy answer alone. Every `Verdict: break` finding was read by an Opus subagent with the clone, judged `real` only when the defect exists with the mechanism stated, whether or not it is the one the later fix corrected.

### Public val: 61 real, 39 clean, three held-out repositories

| | base | adapter |
|---|---|---|
| reached a verdict line within 400 tokens | 56 | 100 |
| AUC on the decision score | 0.61 | 0.76 |
| said `break` on real functions | 15 | 40 |
| genuine defect among those | 3 | 4 |
| of which the defect the fix corrected | 1 | 1 |
| said `break` on clean functions | 10 | 12 |
| genuine defect among those | 1 | 1 |
| balanced accuracy at threshold 0.5 | 0.71 | 0.67 |
| answer length | 1,600 chars | 590 chars |

### Transfer to the work set: 44 real, 43 clean (from `pr_reviews`)

| | base | this adapter | work-trace adapter (9f.11) | forced first-token adapter (9f.8) |
|---|---|---|---|---|
| AUC on the decision score | 0.41 | 0.60 | 0.76 | 0.86 |
| said `break` on real | 7 | 19 | 5 | 44 |
| genuine defect among those | 2 | 3 (one templated test, three repositories) | 3 | 12 |
| said `break` on clean | 5 | 18 | 1 | 44 |
| genuine defect among those | 0 | 0 | 0 | 1 |

### Reading

The real-heavy mix fixed the declining: on its own domain the adapter answers two real functions in three and reaches a verdict every time, and its decision score ranks real above clean with AUC 0.76 against the base's 0.61. What it writes is mostly wrong. Of 52 findings on the public val, 5 name a genuine defect and 1 names the defect the project later fixed; the base, when it manages to conclude, is right about as often (4 of 25). Two of the nine genuine findings across both models are on functions labelled clean, both pre-existing edge cases the project fixed later, so the "no later fix" negative is weak in the same way the silent-reviewer negative was. On the work set the adapter ranks better than the base but worse than either work-trained adapter, and its 37 findings there contain 3 genuine, all the same templated test.

So 286 verified teacher traces teach the decision and the voice, and do not teach the model to find the bug a maintainer later found: on both val sets the 27B student matches the human-found defect on about 2% of functions, trained or not. The content ceiling measured in `pr_reviews` (21 to 23 of 44 reachable by sampling the base) is the number to move, and supervised traces at this scale do not move it. What is left to try with no further teacher spend is preference training on the student's own samples: on the public val the adapter's five sampled traces per function already disagree with each other on 59 of 100 functions, and the fix diff decides which side is right.

### Combined with the work traces, 2026-11-09

One more run on both trace sets together (`~/training-data/code/combined-traces/`, 509 real + 274 clean at the same 65% mix, 121 steps, held-out loss 1.30 → 0.972), and the work-trace adapter scored on the public val to complete the matrix. Decision-score AUC with `--samples 4`; the work-trace adapter's work-val number is its greedy run.

| adapter trained on | work val | public val |
|---|---|---|
| nothing (base) | 0.41 | 0.61 |
| public traces | 0.60 | 0.76 |
| work traces | 0.76 | 0.66 |
| both | 0.85 | 0.73 |

Each single-source adapter loses 0.1 to 0.15 AUC away from home. The mixture keeps most of both and is the best trace-trained decision on the work set, level with the forced first-token adapter. Three Opus readers then verified every `break` finding of the two new cells and of the combined adapter on the work set (`verdicts-combined-on-*.jsonl`, `verdicts-work-adapter-on-public.jsonl`), same protocol as above.

| public val, 61 real + 39 clean | base | public traces | work traces | both |
|---|---|---|---|---|
| said `break` on real | 15 | 40 | 24 | 32 |
| genuine defect among those | 3 | 4 | 3 | 5 |
| of which the defect the fix corrected | 1 | 1 | 3 | 4 |
| said `break` on clean | 10 | 12 | 7 | 8 |
| genuine defect among those | 1 | 1 | 0 | 0 |

| work val, 44 real + 43 clean | base | public traces | work traces (greedy) | both | forced first-token (9f.8) |
|---|---|---|---|---|---|
| said `break` on real | 7 | 19 | 5 | 24 | 44 |
| genuine defect among those | 2 | 3 | 3 | 8 | 12 |
| said `break` on clean | 5 | 18 | 1 | 8 | 44 |
| genuine defect among those | 0 | 0 | 0 | 0 | 1 |

The content moved for the first time. On the public val the combined adapter names the maintainer-fixed defect on 4 of 61 real functions (the Quartz `InterruptAsync` `break`, zod's `floatSafeRemainder` tolerance and `__proto__` skip, Npgsql's `CreateCommand` override) against 1 for the base and the public adapter, and the work-trace adapter, which never saw a public function, matches 3. On the work set the combined adapter finds 8 genuine defects, seven the `MessageBus_Link` test cluster and one the Teams alert card naming the wrong broker, with none of its 8 clean findings genuine. Two readers split on one finding (zod `floatSafeRemainder`, whose stated example does not fail although its mechanism is right): counted genuine for the combined adapter, not for the work-trace adapter, so the two public rows are within one of each other either way. The rates are still small: 7% of real functions on the public val and 18% on the work set, the latter mostly one templated test. The combined adapter is trained on work data and stays unpublishable.

### Three times the traces, 2026-11-10

The dataset was extended twice more in one morning (per-repository cap 300, then fifteen new repositories mined by two Opus subagents: efcore, aspnetcore, Orleans, akka.net, the MongoDB driver, Marten, Avalonia, maui, Swashbuckle, opentelemetry-dotnet, vite, typeorm, typescript-eslint, solid, and part of prisma and playwright). Writers marked 893 of 1,761 real packets unreachable (51%; 62% in the new repositories, which are UI and platform heavy), and checkers now skip rows that would be dropped anyway. Kept: 850 real + 870 clean. Public-only run `sft/20261110-130024`: 759 real + 418 clean, 166 steps, held-out loss 0.993 → 0.877. Combined run `combined-traces/20261110-144225`: 1,009 real + 543 clean, 229 steps, 6,076 s, held-out loss 1.339 → 0.921. Each adapter was evaluated only where it is meant to serve, and one Opus reader per set judged every `break` finding (`verdicts-public-r2-on-public.jsonl`, `verdicts-combined-r2-on-work.jsonl`).

| public val, 61 real + 39 clean | base | 286 real traces | 759 real traces |
|---|---|---|---|
| AUC on the decision score | 0.61 | 0.76 | 0.78 |
| said `break` on real | 15 | 40 | 41 |
| genuine defect among those | 3 | 4 | 5 |
| of which the defect the fix corrected | 1 | 1 | 4 |
| said `break` on clean | 10 | 12 | 11 |
| genuine defect among those | 1 | 1 | 0 |

| work val, 44 real + 43 clean | combined, 509 real | combined, 1,009 real |
|---|---|---|
| AUC on the decision score | 0.85 | 0.82 |
| said `break` on real | 24 | 28 |
| genuine defect among those | 8 | 3 (8 by last night's reader's standard) |
| said `break` on clean | 8 | 8 |
| genuine defect among those | 0 | 0 |

The decision did not move: 0.02 AUC at home, within noise on the work set. The content did, on the public val: the public-only adapter now names the maintainer-fixed defect on 4 of 61 real functions (Quartz `InterruptAsync`, zod's lazy getter and `__proto__` skip, Humanizer's `PrependArticleSuffix`) against 1 with a third of the data, level with what the combined adapter reached yesterday, and none of its 11 clean findings is a genuine defect. On the work set the picture is unchanged: the same `MessageBus_Link` test cluster and the SignalR start timeout. Today's reader applied the wrong-mechanism rule strictly to five findings that name that test but blame its sibling's cleanup, which last night's reader had counted; with the same standard the count is 8 of 28, as yesterday. The rest of the work findings are one template, "no catch in `HandleAsync`", repeated across nine handlers and refuted by the consumer host that catches and dead-letters. The public-only adapter is the one that can be published, and this is its first run with a content rate above the base.

### Direct preference optimisation, 2026-11-14

`finetune/dpo.py` trains an adapter on pairs of answers to the same prompt, one chosen and one rejected: the loss is `-log sigmoid(beta * margin)`, where the margin is how much more the adapter favours the chosen answer over the rejected one than a frozen copy of its starting point did. It starts at log 2 = 0.693 nats per pair. Every run below starts from the public-only adapter `sft/20261110-130024`, with beta 0.1, learning rate 2e-5, eight pairs per optimiser step and pairs that fit 6,144 tokens.

The pairs come from that adapter's own samples: four sampled answers at temperature 0.7 for each of the 1,177 training functions (`generate_eval --split train --samples 4`). Three ways of pairing them were tried.

```
uv run python -m projects.public_bugs.pairs --answers <samples> --data <dataset>/sft/train.jsonl --out <dataset>/dpo/train.jsonl [--balance]
uv run python -m projects.public_bugs.pairs --answers <samples> --data <dataset>/sft/train.jsonl --verdicts <verdicts-*.jsonl> --out <dataset>/dpo-content/train.jsonl --val-out <dataset>/dpo-content/val.jsonl
uv run python -m finetune.dpo --data <dataset>/dpo --output-dir <runs>/dpo --adapter <runs>/sft/<ts> --model Qwen/Qwen3.8-27B --bits 4 --checkpoint
```

- **By verdict**: chosen is a sample with the right verdict for the function's kind, rejected one with the wrong verdict. 513 functions had disagreeing samples, giving 677 pairs, 477 on real functions and 200 on clean.
- **By verdict, balanced**: the same pairs cut to 200 real and 200 clean, with 16 + 16 held out.
- **By content**: both answers say `break` on a real function; chosen names the defect the later fix corrected and rejected does not. Nine Opus readers, working only from packets with the fix diff, judged all 2,133 `break` samples of the 635 real functions that had at least two: 438 matched, 232 functions had a match and 190 had both a match and a non-match, giving 208 training pairs and 20 held out by function. On the held-out repositories the same adapter matches on about one function in ten, so most of these matches are recall of the trace it was trained on.

| | pairs | epochs, steps | pair loss, train | pair loss, held out | saved at step |
|---|---|---|---|---|---|
| by verdict | 664 | 1, 83 | 0.67 → 0.60, noisy | 0.693 → 0.517 | 83 |
| by verdict, balanced | 393 | 1, 50 | 0.67 to 0.75 throughout | 0.693 → 0.643 → 0.714 | 10 |
| by content | 207 | 3, 78 | 0.68 → 0.09 | 0.693 → 0.630 → 0.637 | 20 |

| public val, 61 real + 39 clean | SFT | SFT repeat | by verdict | balanced | by content |
|---|---|---|---|---|---|
| AUC on the decision score | 0.78 | 0.81 | 0.80 | 0.78 | 0.72 |
| said `break` on real | 41 | 43 | 57 | 45 | 17 |
| said `break` on clean | 11 | 12 | 20 | 13 | 3 |
| balanced accuracy | 0.70 | 0.70 | 0.71 | 0.70 | 0.60 |
| named the defect the fix corrected | 6 | | | | 3 |

The SFT repeat is the same recipe run again (`sft/20261113-173045`), so 0.03 AUC is what chance alone moves. The last row is from one Opus reader judging both adapters' greedy `break` answers against the fix diff without knowing which adapter wrote which; it reads from packets only, so its 6 is not comparable with the 4 in the table above, which came from readers with the clones.

None of the three improved the ranking of real above clean. Each moved the adapter's readiness to say `break` in whichever direction its pairs made cheapest: up when 70% of the chosen answers said `break`, nowhere when the two kinds were balanced (the training loss stayed around its starting 0.69), and down when every rejected answer was a `break` answer with the wrong mechanism. The content run found half as many of the fixed defects with the same share of its findings right (3 of 17 against 6 of 41), so it bought no precision with the recall it gave up, and it memorised its 207 pairs from the second epoch while the held-out loss stood still.

The loss and the loop behave as they should: every run did what its pairs asked. What the pairs lacked is information. Two samples from the same adapter on the same function differ in their conclusion or their guess at a mechanism, not in anything that says why one is right, and a few hundred such pairs are fitted fastest by a shift in the threshold. This is the result the `pr_reviews` literature pass reported for LoRA on this task. Preference pairs for it would need a rejected answer that is wrong for a reason visible in the prompt, at a scale well beyond what one adapter's disagreements on a thousand functions supply.

### Linear probe and layer halves, 2026-11-20

A linear probe asks whether a frozen model's internal vectors already separate buggy from clean, whatever its written answer says. `finetune/activations.py` saves the vector at the last prompt token after every block (65 points, 5,120 numbers each) for the base model and for an adapter; `finetune/probe.py` trains one `Linear(5120, 1)` per layer on those vectors and scores it by AUC, the share of buggy and clean pairs it ranks correctly.

```
uv run python -m finetune.activations --data <dataset>/sft --output-dir <runs>/activations --adapter <runs>/sft/<ts> --model Qwen/Qwen3.8-27B --bits 4
uv run python -m finetune.sft --data <dataset>/sft --output-dir <runs>/sft --model Qwen/Qwen3.8-27B --bits 4 --checkpoint --epochs 1 --eval-interval 12 --layers 32 64
uv run python -m finetune.probe
```

A probe finds the cheapest thing that predicts the label, and here two cheap things do. The bug rate differs by repository, so repository alone scores 0.91 on the held-out rows; and real functions are about 45% longer, so "the longer one is buggy" scores 0.72 inside a repository. The controls, in the order they were added:

- **Same-repository pairs only**, so repository identity cannot win a pair.
- **Repositories held out**: five folds over the 34 training repositories, each probe scored on repositories it never saw (7,172 pairs).
- **Repository and length erased** from the vectors: each repository's mean and the straight-line trend on log length are subtracted, fitted on the training folds and applied to the held-out fold.
- **Shuffled labels**: the same probe trained on labels shuffled inside each repository, which gives the range that noise alone produces.
- **Columns standardised and weights decayed**: column spreads differ up to 75 times within a layer and 300 times between layers, and 5,120 weights memorise 1,177 rows (training AUC 1.00 without decay). The decoupled decay removes learning rate x weight decay of every weight per step; 0.0001 x 1000 is used, chosen from seven settings on the held-out scores.

| base model, repositories held out | layer 1 | 24 | 48 | 64 |
|---|---|---|---|---|
| probe | 0.76 | 0.78 | 0.82 | 0.84 |
| probe, repository and length erased | 0.66 | 0.74 | 0.77 | 0.79 |
| same, shuffled labels | 0.50 to 0.52 | 0.46 to 0.56 | 0.44 to 0.53 | 0.43 to 0.52 |

Before standardising and decay the erased row read 0.67, 0.70, 0.68, 0.64, flat with depth, because the deep-layer probes were memorising. The restrained probe shows the signal growing with depth.

The layer-halves runs are the published recipe with LoRA on half the blocks (`--layers`), 54M trainable weights each against 109M.

| public val, 679 same-repository pairs | base | all blocks (`sft/20261113-173045`) | blocks 32 to 63 | blocks 0 to 31 |
|---|---|---|---|---|
| held-out loss, nats per token | 1.306 | 0.873 | 0.894 | 0.904 |
| written verdict, AUC | 0.62 | 0.80 | 0.81 | 0.80 |
| written verdict, length removed | 0.58 | 0.78 | 0.79 | 0.78 |
| probe at layer 64, AUC | 0.79 | 0.83 | 0.83 | 0.82 |
| probe at layer 64, length removed | 0.74 | 0.80 | 0.80 | 0.79 |
| probe at layer 32, length removed | 0.72 | 0.77 | 0.72 | 0.75 |

Probes here are trained on the 1,177 training functions and scored on the public val; "length removed" takes the straight-line trend on log length out of the score before ranking. Two runs of one recipe differ by about 0.03 AUC.

Three readings. The adapter's ranking is not a length artefact: it keeps 0.78 with length removed. Which half of the network carries the LoRA weights does not matter: either half alone reaches the full adapter's ranking, and the adapter trained only on early blocks still shows its gain in the late vectors. And the base model's vectors already rank nearly as well as the adapters' (0.74 against 0.80, inside the noise of 100 functions) while its written verdict does not (0.58): most of what fine-tuning bought is the model using a signal it already carried. About a fifth of pairs stay misranked by every reading, vectors or verdict, which points at the data and not at where in the network the training happens.

### Data for release

`dataset/train.jsonl`, `val.jsonl` and `train-traces.jsonl` carry repository, commits, path, line, licence and the fix pull request per row; the clones are public. Nothing from the work data enters this directory. The adapter is trained only on these traces and can be published under its potato name once a run is worth publishing.
