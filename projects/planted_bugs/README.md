# planted_bugs: find the one changed line

The third project on couch's fine-tuning layer, started 2026-11-13 as a short, visible fine-tune for the Windlet Playground. `public_bugs` asks a model whether a real function had a real bug, which takes a 27B model and teacher traces; here the bug is planted, so the answer is known exactly and a 1.5B model learns it in minutes.

## Idea

Take a function from a permissively licensed open-source repository and change one operator on one line: `==` to `!=`, `<` to `<=`, `&&` to `||`, `true` to `false`, a dropped `!`, a dropped `+ 1` or `- 1`, a `?.` turned into `.`. Show the model the numbered function and ask for `Line: N`, or `Line: none` when it is the untouched original. Every function appears twice, once clean and once planted, so a model that answers by length or by style cannot score.

An earlier probe asked for a verdict alone (`break` / `no break`) on the `public_bugs` functions: the AUC rose from 0.60 to 0.87, but the number of added lines in the diff alone scored 0.76, so the model was mostly reading size. The planted pairs remove that shortcut: the two functions of a pair have the same size.

## Data

Built from the clones under `~/training-data/code/public-bugs/clones/` (see `public_bugs/README.md`), written to `data/planted-bugs.jsonl`: one row per function with its `split`, `pair`, repository, path, licence, the numbered code, `bug_line` (null when clean) and the mutation. Functions are 6 to 60 lines, from non-test, non-generated files, located with tree-sitter; a mutation is never inside a string or a comment.

- train: 5,035 pairs from 35 repositories, at most 150 per repository.
- val: 300 pairs, 100 from each of zod, Humanizer and Quartz.NET, which the training never sees.

The mix is skewed towards what code has most of: `==` / `!=` and booleans are half of it, off-by-ones under 2%.

Unlike the other projects' data it lives in this repository (11.6 MB), because the Windlet Playground offers it as its example by a pinned raw GitHub address. Licences MIT, Apache-2.0, BSD-3-Clause, PostgreSQL; `data/NOTICE` holds each repository's licence text and the commit its functions come from.

## Files

| file | what it does |
|---|---|
| `plant.py` | mines functions from the clones, plants one mutation per function, writes the pairs |
| `train.py` | LoRA (rank 16, every projection) on Qwen2.5-Coder-1.5B-Instruct, loss on the answer tokens only; evaluates at step 0 and every tenth of the run, prints Windlet's `windlet-training/1` metric lines, keeps the adapter of the lowest validation loss and writes before/after answers for a few val functions |
| `Dockerfile` | the image the Playground runs, with the base model baked in, so a node never downloads it from Hugging Face |

```
uv run python -m projects.planted_bugs.plant
uv run python -m projects.planted_bugs.train --steps 300
docker buildx build -f projects/planted_bugs/Dockerfile -t ghcr.io/cameronpyne-smith/couch-planted-bugs:<version> --push .
```

In the image the data is `/input/planted-bugs.jsonl` and everything it keeps goes to `/output`: `adapter.pt`, `samples.json` (the before/after answers) and `result.json`.

## Metrics

- `val_found`: share of planted functions where the answer names the planted line.
- `val_false_alarm`: share of clean functions where the answer names any line.
- `val_pair_ranked`: share of pairs where the planted function scores more likely buggy than its clean twin, scored as 1 − p(`none`) at the answer.
- `val_auc`: the same score over all 600 functions.
- `loss`, `val_loss`: cross-entropy on the answer tokens.

## Results, 2026-11-13 (probe, one seed, RTX 5090)

| step | minutes | found line | false alarm | pair ranked | AUC |
|---|---|---|---|---|---|
| 0 | 0 | 0.15 | 1.00 | 0.48 | 0.49 |
| 50 | 2.2 | 0.36 | 0.36 | 0.75 | 0.69 |
| 100 | 4.5 | 0.60 | 0.34 | 0.72 | 0.79 |
| 150 | 6.7 | 0.75 | 0.39 | 0.76 | 0.82 |
| 200 | 8.9 | 0.60 | 0.23 | 0.79 | 0.84 |
| 250 | 11.1 | 0.73 | 0.30 | 0.88 | 0.87 |
| 300 | 13.4 | 0.71 | 0.26 | 0.88 | 0.87 |

Peak 10.8 GB. The base model names a line for every clean function and is right about the planted one 15% of the time; after 250 steps it finds three in four, on repositories it never trained on. Planted bugs are not real bugs: whether this transfers to `public_bugs` is untested.
