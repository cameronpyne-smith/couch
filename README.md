# couch

The lazy man's LLM coach. You sit down, we'll do the training.

## Layout

- Root: a GPT written from scratch and trained on text (`model.py`, `train.py`, `optimiser.py`, `tokeniser.py`, `generate.py`, `plot.py`) with tests beside each module.
- `finetune/`: the fine-tuning layer. `lora.py` (LoRA wrapper, apply, merge, adapter save and load), `finetune_gpt.py` and `merge_gpt.py` (LoRA on the from-scratch GPT), `sft.py` (supervised fine-tuning of a pretrained chat model with LoRA or QLoRA, masked prompt, token-budget batches), `first_token.py` (first-token probability, AUC, matched thresholds), `verdict.py` (the same decision read at the token after a `Verdict:` marker in a reason-first answer), `sampling.py` (k samples per prompt and any-of-k summaries), `ollama.py` (JSON-schema calls to a local Ollama model, used for LLM judges and labellers).
- `projects/`: one directory per training project built on the layers above, each with its own README recording the data, the recipe and the results. Project code holds paths only; training data stays outside the repository. `pr_reviews/` fine-tunes on an internal code-review export (unpublishable); `public_bugs/` builds the same task from open-source bug-fix pull requests so the data and adapters can be released.
