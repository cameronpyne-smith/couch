import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.first_token import auc, matched_threshold, probability
from finetune.lora import load_lora_adapter
from finetune.sft import IGNORE, MODEL, encode, load_examples, load_model
from finetune.verdict import parse_verdict, verdict_probability
from projects.pr_reviews.generator import NO_BREAK, OUTPUT
from projects.pr_reviews.traces import BREAK, VERDICT
from projects.pr_reviews.traces import NO_BREAK as NO_BREAK_VERDICT

MAX_NEW_TOKENS = 200


def is_no_break(text):
    head = text.strip().strip("`").strip().lower()
    return head.startswith(NO_BREAK) or head.startswith("i cannot find a break")


def score(model, tokeniser, example, device, no_id):
    ids, labels = encode(example, tokeniser)
    input_ids = torch.tensor([ids], device=device)
    output = model(input_ids=input_ids, labels=torch.tensor([labels], device=device))
    prompt_length = labels.count(IGNORE)
    return input_ids, prompt_length, output.loss.item(), probability(output.logits[0, prompt_length - 1], no_id)


def answer(model, tokeniser, example, device, no_id, threshold, max_new_tokens=MAX_NEW_TOKENS):
    input_ids, prompt_length, loss, p_no = score(model, tokeniser, example, device, no_id)
    if p_no >= threshold:
        text = NO_BREAK
    else:
        generated = model.generate(
            input_ids[:, :prompt_length], max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokeniser.pad_token_id, begin_suppress_tokens=[no_id]
        )
        text = tokeniser.decode(generated[0, prompt_length:], skip_special_tokens=True)
    return row(example, text, loss, p_no, is_no_break(text))


def row(example, text, loss, p_no, refused, verdict=None):
    return {
        "kind": example["kind"],
        "repo": example["repo"],
        "number": example["number"],
        "gold": example.get("gold", example["answer"]),
        "generated": text.strip(),
        "loss": loss,
        "p_no": p_no,
        "refused": refused,
        "verdict": verdict,
    }


def verdict_of(tokeniser, generated, scores):
    text = tokeniser.decode(generated, skip_special_tokens=True)
    verdict = parse_verdict(text, VERDICT, (BREAK, NO_BREAK_VERDICT))
    p_break = verdict_probability(tokeniser, generated, scores, VERDICT, BREAK, "no")
    return text, verdict or "none", 0.5 if p_break is None else 1 - p_break


def answer_by_verdict(model, tokeniser, example, device, no_id, max_new_tokens, samples=0, temperature=0.7):
    input_ids, prompt_length, loss, _ = score(model, tokeniser, example, device, no_id)
    prompt = input_ids[:, :prompt_length]
    output = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokeniser.pad_token_id, output_scores=True, return_dict_in_generate=True)
    text, verdict, p_no = verdict_of(tokeniser, output.sequences[0, prompt_length:].tolist(), output.scores)
    sampled = []
    if samples:
        output = model.generate(
            prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=samples,
            pad_token_id=tokeniser.pad_token_id, output_scores=True, return_dict_in_generate=True,
        )
        for k in range(samples):
            sample_text, sample_verdict, sample_p_no = verdict_of(tokeniser, output.sequences[k, prompt_length:].tolist(), [step[k : k + 1] for step in output.scores])
            sampled.append({"verdict": sample_verdict, "p_no": sample_p_no, "text": sample_text})
    result = row(example, text, loss, mean([p_no] + [s["p_no"] for s in sampled]), verdict == NO_BREAK_VERDICT, verdict)
    result["p_no_greedy"] = p_no
    result["samples"] = sampled
    return result


def run(name, model, tokeniser, examples, device, path, threshold, max_new_tokens=MAX_NEW_TOKENS, mode="first", samples=0):
    no_id = tokeniser(NO_BREAK, add_special_tokens=False).input_ids[0]
    model.eval()
    rows = []
    with torch.no_grad(), path.open("w") as handle:
        for index, example in enumerate(examples):
            if mode == "verdict":
                rows.append(answer_by_verdict(model, tokeniser, example, device, no_id, max_new_tokens, samples))
            else:
                rows.append(answer(model, tokeniser, example, device, no_id, threshold, max_new_tokens))
            handle.write(json.dumps(rows[-1]) + "\n")
            handle.flush()
            if index % 50 == 0:
                print(f"  {name} {index}/{len(examples)}", flush=True)
    report(name, rows)
    return rows


def match_threshold(model, tokeniser, examples, device, base):
    no_id = tokeniser(NO_BREAK, add_special_tokens=False).input_ids[0]
    answered = sum(not is_no_break(r["generated"]) for r in base if r["kind"] == "real")
    model.eval()
    with torch.no_grad():
        scores = [score(model, tokeniser, e, device, no_id)[3] for e in examples if e["kind"] == "real"]
    threshold = matched_threshold(scores, answered)
    print(f"matched threshold {threshold:.4f} answers {answered} real examples like the base", flush=True)
    return threshold


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def report(name, rows):
    real = [r for r in rows if r["kind"] == "real"]
    clean = [r for r in rows if r["kind"] == "clean"]
    specificity = mean([r.get("refused", is_no_break(r["generated"])) for r in clean])
    miss = mean([r.get("refused", is_no_break(r["generated"])) for r in real])
    ranking = auc([r["p_no"] for r in clean], [r["p_no"] for r in real])
    unconcluded = sum(r.get("verdict") == "none" for r in rows)
    print(
        f"{name}: real loss {mean([r['loss'] for r in real]):.3f} clean loss {mean([r['loss'] for r in clean]):.3f}"
        f" p_no real {mean([r['p_no'] for r in real]):.3f} clean {mean([r['p_no'] for r in clean]):.3f} auc {ranking:.3f}"
        f" no-break on clean {specificity:.3f} ({len(clean)}) no-break on real {miss:.3f} ({len(real)})"
        f" balanced {(specificity + 1 - miss) / 2:.3f}" + (f" no verdict {unconcluded}" if unconcluded else ""),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--bits", type=int, default=16, choices=(16, 4))
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--alpha", default=32, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--kind", choices=("real", "clean"))
    parser.add_argument("--mode", choices=("first", "verdict"), default="first", help="first: p(refuse) at the first answer token; verdict: generate, then read p(break) at the verdict line")
    parser.add_argument("--samples", type=int, default=0, help="verdict mode: extra sampled traces per example whose verdict probabilities are averaged into p_no")
    parser.add_argument("--run-dir", type=Path, help="adapter run directory; without it only the base model answers")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match", action="store_true", help="pick the adapter threshold that answers as many real examples as the base did")
    parser.add_argument("--skip-base", action="store_true", help="answer with the adapter only")
    args = parser.parse_args()

    device = "cuda"
    if args.run_dir:
        directory = args.run_dir
    else:
        directory = args.data / "base" / args.model.split("/")[-1]
        directory.mkdir(parents=True, exist_ok=True)
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model, _ = load_model(args.rank, args.alpha, device, args.model, args.bits)
    examples = load_examples(args.data / f"{args.split}.jsonl")
    if args.kind:
        examples = [e for e in examples if e["kind"] == args.kind]
    suffix = ("" if args.split == "val" else f"-{args.split}") + (f"-{args.kind}" if args.kind else "") + ("" if args.max_new_tokens == MAX_NEW_TOKENS else f"-n{args.max_new_tokens}") + ("" if args.mode == "first" else f"-{args.mode}") + (f"-s{args.samples}" if args.samples else "") + (f"-{args.data.name}" if args.run_dir else "")
    if args.limit:
        examples = random.Random(0).sample(examples, args.limit)

    base_suffix = suffix.replace(f"-s{args.samples}", "").replace(f"-{args.data.name}", "")
    saved = args.data / "base" / args.model.split("/")[-1] / f"answers-base{base_suffix}.jsonl"
    if args.skip_base:
        base = []
    elif args.run_dir and saved.exists() and not args.limit:
        base = [json.loads(line) for line in saved.read_text().splitlines()]
        report("base", base)
    else:
        base = run("base", model, tokeniser, examples, device, directory / f"answers-base{base_suffix}.jsonl", args.threshold, args.max_new_tokens, args.mode)
    if args.run_dir:
        load_lora_adapter(model, directory / "adapter.pt")
        threshold = match_threshold(model, tokeniser, examples, device, base) if args.match else args.threshold
        adapter = run("adapter", model, tokeniser, examples, device, directory / f"answers-adapter{suffix}.jsonl", threshold, args.max_new_tokens, args.mode, args.samples)
    else:
        adapter = [None] * len(base)

    real_pairs = [pair for pair in zip(base, adapter) if pair[0]["kind"] == "real"]
    for b, a in random.Random(0).sample(real_pairs, min(10, len(real_pairs))):
        print("=" * 80)
        print("gold:     " + b["gold"])
        print("base:     " + b["generated"])
        if a is not None:
            print("adapter:  " + a["generated"])
