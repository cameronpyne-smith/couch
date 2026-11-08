from collections import defaultdict

import torch


def sample(model, tokeniser, prompt_ids, count, temperature, max_new_tokens, suppress=(), top_p=0.95):
    input_ids = torch.tensor([prompt_ids], device=model.device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=count,
        pad_token_id=tokeniser.pad_token_id,
        begin_suppress_tokens=list(suppress) or None,
    )
    return [tokeniser.decode(row[len(prompt_ids) :], skip_special_tokens=True).strip() for row in generated]


def any_of_k(rows, key, hit):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(hit(row))
    prompts = list(groups.values())
    count = max(len(p) for p in prompts)
    print(f"{len(prompts)} prompts, up to {count} samples each")
    print(f"per-sample hit rate {sum(map(sum, prompts)) / sum(map(len, prompts)):.3f}")
    for k in range(1, count + 1):
        print(f"any of first {k}: {sum(any(p[:k]) for p in prompts)}/{len(prompts)}")
    print("hits per prompt:", sorted(map(sum, prompts), reverse=True))
    return groups
