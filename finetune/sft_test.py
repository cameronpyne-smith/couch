import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.sft import token_loss


def test_token_loss_in_chunks_matches_one_pass():
    torch.manual_seed(0)
    head = nn.Linear(7, 11, bias=False)
    head.requires_grad_(False)
    body = nn.Linear(5, 7)
    x = torch.randn(13, 5)
    targets = torch.randint(0, 11, (13,))

    whole = F.cross_entropy(head(body(x)), targets, reduction="sum")
    whole.backward()
    expected = body.weight.grad.clone()
    body.weight.grad = None

    chunked = token_loss(head, body(x), targets, chunk=4)
    chunked.backward()

    assert torch.allclose(chunked, whole)
    assert torch.allclose(body.weight.grad, expected, atol=1e-6)


def test_token_loss_without_gradients():
    head = nn.Linear(7, 11, bias=False)
    states = torch.randn(13, 7)
    targets = torch.randint(0, 11, (13,))
    with torch.no_grad():
        assert torch.allclose(token_loss(head, states, targets, chunk=4), F.cross_entropy(head(states), targets, reduction="sum"))
