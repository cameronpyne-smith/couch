import torch


class AdamW:
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    ):
        self.params = list(params)
        self.learning_rate = learning_rate
        self.b1, self.b2 = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.ms = [torch.zeros_like(param) for param in self.params]
        self.vs = [torch.zeros_like(param) for param in self.params]
    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p, m, v in zip(self.params, self.ms, self.vs):
                g = p.grad
                if g is None:
                    continue
                m[:] = m * self.b1 + (1 - self.b1) * g
                v[:] = v * self.b2 + (1 - self.b2) * g * g
                m_hat = m / (1 - self.b1**self.t)
                v_hat = v / (1 - self.b2**self.t)
                p -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                p -= self.learning_rate * self.weight_decay * p
