"""Differentiable symbolic arithmetic.

Computes the expected value of a digit sum over the CNN's predicted
distributions, rather than committing to a single argmax digit. The arithmetic
itself is exact and fixed -- this module has no learnable parameters.
"""
import torch
import torch.nn as nn


class SymbolicSum(nn.Module):
    """Expected value of d1 + d2 given distributions over each digit.

    Implements  sum_i sum_j  P(d1=i) * P(d2=j) * (i + j)

    which is differentiable in both inputs, so gradients from a loss on the
    predicted sum flow back through exact arithmetic into the CNN.
    """

    def __init__(self, n_digits=10):
        super().__init__()
        digits = torch.arange(n_digits, dtype=torch.float32)

        # sum_table[i, j] = i + j, via broadcasting a column against a row.
        sum_table = digits.unsqueeze(1) + digits.unsqueeze(0)

        # A buffer, not a parameter: it moves with .to(device) and is saved in
        # the state_dict, but the optimizer never touches it.
        self.register_buffer("sum_table", sum_table)

    def forward(self, p1, p2):
        """p1, p2: (batch, n_digits) probabilities. Returns (batch,) sums."""
        return torch.einsum("bi,bj,ij->b", p1, p2, self.sum_table)


if __name__ == "__main__":
    symbolic = SymbolicSum()

    print(f"learnable parameters: {sum(p.numel() for p in symbolic.parameters())}")

    # A confident, correct CNN: one-hot on 3 and 5 should give exactly 8.
    p1 = torch.zeros(1, 10)
    p1[0, 3] = 1.0
    p2 = torch.zeros(1, 10)
    p2[0, 5] = 1.0
    print(f"one-hot 3 + one-hot 5 -> {symbolic(p1, p2).item():.4f}")

    # A maximally confused CNN: uniform over 0-9 has mean 4.5, so 9.0.
    uniform = torch.full((1, 10), 0.1)
    print(f"uniform + uniform     -> {symbolic(uniform, uniform).item():.4f}")

    # Half-confident between 2 and 4 (mean 3) against a certain 5 -> 8.
    p3 = torch.zeros(1, 10)
    p3[0, 2] = 0.5
    p3[0, 4] = 0.5
    print(f"[.5 on 2, .5 on 4] + 5 -> {symbolic(p3, p2).item():.4f}")

    # The whole point: gradients must reach the inputs.
    q1 = torch.full((1, 10), 0.1, requires_grad=True)
    q2 = torch.full((1, 10), 0.1, requires_grad=True)
    loss = (symbolic(q1, q2) - 8.0).abs().sum()
    loss.backward()
    print(f"gradient reaches inputs: {q1.grad is not None and q1.grad.abs().sum() > 0}")

    assert symbolic(p1, p2).item() == 8.0
    assert sum(p.numel() for p in symbolic.parameters()) == 0
    print("all checks passed")
