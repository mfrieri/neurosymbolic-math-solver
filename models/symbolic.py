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
        self.n_digits = n_digits
        self.n_sums = 2 * (n_digits - 1) + 1
        digits = torch.arange(n_digits, dtype=torch.float32)

        # sum_table[i, j] = i + j, via broadcasting a column against a row.
        sum_table = digits.unsqueeze(1) + digits.unsqueeze(0)

        # sum_onehot[i, j, s] = 1 when i + j == s, else 0. Routes the joint
        # distribution over digit pairs into a distribution over sums.
        sum_onehot = torch.zeros(n_digits, n_digits, self.n_sums)
        for i in range(n_digits):
            for j in range(n_digits):
                sum_onehot[i, j, i + j] = 1.0

        # Buffers, not parameters: they move with .to(device) and are saved in
        # the state_dict, but the optimizer never touches them.
        self.register_buffer("sum_table", sum_table)
        self.register_buffer("sum_onehot", sum_onehot)

    def forward(self, p1, p2):
        """Expected value of the sum. p1, p2: (batch, n_digits) probabilities.

        Returns (batch,) floats. Constrains only the *mean* of each digit
        distribution, so a CNN can satisfy it with a hedge -- e.g. representing
        a 4 as half 3 and half 5. See sum_distribution for the fix.
        """
        return torch.einsum("bi,bj,ij->b", p1, p2, self.sum_table)

    def sum_distribution(self, p1, p2):
        """Full distribution over possible sums.

        P(sum = s) = sum over all (i, j) with i + j == s of P(d1=i) * P(d2=j)

        Returns (batch, n_sums). Unlike the expected value, this penalises
        hedged digit distributions: splitting a 4 between 3 and 5 smears
        probability across two sums instead of concentrating it on one.
        """
        return torch.einsum("bi,bj,ijs->bs", p1, p2, self.sum_onehot)


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

    # The distribution form: confident inputs put all mass on one sum.
    dist = symbolic.sum_distribution(p1, p2)
    print(f"\nsum_distribution shape: {tuple(dist.shape)}, total mass {dist.sum():.4f}")
    print(f"one-hot 3 + one-hot 5 -> argmax sum {dist.argmax(dim=1).item()}")

    # The hedge that fools the expected value: half 3 / half 5 against a
    # certain 5 still averages to 9, but splits its mass across sums 8 and 10.
    hedge = torch.zeros(1, 10)
    hedge[0, 3] = 0.5
    hedge[0, 5] = 0.5
    print(f"\nhedged [.5 on 3, .5 on 5] + 5:")
    print(f"  expected value -> {symbolic(hedge, p2).item():.4f}  (looks correct)")
    hedge_dist = symbolic.sum_distribution(hedge, p2)
    nonzero = [(s, round(v, 3)) for s, v in enumerate(hedge_dist[0].tolist()) if v > 0]
    print(f"  distribution   -> {nonzero}  (mass on 8 and 10, none on 9)")

    assert symbolic(p1, p2).item() == 8.0
    assert sum(p.numel() for p in symbolic.parameters()) == 0
    assert torch.allclose(dist.sum(dim=1), torch.ones(1))
    print("\nall checks passed")
