"""Wires the CNN and the symbolic module into an end-to-end pipeline.

A (1, 28, 56) pair image is split into two digit crops, both passed through the
*same* CNN, softmaxed into distributions, and handed to the symbolic module,
which returns a predicted sum. Only the CNN has learnable parameters.
"""
import torch
import torch.nn as nn

from models.cnn import ConvNet
from models.symbolic import SymbolicSum


class NeurosymbolicSolver(nn.Module):
    """Pair image -> predicted sum, differentiable end to end."""

    def __init__(self, digit_width=28):
        super().__init__()
        self.digit_width = digit_width
        self.cnn = ConvNet()
        self.symbolic = SymbolicSum()

    def digit_probs(self, images):
        """Returns the two (batch, 10) digit distributions for a pair image."""
        w = self.digit_width
        left = images[..., :w]
        right = images[..., w:]

        # One CNN, called twice: the weights are shared, so the recognizer is
        # position-independent and sees every digit in the dataset.
        p1 = torch.softmax(self.cnn(left), dim=1)
        p2 = torch.softmax(self.cnn(right), dim=1)
        return p1, p2

    def forward(self, images):
        """images: (batch, 1, 28, 56). Returns (batch,) predicted sums."""
        p1, p2 = self.digit_probs(images)
        return self.symbolic(p1, p2)

    def sum_distribution(self, images):
        """images: (batch, 1, 28, 56). Returns (batch, 19) sum probabilities."""
        p1, p2 = self.digit_probs(images)
        return self.symbolic.sum_distribution(p1, p2)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.make_pairs import get_pair_datasets

    solver = NeurosymbolicSolver()

    cnn_params = sum(p.numel() for p in solver.cnn.parameters())
    total_params = sum(p.numel() for p in solver.parameters())
    print(f"cnn params: {cnn_params}  total params: {total_params}")

    train, _ = get_pair_datasets(n_train=256, n_test=16)
    images, sums, digit1, digit2 = next(iter(DataLoader(train, batch_size=8)))

    predicted = solver(images)
    print(f"input {tuple(images.shape)} -> output {tuple(predicted.shape)}")
    print(f"true sums:      {sums.tolist()}")
    print(f"predicted sums: {[round(v, 2) for v in predicted.tolist()]}")

    # Untrained, both distributions are near-uniform, so sums cluster near 9.
    loss = (predicted - sums).abs().mean()
    loss.backward()
    grad = solver.cnn.conv1.weight.grad
    print(f"gradient reaches conv1: {grad is not None and grad.abs().sum() > 0}")

    assert predicted.shape == sums.shape
    assert total_params == cnn_params, "symbolic module must add no parameters"
    print("all checks passed")
