"""End-to-end regression baseline -- the control for the Tier 1 experiment.

Same convolutional trunk as ConvNet, but it reads the whole (1, 28, 56) pair
image at once and regresses the sum directly. There is no digit bottleneck and
no symbolic layer: the mapping from pixels to sum is entirely learned.

This is what the neurosymbolic pipeline is measured against. It should match on
sums it trained on and fail on sums it never saw, because nothing in it encodes
what addition *is*.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineRegressor(nn.Module):
    """Pair image -> predicted sum, with no symbolic structure."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 28x56 -> 26x54 -> 13x27 -> 11x25 -> 5x12
        self.fc1 = nn.Linear(32 * 5 * 12, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # squeeze so the output is (batch,), matching the neurosymbolic solver
        return self.fc2(x).squeeze(1)


if __name__ == "__main__":
    model = BaselineRegressor()
    x = torch.randn(4, 1, 28, 56)
    out = model(x)
    print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    assert out.shape == (4,)
    print("all checks passed")
