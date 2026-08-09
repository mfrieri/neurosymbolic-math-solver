"""Evaluations for the trained neurosymbolic pipeline.

Part (a): implicit digit accuracy -- feed the frozen CNN individual digits and
check its argmax against true MNIST labels. The CNN was never shown a digit
label during training, only sums, so this measures whether digit recognition
emerged on its own.
"""
import os

import torch
from torch.utils.data import DataLoader

from data.load_mnist import get_datasets
from models.cnn import ConvNet
from pipeline import NeurosymbolicSolver

CHECKPOINT = "outputs/checkpoints/neurosymbolic.pt"
DIST_CHECKPOINT = "outputs/checkpoints/neurosymbolic_dist.pt"
TIER0_CHECKPOINT = "outputs/checkpoints/cnn_tier0.pt"


def load_solver(checkpoint=CHECKPOINT, device="cpu"):
    """Loads a trained pipeline and freezes it."""
    solver = NeurosymbolicSolver().to(device)

    # strict=False so checkpoints saved before a buffer was added still load.
    # Buffers hold fixed constants rebuilt by __init__, not learned state --
    # but every *weight* must still be present, which the assert enforces.
    missing, _ = solver.load_state_dict(torch.load(checkpoint, map_location=device),
                                        strict=False)
    assert not [k for k in missing if not k.startswith("symbolic.")], missing

    solver.eval()
    for param in solver.parameters():
        param.requires_grad = False
    return solver


def digit_accuracy(cnn, device="cpu", batch_size=1000):
    """Per-digit and overall accuracy of a CNN on single MNIST test images.

    Returns (overall_accuracy, per_digit_accuracy) where per_digit is a length
    10 list indexed by true label.
    """
    _, mnist_test = get_datasets()
    loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    correct = torch.zeros(10)
    total = torch.zeros(10)

    cnn.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = cnn(images).argmax(dim=1).cpu()
            labels = labels.cpu()

            for digit in range(10):
                mask = labels == digit
                correct[digit] += (preds[mask] == digit).sum()
                total[digit] += mask.sum()

    per_digit = (correct / total).tolist()
    return (correct.sum() / total.sum()).item(), per_digit


def report(name, per_digit, overall):
    print(f"\n{name}   overall {overall:.4f}")
    for digit, digit_acc in enumerate(per_digit):
        bar = "#" * round(digit_acc * 40)
        print(f"    {digit}  {digit_acc:.4f}  {bar}")


def part_a(device="cpu"):
    print("Part (a): implicit digit accuracy")
    print("These CNNs were trained only on sums -- they never saw a digit label.")

    results = {}

    expected = load_solver(CHECKPOINT, device)
    results["expected-value (L1)"] = digit_accuracy(expected.cnn, device)

    if os.path.exists(DIST_CHECKPOINT):
        dist = load_solver(DIST_CHECKPOINT, device)
        results["sum-distribution (NLL)"] = digit_accuracy(dist.cnn, device)
    else:
        print(f"\n  ({DIST_CHECKPOINT} not found -- run: python -m train distribution)")

    for name, (overall, per_digit) in results.items():
        report(name, per_digit, overall)

    # Reference point: the same architecture trained with full digit labels.
    supervised = ConvNet().to(device)
    supervised.load_state_dict(torch.load(TIER0_CHECKPOINT, map_location=device))
    sup_acc, _ = digit_accuracy(supervised, device)

    print("\nsummary")
    for name, (overall, _) in results.items():
        print(f"  {name:<26} {overall:.4f}   (gap {overall - sup_acc:+.4f})")
    print(f"  {'digit-supervised (tier 0)':<26} {sup_acc:.4f}   (reference)")

    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    part_a(device)
