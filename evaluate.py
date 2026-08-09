"""Evaluations for the trained neurosymbolic pipeline.

Part (a): implicit digit accuracy -- feed the frozen CNN individual digits and
check its argmax against true MNIST labels. The CNN was never shown a digit
label during training, only sums, so this measures whether digit recognition
emerged on its own.
"""
import json
import os

import torch
from torch.utils.data import DataLoader

from data.load_mnist import get_datasets
from models.cnn import ConvNet
from pipeline import NeurosymbolicSolver

CHECKPOINT = "outputs/checkpoints/neurosymbolic.pt"
DIST_CHECKPOINT = "outputs/checkpoints/neurosymbolic_dist.pt"
TIER0_CHECKPOINT = "outputs/checkpoints/cnn_tier0.pt"
HISTORY_PATH = "outputs/generalization.json"
FIGURE_PATH = "outputs/figures/generalization.png"


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


def part_b(epochs=10, force=False):
    """Compositional generalization: train on sums 0-9, test on sums 10-18.

    The symbolic layer computes sums it was never trained on exactly, because
    no learnable parameter sits anywhere on the sum -- P(sum=s) is derived from
    the digit distributions through a fixed table. The regression baseline has
    no such structure: its head only ever produced outputs in 0-9.
    """
    from train import LABELS, train

    print("\nPart (b): compositional generalization")
    print("train on sums 0-9, test on sums 10-18 (never seen together)\n")

    if os.path.exists(HISTORY_PATH) and not force:
        with open(HISTORY_PATH) as f:
            results = json.load(f)
        print(f"  (loaded cached results from {HISTORY_PATH}; pass force=True to retrain)")
    else:
        results = {}
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

        for mode in ["distribution", "expected", "baseline"]:
            print(f"training {LABELS[mode]}...")
            _, history = train(epochs=epochs, loss_mode=mode,
                               train_sums=(0, 9), test_sums=(10, 18),
                               checkpoint=None, verbose=True)
            results[mode] = history

            # Write after each model so a later failure doesn't discard runs
            # that already finished.
            with open(HISTORY_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print()

    print("held-out sum accuracy (sums 10-18):")
    for mode, history in results.items():
        print(f"  {LABELS[mode]:<34} final {history[-1]:.4f}   best {max(history):.4f}")

    plot_generalization(results)
    return results


def plot_generalization(results, path=FIGURE_PATH):
    """Accuracy curves on held-out sums, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from train import LABELS

    # Categorical slots 1-3, validated for all-pairs CVD separation.
    colors = {"distribution": "#2a78d6", "expected": "#eb6834", "baseline": "#1baf7a"}

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for mode, history in results.items():
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history, linewidth=2, color=colors[mode],
                marker="o", markersize=5, label=LABELS[mode])
        # Direct labels: required relief for the low-contrast slot, and they
        # keep identity off color alone.
        ax.annotate(f"{history[-1]:.1%}", (len(history), history[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=9, color="#52514e")

    ax.set_xlabel("Epoch", fontsize=10, color="#52514e")
    ax.set_ylabel("Accuracy on unseen sums (10-18)", fontsize=10, color="#52514e")
    ax.set_title("Compositional generalization: trained only on sums 0-9",
                 fontsize=12, color="#0b0b0b", pad=12)

    n_epochs = len(next(iter(results.values())))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0.7, n_epochs + 1.1)
    ax.set_xticks(range(1, n_epochs + 1))
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors="#52514e", labelsize=9)

    # Legend below the axes so it can never collide with a curve.
    ax.legend(frameon=False, fontsize=9, labelcolor="#52514e", ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    print(f"\nwrote {path}")


if __name__ == "__main__":
    import sys

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("a", "all"):
        part_a(device)
    if which in ("b", "all"):
        part_b(force="--force" in sys.argv)
