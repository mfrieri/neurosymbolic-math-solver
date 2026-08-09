"""Trains the neurosymbolic pipeline on sums alone.

The only supervision is (pair image, true sum). Digit labels are present in the
dataset for evaluation, but are discarded here -- the CNN has to discover digit
recognition purely from being right or wrong about sums.

Two objectives are available:

  "expected"     L1 loss on the expected value of the sum. Matches the classic
                 hand-rolled formulation, but constrains only the mean of each
                 digit distribution, so the CNN can hedge (representing a 4 as
                 half 3 and half 5) and still score perfectly.

  "distribution" NLL loss on the full distribution over sums, as DeepProbLog
                 does. Hedging smears probability across neighbouring sums, so
                 this objective forces genuinely peaked digit distributions.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.make_pairs import get_pair_datasets
from models.baseline import BaselineRegressor
from pipeline import NeurosymbolicSolver

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CHECKPOINTS = {
    "expected": "outputs/checkpoints/neurosymbolic.pt",
    "distribution": "outputs/checkpoints/neurosymbolic_dist.pt",
    "baseline": "outputs/checkpoints/baseline.pt",
}
LABELS = {
    "expected": "Neurosymbolic (expected value)",
    "distribution": "Neurosymbolic (sum distribution)",
    "baseline": "End-to-end regression",
}


def evaluate(model, loader, device, loss_mode="expected"):
    """Fraction of pairs whose predicted sum matches the true sum."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, sums, _, _ in loader:
            images, sums = images.to(device), sums.to(device)

            # Only the distribution model predicts by argmax over sums; the
            # expected-value and baseline models emit a float to round.
            if loss_mode == "distribution":
                predicted = model.sum_distribution(images).argmax(dim=1).float()
            else:
                predicted = model(images).round()

            correct += (predicted == sums).sum().item()
            total += sums.size(0)

    model.train()
    return correct / total


def train(epochs=EPOCHS, loss_mode="expected", train_sums=(0, 18),
          test_sums=(0, 18), checkpoint=None, seed=0, verbose=True):
    """Trains one model. Returns (model, history) where history is a list of
    test accuracies, one per epoch."""
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if checkpoint is None:
        checkpoint = CHECKPOINTS[loss_mode]

    train_set, test_set = get_pair_datasets(train_sums=train_sums,
                                            test_sums=test_sums)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    if loss_mode == "baseline":
        model = BaselineRegressor().to(device)
        criterion = nn.L1Loss()
    else:
        model = NeurosymbolicSolver().to(device)
        criterion = nn.L1Loss() if loss_mode == "expected" else nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(epochs):
        running_loss = 0.0

        # The two underscores are digit labels. They are never used.
        for images, sums, _, _ in train_loader:
            images, sums = images.to(device), sums.to(device)

            optimizer.zero_grad()
            if loss_mode == "distribution":
                # NLLLoss wants log-probabilities; the epsilon keeps log finite
                # when the model assigns a sum exactly zero probability.
                probs = model.sum_distribution(images)
                loss = criterion(torch.log(probs + 1e-12), sums.long())
            else:
                loss = criterion(model(images), sums)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        acc = evaluate(model, test_loader, device, loss_mode)
        history.append(acc)

        if verbose:
            avg_loss = running_loss / len(train_loader)
            print(f"epoch {epoch + 1} loss {avg_loss:.4f} sum acc {acc:.4f}")

    if checkpoint:
        torch.save(model.state_dict(), checkpoint)

    return model, history


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "expected"
    print(f"training with loss_mode={mode!r}")
    train(loss_mode=mode)
