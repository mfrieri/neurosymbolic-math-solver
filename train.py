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
from pipeline import NeurosymbolicSolver

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CHECKPOINTS = {
    "expected": "outputs/checkpoints/neurosymbolic.pt",
    "distribution": "outputs/checkpoints/neurosymbolic_dist.pt",
}


def evaluate(model, loader, device, loss_mode="expected"):
    """Fraction of pairs whose predicted sum matches the true sum."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, sums, _, _ in loader:
            images, sums = images.to(device), sums.to(device)

            if loss_mode == "expected":
                predicted = model(images).round()
            else:
                predicted = model.sum_distribution(images).argmax(dim=1).float()

            correct += (predicted == sums).sum().item()
            total += sums.size(0)

    model.train()
    return correct / total


def train(epochs=EPOCHS, loss_mode="expected", train_sums=(0, 18),
          test_sums=(0, 18), checkpoint=None, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if checkpoint is None:
        checkpoint = CHECKPOINTS[loss_mode]

    train_set, test_set = get_pair_datasets(train_sums=train_sums,
                                            test_sums=test_sums)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = NeurosymbolicSolver().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() if loss_mode == "expected" else nn.NLLLoss()

    for epoch in range(epochs):
        running_loss = 0.0

        # The two underscores are digit labels. They are never used.
        for images, sums, _, _ in train_loader:
            images, sums = images.to(device), sums.to(device)

            optimizer.zero_grad()
            if loss_mode == "expected":
                loss = criterion(model(images), sums)
            else:
                # NLLLoss wants log-probabilities; the epsilon keeps log finite
                # when the model assigns a sum exactly zero probability.
                probs = model.sum_distribution(images)
                loss = criterion(torch.log(probs + 1e-12), sums.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            avg_loss = running_loss / len(train_loader)
            acc = evaluate(model, test_loader, device, loss_mode)
            print(f"epoch {epoch + 1} loss {avg_loss:.4f} sum acc {acc:.4f}")

    if checkpoint:
        torch.save(model.state_dict(), checkpoint)

    return model


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "expected"
    print(f"training with loss_mode={mode!r}")
    train(loss_mode=mode)
