"""Trains the neurosymbolic pipeline on sums alone.

The only supervision is (pair image, true sum). Digit labels are present in the
dataset for evaluation, but are discarded here -- the CNN has to discover digit
recognition purely from being right or wrong about sums.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.make_pairs import get_pair_datasets
from pipeline import NeurosymbolicSolver

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CHECKPOINT = "outputs/checkpoints/neurosymbolic.pt"


def evaluate(model, loader, device):
    """Fraction of pairs whose predicted sum rounds to the true sum."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, sums, _, _ in loader:
            images, sums = images.to(device), sums.to(device)
            predicted = model(images)
            correct += (predicted.round() == sums).sum().item()
            total += sums.size(0)

    model.train()
    return correct / total


def train(epochs=EPOCHS, train_sums=(0, 18), test_sums=(0, 18),
          checkpoint=CHECKPOINT, seed=0):
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_set, test_set = get_pair_datasets(train_sums=train_sums,
                                            test_sums=test_sums)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = NeurosymbolicSolver().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        running_loss = 0.0

        # The two underscores are digit labels. They are never used.
        for images, sums, _, _ in train_loader:
            images, sums = images.to(device), sums.to(device)

            optimizer.zero_grad()
            predicted = model(images)
            loss = criterion(predicted, sums)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1} loss {avg_loss:.4f} sum acc {acc:.4f}")

    if checkpoint:
        torch.save(model.state_dict(), checkpoint)

    return model


if __name__ == "__main__":
    train()
