import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data.load_mnist import get_datasets
from models.cnn import ConvNet

device = "mps" if torch.backends.mps.is_available() else "cpu"

train_set, test_set = get_datasets()
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

def evaluate(model, loader, device):
    """
    Returns test accuracy as a fraction in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total

for epoch in range(5):

    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    acc = evaluate(model, test_loader, device)
    print(f"epoch {epoch + 1} loss {avg_loss:.4f} test acc {acc:.4f}")

torch.save(model.state_dict(), "outputs/checkpoints/cnn_tier0.pt")


        