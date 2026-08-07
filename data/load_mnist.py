"""Downloads MNIST and sanity-checks it. Run once before building the CNN."""
import torch
from torchvision import datasets, transforms

# 0.1307 / 0.3081 are the standard MNIST channel mean and std.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_datasets(root="data"):
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    return train, test


if __name__ == "__main__":
    train, test = get_datasets()
    print(f"train: {len(train)}  test: {len(test)}")

    img, label = train[0]
    print(f"image shape: {tuple(img.shape)}  dtype: {img.dtype}  label: {label}")
    print(f"pixel range: {img.min():.3f} to {img.max():.3f}")

    loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    batch, labels = next(iter(loader))
    print(f"batch shape: {tuple(batch.shape)}  labels shape: {tuple(labels.shape)}")
