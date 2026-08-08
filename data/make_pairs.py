"""Synthesizes digit-pair images and sum labels from MNIST.

Each sample is two MNIST digits concatenated side by side into a (1, 28, 56)
image, labelled with the sum of the two digits. The individual digit labels are
returned as well, but they exist only for evaluation -- training must never
supervise on them.
"""
import random

import torch
from torch.utils.data import Dataset

from data.load_mnist import get_datasets


class MNISTPairs(Dataset):
    """Pairs of MNIST digits labelled by their sum.

    Stores only index pairs into the underlying MNIST set; images are
    concatenated on demand in __getitem__.

    Args:
        mnist: an MNIST dataset from load_mnist.get_datasets()
        n_pairs: how many pairs to generate
        sum_range: inclusive (low, high) bounds on the digit sum. Used to hold
            out sums for the compositional generalization test -- train on
            (0, 9), test on (10, 18).
        seed: makes the sampled pairs reproducible across runs and models
    """

    def __init__(self, mnist, n_pairs, sum_range=(0, 18), seed=0):
        self.mnist = mnist
        low, high = sum_range
        rng = random.Random(seed)

        # .targets is the label tensor, so we can filter on sums without
        # decoding any images.
        targets = mnist.targets.tolist()
        n = len(targets)

        self.pairs = []
        while len(self.pairs) < n_pairs:
            i, j = rng.randrange(n), rng.randrange(n)
            if low <= targets[i] + targets[j] <= high:
                self.pairs.append((i, j))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        img1, digit1 = self.mnist[i]
        img2, digit2 = self.mnist[j]

        # (1, 28, 28) + (1, 28, 28) -> (1, 28, 56); dim=2 is width in (C, H, W).
        image = torch.cat([img1, img2], dim=2)
        digit_sum = torch.tensor(float(digit1 + digit2))

        return image, digit_sum, digit1, digit2


def get_pair_datasets(n_train=60_000, n_test=10_000, train_sums=(0, 18),
                      test_sums=(0, 18), root="data"):
    """Builds train/test pair datasets from the matching MNIST splits."""
    mnist_train, mnist_test = get_datasets(root)
    train = MNISTPairs(mnist_train, n_train, sum_range=train_sums, seed=0)
    test = MNISTPairs(mnist_test, n_test, sum_range=test_sums, seed=1)
    return train, test


if __name__ == "__main__":
    train, test = get_pair_datasets(n_train=1000, n_test=200)
    image, digit_sum, digit1, digit2 = train[0]

    print(f"train: {len(train)}  test: {len(test)}")
    print(f"image shape: {tuple(image.shape)}  sum: {digit_sum.item()}")
    print(f"digits: {digit1} + {digit2} = {digit1 + digit2}")

    assert image.shape == (1, 28, 56)
    assert digit_sum.item() == digit1 + digit2

    held_out = MNISTPairs(train.mnist, 500, sum_range=(0, 9))
    sums = [held_out[k][1].item() for k in range(len(held_out))]
    print(f"sum_range=(0, 9) -> observed sums {min(sums):.0f} to {max(sums):.0f}")
    assert max(sums) <= 9
    print("all checks passed")
