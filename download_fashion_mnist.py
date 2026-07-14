from pathlib import Path

from torchvision.datasets import FashionMNIST


DATA_DIR = Path(__file__).resolve().parent / "data"
FASHION_MNIST_MIRROR = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion/"
)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    FashionMNIST.mirrors = [FASHION_MNIST_MIRROR, *FashionMNIST.mirrors]
    train_dataset = FashionMNIST(root=str(DATA_DIR), train=True, download=True)
    test_dataset = FashionMNIST(root=str(DATA_DIR), train=False, download=True)

    print(f"Downloaded Fashion-MNIST to: {DATA_DIR}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()
