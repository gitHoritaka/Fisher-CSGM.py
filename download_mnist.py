from pathlib import Path

from torchvision.datasets import MNIST


DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = MNIST(root=str(DATA_DIR), train=True, download=True)
    test_dataset = MNIST(root=str(DATA_DIR), train=False, download=True)

    print(f"Downloaded MNIST to: {DATA_DIR}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()
