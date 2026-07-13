from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST


DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    dataset = MNIST(root=str(DATA_DIR), train=True, download=False)
    image, label = dataset[0]

    print(f"Label: {label}")
    plt.imshow(image, cmap="gray")
    plt.title(f"MNIST train[0] label={label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()