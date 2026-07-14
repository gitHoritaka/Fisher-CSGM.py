from pathlib import Path
import os

from torchvision.datasets import FashionMNIST


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt


def main() -> None:
    dataset = FashionMNIST(root=str(DATA_DIR), train=True, download=False)
    image, label = dataset[0]
    class_name = dataset.classes[label]

    print(f"Label: {label} ({class_name})")
    plt.imshow(image, cmap="gray")
    plt.title(f"Fashion-MNIST train[0] label={label} ({class_name})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
