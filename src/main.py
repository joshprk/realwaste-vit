from torchvision.datasets import ImageFolder
from torchvision.models import VisionTransformer
from torchvision import transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torch.utils import data
from torch import Tensor, DeviceObjType
import torch

from typing import Tuple
from pathlib import Path
from urllib import request
from zipfile import ZipFile
import shutil
import os

CLASSES = ["Plastic", "Vegetation"]

DATASET_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
DATASET_TEST_PERCENT = 0.2

EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

ZIP_PATH = Path("./realwaste.zip")
DATASET_PATH = Path("./RealWaste")
MODEL_PATH = Path("./model.pth")


class NamedImageFolder(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        original = super(NamedImageFolder, self).__getitem__(index)
        name = self.imgs[index][0].split("/")[-2:]

        return original + (name,)


def get_datasets() -> Tuple[Subset]:
    download = not DATASET_PATH.exists()

    if not download and set(CLASSES) != set(os.listdir(DATASET_PATH)):
        download = True
        shutil.rmtree(DATASET_PATH)

    if download:
        if not ZIP_PATH.is_file():
            request.urlretrieve(DATASET_URL, ZIP_PATH)

        ZipFile(ZIP_PATH).extractall(".")

        if not DATASET_PATH.exists():
            os.mkdir(DATASET_PATH)

        for folder in os.walk("./realwaste-main/RealWaste/"):
            class_name = folder[0].split("/")[-1:][0]
            if class_name in CLASSES:
                shutil.move(folder[0], DATASET_PATH)

        shutil.rmtree("./realwaste-main/")

    dataset = NamedImageFolder(
        root=DATASET_PATH,
        transform=transforms.Compose(
            [transforms.Resize((500, 500)), transforms.ToTensor()]
        ),
    )

    split = [1 - DATASET_TEST_PERCENT, DATASET_TEST_PERCENT]
    dataset, test_dataset = data.random_split(dataset, split)

    return dataset, test_dataset


def train_transformer(dataset: Subset, device: DeviceObjType) -> VisionTransformer:
    model = VisionTransformer(
        image_size=500,
        patch_size=20,
        num_layers=8,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=2048,
        num_classes=2,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    n = 1
    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))

        train_loss = 0.0
        for batch in loader:
            print("-> batch {}".format(n))
            print("   train_loss {}".format(train_loss), end="\033[F")
            n += int(1)

            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += float(loss.detach().cpu().item() / len(loader))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("\n\n", end="")
    
    torch.save(model, MODEL_PATH)

    return model


def main():
    dataset, test_dataset = get_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    try:
        model = torch.load(MODEL_PATH, weights_only=False).to(device)
    except FileNotFoundError:
        model = train_transformer(dataset, device)

    with torch.inference_mode():
        loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        total = (len(loader) * BATCH_SIZE) - 1
        correct = 0

        for batch in loader:
            x, y, n = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            for n in range(len(y_hat)):
                if y_hat[n].argmax().item() == y[n].item():
                    correct += 1

        print("total correct {}/{}".format(correct, total))
        print("percent: {}".format(correct / total))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Canceled by KeyboardInterrupt')