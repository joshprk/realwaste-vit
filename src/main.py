#!/usr/bin/python
from torchvision.models import VisionTransformer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import torch
import torchvision

from pathlib import Path
from urllib import request
from zipfile import ZipFile
import os

# Static URL to UCI Machine Learning Repository
DATASET_URL = \
    'https://archive.ics.uci.edu/static/public/908/realwaste.zip'

TEST_DATA_PERCENT = .2
N_EPOCHS = 10
LEARNING_RATE = 0.001

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

if not os.path.exists('realwaste-main'):
    print('dataset missing')
    print("download directly from {}".format(DATASET_URL))
    exit(1)
    """
    zip_path = Path('./realwaste.zip')

    print('dataset directory {DATASET_DIR} does not exist')
    if not zip_path.is_file():
        print('downloading realwaste.zip')
        request.urlretrieve(DATASET_URL, zip_path)

    print('extracting dataset to {DATASET_DIR}')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    """

dataset = ImageFolderWithPaths(
    root='./realwaste-main/RealWaste/', 
    transform=transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor()
        ]),
)
dataset, test_dataset = data.random_split(
    dataset, 
    [1 - TEST_DATA_PERCENT, TEST_DATA_PERCENT]
)

model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path('./model')
if model_path.is_file():
    model = torch.load(model_path, weights_only=False).to(device)
else:
    model = VisionTransformer(
        image_size=500,
        patch_size=20,
        num_layers=8,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=2,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=4)

    n = 0
    for epoch in range(N_EPOCHS):
        print("epoch {}".format(epoch))

        train_loss = 0.0
        for batch in loader:
            n += 1
            print("-> batch {}".format(n))

            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += float(loss.detach().cpu().item() / len(loader))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n = 0

    torch.save(model, './model')
    model = torch.load('./model', weights_only=False)

loader = DataLoader(test_dataset, batch_size=4)

total = 0
correct = 0
with torch.inference_mode():
    for batch in loader:
        x, y, p = batch
        x, y, p = x.to(device), y.to(device), p
        outputs = model(x)

        for n in range(len(outputs)):
            if outputs[n].argmax().item() == y[n].item():
                correct += 1
            total += 1

        print(p)        
        print(y)
        print()
        print(outputs)
        print('-----------------')

print("total correct: {}/{}".format(correct, total))
print("percent: {}".format(correct / total))