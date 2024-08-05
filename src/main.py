#!/usr/bin/python
from torchvision.datasets import ImageFolder
from torchvision.models import VisionTransformer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import torch

from pathlib import Path
from urllib import request
from zipfile import ZipFile
import os

# Static URL to UCI Machine Learning Repository
DATASET_URL = \
    'https://archive.ics.uci.edu/static/public/908/realwaste.zip'

TEST_DATA_PERCENT = .2
N_EPOCHS = 5
LEARNING_RATE = 0.005

if not os.path.exists('realwaste-main'):
    zip_path = Path('./realwaste.zip')

    print('dataset directory {DATASET_DIR} does not exist')
    if not zip_path.is_file():
        print('downloading realwaste.zip')
        request.urlretrieve(DATASET_URL, zip_path)

    print('extracting dataset to {DATASET_DIR}')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

dataset = ImageFolder(
    root='./realwaste-main/RealWaste/', 
    transform=Compose([ToTensor()])
)
dataset, test_dataset = data.random_split(
    dataset, 
    [1 - TEST_DATA_PERCENT, TEST_DATA_PERCENT]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = VisionTransformer(
    image_size=524,
    patch_size=4,
    num_layers=4,
    num_heads=2,
    hidden_dim=4,
    mlp_dim=4,
    num_classes=9,
).to(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()
loader = DataLoader(dataset, batch_size=1)

n = 0
for epoch in range(N_EPOCHS):
    print("epoch {}".format(epoch))

    train_loss = 0.0
    for batch in loader:
        n += 1
        print("-> batch {}".format(n))

        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += float(loss.detach().cpu().item() / len(loader))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del x
        del y
        del y_hat
        del loss
        torch.cuda.empty_cache()

    n = 0

torch.save(model, './model')