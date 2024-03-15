import numpy as numpy
import pandas as pd
import os

for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

train_set = pd.read_csv("kaggle/input/train.csv")
X = train_set.drop('label', axis=1)
y = train_set['label']

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
examples = train_set.sample(9)
labels = examples['label'].values
examples = examples.drop('label', axis=1)
for i in range(1, cols * rows + 1):
    label = labels[i - 1]
    example = examples.iloc[i - 1].values
    example = example.reshape((28, 28))
    figure.add_subplot(rows, cols, i)
    plt.title(label)

    plt.axis("off")
    plt.imshow(example, cmap="gray")
plt.show()

from sklearn.preprocessing import StandardScaler

class Digits(Dataset):
    def __init__(self, path_to_dataset: str, transform=None, has_label=True) -> None:
        self.path = path_to_dataset
        self.transform = transform
        self.has_label = has_label

        dataset = pd.read_csv(path_to_dataset)
        scaler = StandardScaler()
        if has_label:
            self.labels = dataset['label'].values
            self.data = dataset.drop('label', axis=1).values
        else:
            self.data = dataset.values
        self.data = scaler.fit_transform(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        
        if self.has_label:
            return sample, self.labels[index]
        else:
            return sample

class DigitsClassifier(nn.Module):
    def __init__(self, input_channels: int, classes: int):
        super(DigitsClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
            nn.Flatten(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

from torchvision import transforms

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Digits("kaggle/input/train.csv")
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
model = DigitsClassifier(1, 10)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
lossFn = nn.NLLLoss()

# Train
for e in range(50):
    model.train()

    totalTrainLoss = 0
    trainCorrect = 0

    for (x, y) in loader:
        (x, y) = (x.type(torch.float32).to(device), y.to(device))

        pred = model(x)
        loss = lossFn(pred, y)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

# Test on train
example, label = dataset[0]
example = troch.from_numpy(np.array([example])).type(torch.float32).to(device)
model(example).argmax(1).item()

test_dataset = Digits("kaggle/input/test.csv", has_label=False)
example = test_dataset[0]
example = torch.from_numpy(np.array([example])).type(torch.float32).to(device)
model(example).argmax(1).item()

# Plot
figure = plt.figure(figsize=(8,8))
figure.add_subplot(1, 1, 1)
plt.title(label)
plt.axis("off")
test_example = example.cpu().view((28, 28))
plt.imshow(test_example, cmap='gray')

# Getting labels for testset
test_dataset = Digits("kaggle/input/test.csv", has_label=False)
n = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n, shuffle=False)

for examples in test_loader:
    examples = examples.to(device)
    results = model(examples.type(torch.float32))
    values = results.argmax(1)
    result = np.concatenate((np.arange(1, n+1).reshape(-1, 1), values.cpu().numpy().reshape(-1, 1)), axis=1)

df = pd.DataFrame({"ImageId": np.arange(1, n+1), "Label": values.cpu()})
df.to_csv("submission.csv", index=False)
df.sample(5)