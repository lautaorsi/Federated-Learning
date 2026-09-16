"""pytorchexample: A Flower / PyTorch app.

Model definition and local train/test steps. Dataset loading and client
partitioning live in `dataset.py`, a sibling module in this same package.
It's an identical copy of the one used by the other pytorchexample_*
strategy apps in this repo (pytorchexample_fedprox/, ...) -- if you change
the split ratios, Dirichlet alpha, or partitioning logic, remember to copy
the change into each strategy's dataset.py to keep them in sync.
"""

import torch
import torch.nn as nn

from dataset import get_centralized_dataloader, get_client_dataloaders


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Feature Extraction
            nn.Conv2d(3, 8, (7, 7), padding="same", stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 16, (5, 5), padding="same", stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, (3, 3), padding="same", stride=1),
            nn.ReLU(),
            # Classifier
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 128),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Thin wrappers kept so client_app.py / server_app.py's existing imports
# (`from fedadam.task import load_data, load_centralized_dataset`)
# keep working unchanged.
# ---------------------------------------------------------------------------


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    return get_client_dataloaders(partition_id, num_partitions, batch_size)


def load_centralized_dataset():
    return get_centralized_dataloader()


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.NLLLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
