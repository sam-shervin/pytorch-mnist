import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_layer(x)
        predictions = self.softmax(logits)
        return predictions
    


def download_data():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data

if __name__ == "__main__":
    # download data
    training_data, _ = download_data()
    print("MNIST data downloaded successfully!")

    # create data loader
    train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
    print("Data loaded successfully!")
