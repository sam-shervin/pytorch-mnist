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
            nn.Linear(28*28, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh(),
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


def train_one_epoch(model, inputs, targets, loss_fn, optimizer):
        # Compute prediction error
    pred = model(inputs)
    loss = loss_fn(pred, targets )

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")


def train(model, inputs, targets, loss_fn, optimizer, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, inputs, targets, loss_fn, optimizer)
        print("---------------------------")
    print("Finished training!")


if __name__ == "__main__":
    # download data
    training_data, _ = download_data()
    print("MNIST data downloaded successfully!")

    # create data loader
    train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
    print("Data loaded successfully!")

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to("cuda"), targets.to("cuda")

    # create model
    model = NeuralNetwork().to("cuda")
    print("Model created successfully!")

    # train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, inputs, targets, loss_fn, optimizer, epochs=1000)
    print("Model trained successfully!")

    # save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved successfully!")

