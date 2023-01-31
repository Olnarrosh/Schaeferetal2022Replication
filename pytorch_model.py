import torch
from torch.utils.data import DataLoader
from custom_dataset_pytorch import CustomEmbeddingDataset
from torch import nn

# dummy-Datensatz for testing purposes

data_train = [
    ["Das ist ein Test Satz", [1, 2, 0, 4, 5, 6, 7, 8, 9], 1],
    ["Dieses Model wird super", [0, 0, 1, 0, 0, 0, 0, 0, 0], 0],
    ["Ich brauche noch einen dritten Satz", [40, 10, 0, 41, 34, 89, 356, 8, 154], 1],
]

data_test = [
    ["Wow, noch mehr Sätze", [4, 5, 0, 7, 8, 9, 10, 11, 12], 1],
    ["Test-Satz nummer 2", [0, 15, 1, 1, 3, 4, 3, 8, 0], 0],
    ["Und Drittens", [1, 6, 0, 4, 5, 6, 7, 8, 9], 1],
]


# define NeuralNetwork class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_size = 9
        self.output_size = 3
        # TODO still need to fix the in_features and out_features
        # integers for nn.Linear functions
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU(),
            nn.Linear(self.output_size, self.output_size),
            nn.ReLU(),
            nn.Linear(self.output_size, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"X: {X}")
        print(f"y: {y}")
        X, y = X.to(device), y.to(device)
        print(f"X: {X}")
        print(f"y: {y}")

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    train_dataset = CustomEmbeddingDataset(data_train)
    test_dataset = CustomEmbeddingDataset(data_test)

    # use DataLoaders to read in CustomDataset objects
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # define Device that is used to compute
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train(train_dataloader, model, loss_fn, optimizer, device)

    test(test_dataloader, model, loss_fn, device)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")


# test NeuralNetwork
if __name__ == "__main__":
    main()


"""
	X = torch.rand(9, device=device)
	logits = model(X)
	pred_probab = nn.Softmax()(logits)
	y_pred = pred_probab.argmax(1)
	print(f"Predicted class: {y_pred}")
"""
