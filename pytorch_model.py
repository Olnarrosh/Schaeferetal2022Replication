import torch
from torch.utils.data import DataLoader
from custom_dataset_pytorch import CustomEmbeddingDataset
from torch import nn
from sklearn.metrics import f1_score

# dummy-Datensatz for testing purposes

data_train = [
    ["Das ist ein Test Satz", [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1.0],
    ["Dieses Model wird super", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0],
    ["Ich brauche noch einen dritten Satz", [40.0, 10.0, 0.0, 41.0, 34.0, 89.0, 356.0, 8.0, 154.0], 1.0],
]

data_test = [
    ["Wow, noch mehr Sätze", [4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 1.0],
    ["Test-Satz nummer 2", [0.0, 15.0, 1.0, 1.0, 3.0, 4.0, 3.0, 8.0, 0.0], 0.0],
    ["Und Drittens", [1.0, 6.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1.0],
    ["Das ist ein Test Satz", [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1.0],
    ["Ich brauche noch einen dritten Satz", [40.0, 10.0, 0.0, 41.0, 34.0, 89.0, 356.0, 8.0, 154.0], 1.0]
]


# define NeuralNetwork class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=9, hidden_size=3):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # TODO still need to fix the in_features and out_features
        # integers for nn.Linear functions
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )

    def forward(self, x):
        # breakpoint()
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch.float)
        y = y.to(torch.float)
        target_y = torch.tensor(y, dtype=torch.long, device=device)
        X, y = X.to(device), y.to(device)

        # Compute prediction error

        pred = model(X)
        loss = loss_fn(pred, target_y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    pred_list = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, test_correct, counter = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            print(f"this is iteration number: {counter}")
            X = X.to(torch.float)
            y = y.to(torch.float)
            X, y = X.to(device), y.to(device)
            target_y = torch.tensor(y, dtype=torch.long, device=device)
            pred = model(X)
            preds = torch.argmax(pred,1)
            test_loss += loss_fn(pred, target_y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_correct += (pred.argmax(1) == y)
            
            if pred.argmax(1) == y:
                pred_list.append(1)
                print("prediction was right")
            else:
                pred_list.append(0)
                print("prediction was wrong")
            counter += 1
            
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    print(f"correct: {correct}\ntest_correct: {test_correct}\npred: {pred}\npreds: {preds}\npred_list: {pred_list}\ncounter: {counter}\n")


def make_predictions(dataloader, model, device):
    pred_list = []
    size = len(dataloader.dataset)
    model.eval()
    correct, test_correct, counter = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            print(f"this is iteration number: {counter}")
            X = X.to(torch.float)
            y = y.to(torch.float)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_correct += (pred.argmax(1) == y)
            
            if pred.argmax(1) == y:
                pred_list.append(1)
                print("prediction was right")
            else:
                pred_list.append(0)
                print("prediction was wrong")
            counter += 1
            
    correct /= size
    #print(
    #    f"Accuracy: {correct}\ntest_correct: {test_correct}\npred: {pred}\npred_list: {pred_list}\ncounter: {counter}\n"
    #    )
    return pred_list

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
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


def train_model(train_data: list, hidden_size=256, num_epochs=5):
    train_dataset = CustomEmbeddingDataset(train_data)
    train_dataloader = DataLoader(train_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    input_size = len(train_data[0][1])
    hidden_size = 256
    
    model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size).to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = num_epochs
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        print("Done!")
    
    return model            


# test NeuralNetwork
if __name__ == "__main__":
    main()

