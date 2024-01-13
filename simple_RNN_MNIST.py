
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_len = 28
hidden_size = 256
num_layers = 2
num_classes = 10
batch_size = 64
lr = 1e-3
epochs = 2
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_len, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_len, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for idx, (data, labels) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        labels = labels.to(device)
        #forward pass
        scores = model(data)
        loss = criterion(scores, labels)
        #backprop
        optimizer.zero_grad()
        loss.backward()
        #adamstep
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train data...")
    else:
        print("Checking accuracy on test data...")
    num_truepreds = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device).squeeze(1)
            labels = labels.to(device)
            scores = model(data)
            _, predictions = scores.max(1)
            num_truepreds += (predictions == labels).sum()
            num_samples += predictions.size(0)
        print(f'Model Performance: {num_truepreds}/{num_samples} \n')
        print(f'Accuracy: {float(num_truepreds)/float(num_samples)*100:.2f}')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)



