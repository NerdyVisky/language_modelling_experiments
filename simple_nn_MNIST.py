
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, ip, nc):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(ip, 50)
        self.fc2 = nn.Linear(50, nc)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ip = 784
nc = 10
batch_size = 64
lr = 1e-3
epochs = 1

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(ip=ip, nc=nc).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        # print(data.shape)
        data = data.reshape(data.shape[0], -1)
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
            data = data.to(device)
            labels = labels.to(device)
            data = data.reshape(data.shape[0], -1)
            scores = model(data)
            _, predictions = scores.max(1)
            num_truepreds += (predictions == labels).sum()
            num_samples += predictions.size(0)
        print(f'Model Performance: {num_truepreds}/{num_samples} \n')
        print(f'Accuracy: {float(num_truepreds)/float(num_samples)*100:.2f}')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)



