import torch
from io import open
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import string
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CountryWiseNames(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.categories= []
        self.category_lines = {}
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        self.n_per_category = {}
        self._load_data()
    def _load_data(self):
        for filename in os.listdir(self.root_dir):
            category = os.path.splitext(filename)[0]
            self.categories.append(category)
            lines = self._read_lines(os.path.join(self.root_dir, filename))
            self.category_lines[category] = lines
            self.n_per_category[category] = len(lines)

    def _read_lines(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
        return [self._unicode_to_ascii(line) for line in lines]

    def _unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in self.all_letters)

    def __len__(self):
        return sum(len(lines) for lines in self.category_lines.values())

    def __getitem__(self, index):
        cfreq = 0
        for cat_name, freq in self.n_per_category.items():
            if(index > cfreq and index < (cfreq + freq)):
                break
            cfreq += freq
        category_index = 0 if cfreq == 0 else (index // cfreq)
        category = self.categories[category_index]
        line = self.category_lines[category][index if cfreq == 0 else (index % cfreq)]
        category_tensor = torch.tensor([self.categories.index(category)], dtype=torch.long)
        line_tensor = self._line_to_tensor(line)
        # category_tensor = torch.zeros(3, 3)
        # line_tensor = torch.zeros(3, 3)
        return category_tensor, line_tensor

    def _line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for i, letter in enumerate(line):
            tensor[i][0][self._letter_to_index(letter)] = 1
        return tensor

    def _letter_to_index(self, letter):
        return self.all_letters.find(letter)

input_size = 28
sequence_len = 28
hidden_size = 256
num_layers = 2
num_classes = 10
batch_size = 64
lr = 1e-3
epochs = 2
dataset = CountryWiseNames(root_dir='data/names/')
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
for i, batch in enumerate(dataloader):
    print(i, batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

