import os
import numpy as np
# from participant import Participant
import matplotlib.pyplot as plt
import pickle
import random

import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd

# print(f"number of loaded participants: {len(participants)}")
# sample_participant = random.choice(participants)
# # print(sample_participant.meta_data)
# sample_participant.heavy_cough.play_audio()
# # sample_participant.heavy_cough.show_waveform()
# # plt.show()

metadata = pd.read_csv("data/Coswara_processed/reformatted_metadata.csv")
print(metadata.covid_test_result.value_counts())
print(metadata.covid_health_status.value_counts())


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "rb") as f:
            self.participants = pickle.load(f)

    def __getitem__(self, idx):
        input_data = self.participants[idx].heavy_cough.get_MFCCs()
        if self.transform:
            input_data = self.transform(input_data)

        label = self.participants[idx].metadata.covid_test_result
        if label == "p":
            label = 1
        elif label == "n":
            label = 0
        else:
            label = 0
            # TODO how to handle this? (label == "na" || label == "ut") --> what is "ut" anyways? untested?
            # only 1300 of 2700 have a test result entry.
            # only 900-1000 have an actual result (680 positive, 250 negative?????)

        return input_data, torch.tensor(label)

    def __len__(self):
        return len(self.participants)


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (302, 15, 1)
        n_filters = 64
        placeholder = 10

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(3, 3), padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(2, 2), padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=placeholder),
            # TODO no specification given in paper
            nn.Flatten(),
            nn.Linear(in_features=placeholder, out_features=256, bias=True),
            nn.ReLU(),
            # TODO add kernel, bias and activity regularizers???
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        prediction = self.model(input_data)
        return prediction


batch_size = 32
n_epochs = 5
learning_rate = 0.0001
device = "cpu"

data_set = CustomDataset(transform=ToTensor)
train_set, test_set = torch.utils.data.random_split(data_set, [500, 500], torch.Generator.manual_seed(42))
data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

my_cnn = MyCNN().to(device)
optimizer = torch.optim.Adam(my_cnn.parameters(), lr=learning_rate)
loss_func = nn.BCELoss()

for epoch in range(n_epochs):
    for batch in data_loader:
        input_data, label = batch
        input_data, label = input_data.to(device), label.to(device)
        prediction = my_cnn(input_data)
        loss = loss_func(prediction, label)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()










