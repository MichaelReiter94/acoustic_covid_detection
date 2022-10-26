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
# from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd

# print(f"number of loaded participants: {len(participants)}")
# sample_participant = random.choice(participants)
# # print(sample_participant.meta_data)
# sample_participant.heavy_cough.play_audio()
# # sample_participant.heavy_cough.show_waveform()
# # plt.show()

metadata = pd.read_csv("data/Coswara_processed/reformatted_metadata.csv")
print(metadata.covid_health_status.value_counts())


# # there are some sort of labels included in the coswara GitHub page but the labels do not correlate with the #
# "covid_health_status" (There are two labels: "1" and "2" and there are "healthy" participants in both label
# categories labelsPath_original_coswara =
# "D:/Archiv/Studium/Master/6.-Semester/Masters_Thesis/Git/acoustic_covid_detection" \
# "/python/data/Coswara-Data/technical_validation/data/cough-heavy/all " with open(labelsPath_original_coswara) as f:
# lines = f.readlines() lines = [line.strip().split() for line in lines] ids = [line[0] for line in lines] labels = [
# int(line[1]) for line in lines]



class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "rb") as f:
        with open("data/Coswara_processed/pickles/participant_objects.pickle", "rb") as f:
            self.participants = pickle.load(f)
        self.drop_invalid_labels()

    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]

    def __getitem__(self, idx):
        input_data = self.participants[idx].heavy_cough.get_MFCCs()
        if self.transform:
            input_data = self.transform(input_data)
        label = self.participants[idx].get_label()
        return input_data, torch.tensor(label)

    def __len__(self):
        return len(self.participants)

    def get_object(self, idx):
        return self.participants[idx]



class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_size = (302, 15, 1)
        self.input_size = (431, 15, 1)
        n_filters = 64
        placeholder = 10

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=n_filters),
            # TODO no specification given in paper
            nn.Flatten(start_dim=1),
            # we start with 431x15
            # conv2D    with stride 1 and kernel size 3 --> 429x13 (x64 channels)
            # maxppol2D with stride 1 and kernel size 2 --> 428x12 (x64 channels)
            # Conv2D    with stride 1 and kernel size 2 --> 427x11 (x64 channels) = 300608

            nn.Linear(in_features=427*11*n_filters, out_features=256, bias=True),
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


# <editor-fold desc="NN set-up">
batch_size = 32
n_epochs = 5
learning_rate = 0.0001
device = "cpu"

data_set = CustomDataset(transform=ToTensor())
# train_set, test_set = torch.utils.data.random_split(data_set, [10, 10], torch.Generator.manual_seed(42))
# train_set, test_set = torch.utils.data.random_split(data_set, [32, 32])
data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)

my_cnn = MyCNN().to(device)
optimizer = Adam(my_cnn.parameters(), lr=learning_rate)
loss_func = nn.BCELoss()
# </editor-fold>


# <editor-fold desc="Section">
for epoch in range(n_epochs):
    for batch in data_loader:
        input_data, label = batch
        input_data, label = input_data.to(device), label.to(device, dtype=float)
        prediction = torch.squeeze(my_cnn(input_data)).to(float)
        loss = loss_func(prediction, label)
        # TODO why do I need to cast both to float type??? can it work in a different way too?
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# </editor-fold>
