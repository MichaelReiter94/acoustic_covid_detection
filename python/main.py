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
from torchinfo import summary
from models import BrogrammersModel, BrogrammersSequentialModel


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
        input_features = self.participants[idx].heavy_cough.get_MFCCs()
        if self.transform:
            input_features = self.transform(input_features)
        output_label = self.participants[idx].get_label()
        return input_features, torch.tensor(output_label).float()

    def __len__(self):
        return len(self.participants)

    def get_object(self, idx):
        return self.participants[idx]


metadata = pd.read_csv("data/Coswara_processed/reformatted_metadata.csv")
print(metadata.covid_health_status.value_counts())

############################################# dataset setup ############################################################
batch_size = 32
data_set = CustomDataset(transform=ToTensor())
# train_set, test_set = torch.utils.data.random_split(data_set, [10, 10], torch.Generator.manual_seed(42))
# train_set, test_set = torch.utils.data.random_split(data_set, [32, 32])
data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)


################################################ CNN setup #############################################################
device = "cpu"
n_epochs = 5
learning_rate = 0.0001
my_cnn = BrogrammersModel().to(device)
summary(my_cnn, (batch_size, 1, 15, 431))
optimizer = Adam(my_cnn.parameters(), lr=learning_rate)
loss_func = nn.BCELoss()

################################################## training ############################################################
for epoch in range(n_epochs):
    for batch in data_loader:
        input_data, label = batch
        input_data, label = input_data.to(device), label.to(device)
        prediction = torch.squeeze(my_cnn(input_data))
        loss = loss_func(prediction, label)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
