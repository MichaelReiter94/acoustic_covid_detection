from models import BrogrammersModel, BrogrammersSequentialModel
from evaluation_and_tracking import ModelEvaluator

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchinfo import summary
import os
import pandas as pd
# from participant import Participant
import random

MODEL_NAME = "brogrammers_2022_10_30"
MODEL_PATH =         f"data/Coswara_processed/models/{MODEL_NAME}.pth"
MODEL_TRACKER_PATH = f"data/Coswara_processed/models/{MODEL_NAME}_tracker.pickle"
OPTIMIZER_PATH =     f"data/Coswara_processed/models/{MODEL_NAME}_optimizer.pickle"

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "rb") as f:
        with open("data/Coswara_processed/pickles/participant_objects.pickle", "rb") as f:
            self.participants = pickle.load(f)
        self.drop_invalid_labels()
        self.drop_bad_audio()
        self.labels = np.array([int(participant.get_label()) for participant in self.participants])

    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]

    def drop_bad_audio(self):
        self.participants = [participant for participant in self.participants if
                             participant.meta_data["audio_quality_heavy_cough"] > 0.0]

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

    def label_counts(self):
        return np.unique(self.labels, return_counts=True)


def train_on_batch(model, current_batch, current_loss_func, current_optimizer):
    model.train()
    input_data, label = current_batch
    input_data, label = input_data.to(device), label.to(device)
    prediction = torch.squeeze(model(input_data))
    loss = current_loss_func(prediction, label)

    # backpropagation
    current_optimizer.zero_grad()
    loss.backward()
    current_optimizer.step()
    accuracy = get_accuracy(prediction, label)
    # returns a copy/a float version of the scalar tensor of the loss function
    return float(loss), accuracy


def evaluate_batch(model, batch, loss_func):
    model.eval()
    input_data, label = batch
    input_data, label = input_data.to(device), label.to(device)
    prediction = torch.squeeze(model(input_data))
    loss = loss_func(prediction, label)
    accuracy = get_accuracy(prediction, label)
    # returns a copy/a float version of the scalar tensor of the loss function
    return float(loss), accuracy


def get_accuracy(predictions, labels, threshold=0.5):
    predicted_labels = predictions > threshold
    labels_bool = labels > threshold
    n_correctly_predicted = torch.sum(predicted_labels == labels_bool)
    return n_correctly_predicted.item() / len(predictions)


# metadata = pd.read_csv("data/Coswara_processed/reformatted_metadata.csv")
# print(metadata.covid_health_status.value_counts())

############################################# dataset setup ############################################################
# hyperparameters
batch_size = 32
learning_rate = 0.0001
n_epochs = 1
device = "cpu"

data_set = CustomDataset(transform=ToTensor())
print(f"The length of the dataset = {len(data_set)}")
n_train_samples = int(0.8 * len(data_set))
n_val_samples = len(data_set) - int(n_train_samples)
train_set, eval_set = random_split(data_set, [n_train_samples, n_val_samples],
                                   generator=torch.Generator().manual_seed(42))
# np.unique(data_set.labels[[val_set.indices]], return_counts=True)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, shuffle=True)

################################################ CNN setup #############################################################
my_cnn = BrogrammersModel().to(device)
summary(my_cnn, (batch_size, 1, 15, 259))
try:
    my_cnn.load_state_dict(torch.load(MODEL_PATH))
except FileNotFoundError:
    print("no saved model parameters found")


optimizer = Adam(my_cnn.parameters(), lr=learning_rate)
try:
    optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
except FileNotFoundError:
    print("no optimizer state found")


try:
    with open(MODEL_TRACKER_PATH, "rb") as f:
        tracker = pickle.load(f)
except FileNotFoundError:
    tracker = ModelEvaluator()
tracker.train()

loss_func = nn.BCELoss()
################################################## training ############################################################
for epoch in range(n_epochs):

    print("TRAINING")
    my_cnn.train()
    for i, batch in enumerate(train_loader):
        current_loss, current_accuracy = train_on_batch(my_cnn, batch, loss_func, optimizer)
        print(f"Loss per sample = {current_loss:.3f} -- Accuracy:{current_accuracy * 100.0:.1f}% --  "
              f"iteration {i + 1}/{len(train_loader)}")
        tracker.track_loss(current_loss, mode="train")
        print(my_cnn.dense1.weight[0])

    print("EVALUATION")
    my_cnn.eval()
    for i, batch in enumerate(eval_loader):
        current_loss, current_accuracy = evaluate_batch(my_cnn, batch, loss_func)
        tracker.track_loss(current_loss, mode="eval")
        print(f"Loss per sample = {current_loss:.3f} -- Accuracy:{current_accuracy * 100.0:.1f}% --  "
              f"iteration {i + 1}/{len(eval_loader)}")

    tracker.epoch_has_finished()

    # epoch_loss = tracker.get_loss(granularity="epoch", mode="eval")
    # if epoch_loss[-1] <= min(epoch_loss):
    #     print("saving new model!")
    #     torch.save(my_cnn.state_dict(), MODEL_PATH)
    #     torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

plt.figure()
plt.plot(tracker.get_loss(granularity="epoch", mode="train"))
plt.plot(tracker.get_loss(granularity="epoch", mode="eval"))
plt.legend(["training loss", "evaluation loss"])
plt.show()

with open(MODEL_TRACKER_PATH, "wb") as f:
    pickle.dump(tracker, f)

print("saving new model!")
torch.save(my_cnn.state_dict(), MODEL_PATH)
torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
