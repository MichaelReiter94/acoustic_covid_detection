from models import BrogrammersModel, BrogrammersSequentialModel
from evaluation_and_tracking import ModelEvaluator, IntraEpochMetricsTracker
from datetime import datetime
import time
import librosa

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
from torch.utils.tensorboard import SummaryWriter

RUN_SUBFOLDER = "fixed_dropout_lr=0.00001_weightDecay=0,01_batchSize=64"
VERBOSE = False
LOAD_FROM_DISC = False

MODEL_NAME = "brogrammers_2022_10_30"
MODEL_PATH = f"data/Coswara_processed/models/{MODEL_NAME}.pth"
MODEL_TRACKER_PATH = f"data/Coswara_processed/models/{MODEL_NAME}_tracker.pickle"
OPTIMIZER_PATH = f"data/Coswara_processed/models/{MODEL_NAME}_optimizer.pickle"


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "rb") as f:
        with open("data/Coswara_processed/pickles/participant_objects.pickle", "rb") as f:
            self.participants = pickle.load(f)
        self.drop_invalid_labels()
        # self.drop_bad_audio()
        self.labels = np.array([int(participant.get_label()) for participant in self.participants])
        self.mean_mfccs = np.expand_dims(np.load("Data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy"), axis=1)
        self.std_mfccs = np.expand_dims(np.load("Data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy"), axis=1)

    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]

    def drop_bad_audio(self):
        self.participants = [participant for participant in self.participants if
                             participant.meta_data["audio_quality_heavy_cough"] > 0.0]

    def __getitem__(self, idx):
        input_features = self.z_normalize(self.participants[idx].heavy_cough.MFCCs)
        if self.transform:
            input_features = self.transform(input_features)
        output_label = self.participants[idx].get_label()
        return input_features, torch.tensor(output_label).float()

    def __len__(self):
        return len(self.participants)

    def get_object(self, idx):
        return self.participants[idx]

    def z_normalize(self, mfccs):
        return (mfccs - self.mean_mfccs) / self.std_mfccs

    def label_counts(self):
        return np.unique(self.labels, return_counts=True)


def train_on_batch(model, current_batch, current_loss_func, current_optimizer, tracker):
    model.train()
    input_data, label = current_batch
    input_data, label = input_data.to(device), label.to(device)
    prediction = torch.squeeze(model(input_data))
    loss = current_loss_func(prediction, label)
    accuracy = get_accuracy(prediction, label)
    tracker.add_metrics(loss, accuracy, label, prediction)

    # backpropagation
    current_optimizer.zero_grad()
    loss.backward()
    current_optimizer.step()
    # returns a copy/a float version of the scalar tensor of the loss function
    # return float(loss), accuracy


def evaluate_batch(model, batch, loss_func, tracker):
    model.eval()
    input_data, label = batch
    input_data, label = input_data.to(device), label.to(device)
    prediction = torch.squeeze(model(input_data))
    loss = loss_func(prediction, label)
    accuracy = get_accuracy(prediction, label)
    tracker.add_metrics(loss, accuracy, label, prediction)

    # returns a copy/a float version of the scalar tensor of the loss function
    # return float(loss), accuracy


def get_accuracy(predictions, labels, threshold=0.5):
    predicted_labels = predictions > threshold
    labels_bool = labels > threshold
    n_correctly_predicted = torch.sum(predicted_labels == labels_bool)
    return n_correctly_predicted.item() / len(predictions)


# metadata = pd.read_csv("data/Coswara_processed/reformatted_metadata.csv")
# print(metadata.covid_health_status.value_counts())

# ############################################ dataset setup ###########################################################
# hyper parameters
batch_size = 64
learning_rate = 0.00001
n_epochs = 50
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

data_set = CustomDataset(ToTensor())
print(f"The length of the dataset = {len(data_set)}")
n_train_samples = int(0.8 * len(data_set))
n_val_samples = len(data_set) - int(n_train_samples)
train_set, eval_set = random_split(data_set, [n_train_samples, n_val_samples],
                                   generator=torch.Generator().manual_seed(42))
# np.unique(data_set.labels[[val_set.indices]], return_counts=True)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# TODO check if batch_size=len(eval_set) works fine
eval_loader = DataLoader(dataset=eval_set, batch_size=len(eval_set), shuffle=True)

# ############################################### CNN setup ############################################################
my_cnn = BrogrammersModel().to(device)
summary(my_cnn, (batch_size, 1, 15, 259))

optimizer = Adam(my_cnn.parameters(), lr=learning_rate, weight_decay=0.01)
if LOAD_FROM_DISC:
    try:
        my_cnn.load_state_dict(torch.load(MODEL_PATH))
        print("model weights loaded from disc")
    except FileNotFoundError:
        print("no saved model parameters found")

    try:
        optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
        print("optimizer state loaded from disc")
    except FileNotFoundError:
        print("no optimizer state found")

#
# try:
#     with open(MODEL_TRACKER_PATH, "rb") as f:
#         tracker = pickle.load(f)
# except FileNotFoundError:
#     tracker = ModelEvaluator()
# tracker.train()
tracker = IntraEpochMetricsTracker()
# loss_func = nn.BCELoss()
# adding a weight to the positive class (which is the underrepresented class --> pas_weight > 1)
# loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
# TODO check if this works or is necessary
loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3])).to(device)

# ############################################## Tensorboard Tracker ###################################################
date = datetime.today().strftime("%Y-%m-%d")
writer = SummaryWriter(log_dir=f"run/{RUN_SUBFOLDER}-{date}")

# ################################################# training ###########################################################
for epoch in range(n_epochs):
    epoch_start = time.time()
    # print(f"training epoch")
    my_cnn.train()
    tracker.reset()
    for i, batch in enumerate(train_loader):
        train_on_batch(my_cnn, batch, loss_func, optimizer, tracker)
        if VERBOSE:
            print(f"Loss per sample = {tracker.loss[-1]:.3f} -- Accuracy:{tracker.accuracy[-1] * 100.0:.1f}% --  "
                  f"iteration {i + 1}/{len(train_loader)}")
    loss_t, acc_t, aucroc_t, tpr_at_95_t, auc_pr_t = tracker.get_epoch_metrics()

    # print("EVALUATION")
    with torch.no_grad():
        my_cnn.eval()
        tracker.reset()
        for i, batch in enumerate(eval_loader):
            evaluate_batch(my_cnn, batch, loss_func, tracker)
            if VERBOSE:
                print(f"Loss per sample = {tracker.loss[-1] :.3f} -- Accuracy:{tracker.accuracy[-1] * 100.0:.1f}% --  "
                      f"iteration {i + 1}/{len(eval_loader)}")
        loss_v, acc_v, aucroc_v, tpr_at_95_v, auc_pr_v = tracker.get_epoch_metrics()

        writer.add_scalars("01_loss", {"train": loss_t, "validation": loss_v}, epoch)
        writer.add_scalars("02_accuracy", {"train": acc_t, "validation": acc_v}, epoch)
        writer.add_scalars("03_auc_roc", {"train": aucroc_t, "validation": aucroc_v}, epoch)
        writer.add_scalars("04_tpr_at_95", {"train": tpr_at_95_t, "validation": tpr_at_95_v}, epoch)
        writer.add_scalars("05_auc_precision_recall", {"train": auc_pr_t, "validation": auc_pr_v}, epoch)

    print(f"Training + Validation epoch #{epoch} took {((time.time() - epoch_start) / 60):.2f} minutes to calculate")

    # if epoch_loss[-1] <= min(epoch_loss):
    #     print("saving new model!")
    #     torch.save(my_cnn.state_dict(), MODEL_PATH)
    #     torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

writer.close()
# print("saving new model!")
# torch.save(my_cnn.state_dict(), MODEL_PATH)
# torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
