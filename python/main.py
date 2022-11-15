from models import BrogrammersModel, BrogrammersSequentialModel
from evaluation_and_tracking import IntraEpochMetricsTracker
from datasets import CustomDataset
from datetime import datetime
import time
import librosa

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


# <editor-fold desc="function definitions">
class Params:
    def __init__(self, batch_size=256, lr=0.0000001, n_epochs=20, weight_decay=0.01, verbose=True):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(self)

    def __str__(self):
        out = "Parameters for this Run are:\n"
        out += f"Batch Size: {self.batch_size}\n"
        out += f"Learning Rate: {self.lr}\n"
        out += f"Number of Epochs: {self.n_epochs}\n"
        out += f"Weight Decay: {self.weight_decay}\n"
        out += f"Training is done on '{self.device}'\n"
        return out


def train_on_batch(model, current_batch, current_loss_func, current_optimizer, my_tracker):
    model.train()
    input_data, label = current_batch
    input_data, label = input_data.to(p.device), label.to(p.device)
    prediction = torch.squeeze(model(input_data))
    loss = current_loss_func(prediction, label)
    accuracy = get_accuracy(prediction, label)
    my_tracker.add_metrics(loss, accuracy, label, prediction)

    # backpropagation
    current_optimizer.zero_grad()
    loss.backward()
    current_optimizer.step()


def evaluate_batch(model, current_batch, loss_function, my_tracker):
    model.eval()
    input_data, label = current_batch
    input_data, label = input_data.to(p.device), label.to(p.device)
    prediction = torch.squeeze(model(input_data))
    loss = loss_function(prediction, label)
    accuracy = get_accuracy(prediction, label)
    my_tracker.add_metrics(loss, accuracy, label, prediction)


def get_accuracy(predictions, labels, threshold=0.5):
    predicted_labels = predictions > threshold
    labels_bool = labels > threshold
    n_correctly_predicted = torch.sum(predicted_labels == labels_bool)
    return n_correctly_predicted.item() / len(predictions)


def get_data_loaders(dataset, percent_train_set=0.8, rand_seed=None):
    n_train_samples = int(percent_train_set * len(data_set))
    n_val_samples = len(data_set) - int(n_train_samples)
    rng = torch.Generator()
    if rand_seed is not None:
        rng.manual_seed(rand_seed)
    train_set, eval_set = random_split(dataset,[n_train_samples, n_val_samples], generator=rng)
    train = DataLoader(dataset=train_set, batch_size=p.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(dataset=eval_set, batch_size=len(eval_set))
    return train, val


def get_model(model_name, dataset=None, verbose=True, load_from_disc=False):
    model_dict = {
        "brogrammers": BrogrammersModel,
        "brogrammers_old": BrogrammersSequentialModel
    }
    my_model = model_dict[model_name]().to(p.device)

    # print model summary
    if verbose and dataset is not None:
        full_input_shape = [p.batch_size]
        for dim in dataset.get_input_shape():
            full_input_shape.append(dim)
        summary(my_model, tuple(full_input_shape))

    if load_from_disc:
        try:
            path = f"data/Coswara_processed/models/{model_name}/model.pth"
            my_model.load_state_dict(torch.load(path))
            print("model weights loaded from disc")
        except FileNotFoundError:
            print("no saved model parameters found")

    return my_model


def get_optimizer(model_name, verbose=True, load_from_disc=False):
    my_optimizer = Adam(my_cnn.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    if load_from_disc:
        try:
            path = f"data/Coswara_processed/models/{model_name}/optimizer.pickle"
            my_optimizer.load_state_dict(torch.load(path))
            print("optimizer state loaded from disc")
        except FileNotFoundError:
            print("no optimizer state found on disc. Starting from scratch")
    return my_optimizer


def write_metrics_to_tensorboard(mode):
    loss, acc, aucroc, tpr_at_95, auc_pr = tracker.get_epoch_metrics()
    writer.add_scalar(f"{mode}/01_loss", loss, epoch)
    writer.add_scalar(f"{mode}/02_accuracy", acc, epoch)
    writer.add_scalar(f"{mode}/03_AUC-ROC", aucroc, epoch)
    writer.add_scalar(f"{mode}/04_true_positives_at_95", tpr_at_95, epoch)
    writer.add_scalar(f"{mode}/05_AUC-precision-recall", auc_pr, epoch)
# </editor-fold>


VERBOSE = False
LOAD_FROM_DISC = False
SAVE_TO_DISC = False
p = Params(batch_size=64, lr=0.0000001, n_epochs=5, weight_decay=0.01, verbose=VERBOSE)
MODEL_NAME = "brogrammers"
RUN_COMMENT = "test_tensorboard_structure"

date = datetime.today().strftime("%Y-%m-%d")
RUN_NAME = f"{date}_{MODEL_NAME}_{RUN_COMMENT}"

# ############################################ setup ###################################################################
data_set = CustomDataset(ToTensor(), verbose=VERBOSE)
train_loader, eval_loader = get_data_loaders(data_set, percent_train_set=0.8)

learning_rates = [1e-4, 1e-5, 1e-6]
for lr in learning_rates:
    p.lr = lr
    writer = SummaryWriter(log_dir=f"run/{RUN_NAME}/lr={p.lr}")

    my_cnn = get_model(MODEL_NAME, data_set, load_from_disc=LOAD_FROM_DISC, verbose=VERBOSE)
    optimizer = get_optimizer(MODEL_NAME, load_from_disc=LOAD_FROM_DISC, verbose=VERBOSE)
    tracker = IntraEpochMetricsTracker()
    # adding a weight to the positive class (which is the underrepresented class --> pas_weight > 1)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3])).to(p.device)

# ################################################# training ###########################################################
    for epoch in range(p.n_epochs):
        epoch_start = time.time()
        tracker.reset()
        for i, batch in enumerate(train_loader):
            train_on_batch(my_cnn, batch, loss_func, optimizer, tracker)

        write_metrics_to_tensorboard(mode="train")

        with torch.no_grad():
            tracker.reset()
            for i, batch in enumerate(eval_loader):
                evaluate_batch(my_cnn, batch, loss_func, tracker)
            write_metrics_to_tensorboard(mode="eval")

        delta_t = time.time() - epoch_start
        if VERBOSE or True:
            print(f"Epoch #{epoch} took [{int(delta_t // 60)}min {int(delta_t % 60)}s] to calculate")

    writer.close()
    if SAVE_TO_DISC:
        print("saving new model!")
        MODEL_PATH = f"data/Coswara_processed/models/{MODEL_NAME}/model.pth"
        torch.save(my_cnn.state_dict(), MODEL_PATH)
        # optimizer.zero_grad()
        OPTIMIZER_PATH = f"data/Coswara_processed/models/{MODEL_NAME}/optimizer.pickle"
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
