from models import BrogrammersModel, BrogrammersSequentialModel
from evaluation_and_tracking import IntraEpochMetricsTracker
from utils.augmentations_and_transforms import AddGaussianNoise, CyclicTemporalShift
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
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import random_split
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from itertools import product

# <editor-fold desc="function definitions">


def get_parameter_combinations(param_dict):
    Run = namedtuple("Run", param_dict.keys())
    runs = []
    for run_combination in product(*param_dict.values()):
        runs.append(Run(*run_combination))
    return runs


def train_on_batch(model, current_batch, current_loss_func, current_optimizer, my_tracker):
    model.train()
    input_data, label = current_batch
    input_data, label = input_data.to(device), label.to(device)
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
    input_data, label = input_data.to(device), label.to(device)
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
    train_set, eval_set = random_split(dataset, [n_train_samples, n_val_samples], generator=rng)
    train = DataLoader(dataset=train_set, batch_size=p.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(dataset=eval_set, batch_size=len(eval_set))
    return train, val


def get_model(model_name, dataset=None, verbose=True, load_from_disc=False):
    model_dict = {
        "brogrammers": BrogrammersModel,
        "brogrammers_old": BrogrammersSequentialModel
    }
    my_model = model_dict[model_name]().to(device)

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


def get_optimizer(model_name, load_from_disc=False):
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
    if TRACK_METRICS:
        loss, acc, aucroc, tpr_at_95, auc_pr = tracker.get_epoch_metrics()
        writer.add_scalar(f"01_loss/{mode}", loss, epoch)
        writer.add_scalar(f"02_accuracy/{mode}", acc, epoch)
        writer.add_scalar(f"03_AUC-ROC/{mode}", aucroc, epoch)
        writer.add_scalar(f"{mode}/04_true_positives_at_95", tpr_at_95, epoch)
        writer.add_scalar(f"{mode}/05_AUC-precision-recall", auc_pr, epoch)


# </editor-fold>

# ###############################################  manual setup  #######################################################
parameters = dict(
    batch_size=[128],
    lr=[1e-5+1e-8],
    n_epochs=[100],
    weight_decay=[1e-5])


MODEL_NAME = "brogrammers"
RUN_COMMENT = "testing_basic_augmentations"
VERBOSE = True
LOAD_FROM_DISC = False
SAVE_TO_DISC = False
TRACK_METRICS = True

# ############################################ setup ###################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
date = datetime.today().strftime("%Y-%m-%d")
RUN_NAME = f"{date}_{MODEL_NAME}_{RUN_COMMENT}"
transforms = Compose([
    ToTensor(),
    AddGaussianNoise(0, 0.1),
    CyclicTemporalShift()
])
data_set = CustomDataset(ToTensor(), verbose=VERBOSE)

for p in get_parameter_combinations(parameters):
    # p.lr = lr
    train_loader, eval_loader = get_data_loaders(data_set, percent_train_set=0.8)
    if TRACK_METRICS:
        writer = SummaryWriter(log_dir=f"run/{RUN_NAME}/{p}")
    my_cnn = get_model(MODEL_NAME, data_set, load_from_disc=LOAD_FROM_DISC, verbose=VERBOSE)
    optimizer = get_optimizer(MODEL_NAME, load_from_disc=LOAD_FROM_DISC)
    tracker = IntraEpochMetricsTracker()
    # adding a weight to the positive class (which is the underrepresented class --> pas_weight > 1)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3])).to(device)
# ################################################# training ###########################################################
    epoch_start = time.time()
    for epoch in range(p.n_epochs):
        tracker.reset()
        for i, batch in enumerate(train_loader):
            train_on_batch(my_cnn, batch, loss_func, optimizer, tracker)
        write_metrics_to_tensorboard(mode="train")

        with torch.no_grad():
            tracker.reset()
            for i, batch in enumerate(eval_loader):
                evaluate_batch(my_cnn, batch, loss_func, tracker)
            write_metrics_to_tensorboard(mode="eval")

    if VERBOSE or True:
        delta_t = time.time() - epoch_start
        print(f"Run {p} took [{int(delta_t // 60)}min {int(delta_t % 60)}s] to calculate")

    if TRACK_METRICS:
        writer.close()
    if SAVE_TO_DISC:
        print("saving new model!")
        MODEL_PATH = f"data/Coswara_processed/models/{MODEL_NAME}/model.pth"
        torch.save(my_cnn.state_dict(), MODEL_PATH)
        # optimizer.zero_grad()
        OPTIMIZER_PATH = f"data/Coswara_processed/models/{MODEL_NAME}/optimizer.pickle"
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
