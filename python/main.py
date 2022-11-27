from models import BrogrammersModel, BrogrammersSequentialModel
from evaluation_and_tracking import IntraEpochMetricsTracker
from utils.augmentations_and_transforms import AddGaussianNoise, CyclicTemporalShift
from datasets import ResnetLogmelDataset, BrogrammersMFCCDataset
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision.models import resnet18
from datetime import datetime
import time
import numpy as np
import pickle
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from itertools import product
import os
import random


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


def get_ids_of(participants_filename):
    # get list of positive, negative and invalid ids from a list of participant instances
    path = os.path.join("data", "Coswara_processed", "pickles", participants_filename)
    with open(path, "rb") as f:
        parts = pickle.load(f)
    ids_pos, ids_neg, ids_invalid = [], [], []
    for part in parts:
        if part.get_label() == 1:
            ids_pos.append(part.id)
        elif part.get_label() == 0:
            ids_neg.append(part.id)
        else:
            ids_invalid.append(part.id)
    return ids_pos, ids_neg, ids_invalid


def randomly_split_list_into_two(input_list, ratio=0.8, random_seed=None):
    input_list_temp = input_list.copy()
    split_index = int(np.floor(len(input_list_temp) * ratio))
    random.Random(random_seed).shuffle(input_list_temp)
    return input_list_temp[:split_index], input_list_temp[split_index:]


def get_datasets(dataset_name, split_ratio=0.8, transform=None, train_augmentation=None, random_seed=None):
    dataset_collection = {
        "brogrammers": {
            "dataset_class": BrogrammersMFCCDataset,
            "participants_file": "participants_validLabelsOnly.pickle",
            "augmented_files": ["participants_oversampledPositives.pickle"]
        },
        "resnet": {
            "dataset_class": ResnetLogmelDataset,
            "participants_file": "2022-11-25-added_logmel224x224_no_augmentations.pickle",
            "augmented_files": ["2022-11-25-added_logmel224x224.pickle"]
            # "augmented_files": None
        }
    }
    dataset_dict = dataset_collection[dataset_name]
    DatasetClass = dataset_dict["dataset_class"]

    # participants_file = "participants_validLabelsOnly.pickle"
    # augmented_participant_files = ["participants_oversampledPositives.pickle"]
    # participants_file = "2022-11-25-added_logmel224x224_no_augmentations.pickle"
    # augmented_participant_files = ["2022-11-25-added_logmel224x224.pickle"]

    pos_ids, neg_ids, invalid_ids = get_ids_of(dataset_dict["participants_file"])
    pos_ids_train, pos_ids_val = randomly_split_list_into_two(pos_ids, ratio=split_ratio, random_seed=random_seed)
    neg_ids_train, neg_ids_val = randomly_split_list_into_two(neg_ids, ratio=split_ratio, random_seed=random_seed)
    train_ids = pos_ids_train + neg_ids_train
    validation_ids = pos_ids_val + neg_ids_val

    # train_augmentations = Compose([AddGaussianNoise(0, 0.15), CyclicTemporalShift()])

    # training_set = ResnetLogmelDataset(user_ids=train_ids, original_files=participants_file,
    #                                    transform=transform, augmented_files=augmented_participant_files,
    #                                    augmentations=train_augmentation, verbose=VERBOSE)
    # validation_set = ResnetLogmelDataset(user_ids=validation_ids,  original_files=participants_file,
    #                                      transform=transform, verbose=VERBOSE)
    #
    training_set = DatasetClass(user_ids=train_ids, original_files=dataset_dict["participants_file"],
                                transform=transform, augmented_files=dataset_dict["augmented_files"],
                                augmentations=train_augmentation, verbose=VERBOSE)
    validation_set = DatasetClass(user_ids=validation_ids, original_files=dataset_dict["participants_file"],
                                  transform=transform, verbose=VERBOSE)

    # training_set = BrogrammersMFCCDataset(user_ids=train_ids, original_files=participants_file,
    #                                       transform=transform, augmented_files=augmented_participant_files,
    #                                       augmentations=train_augmentation, verbose=VERBOSE)
    # validation_set = BrogrammersMFCCDataset(user_ids=validation_ids, original_files=participants_file,
    #                                         transform=transform, verbose=VERBOSE)
    return training_set, validation_set


# def get_data_loaders_deprecated(dataset, percent_train_set=0.8, rand_seed=None):
# n_train_samples = int(percent_train_set * len(data_set))
# n_val_samples = len(data_set) - int(n_train_samples)
# rng = torch.Generator()
# if rand_seed is not None:
#     rng.manual_seed(rand_seed)
# train_set, eval_set = random_split(dataset, [n_train_samples, n_val_samples], generator=rng)
# train = DataLoader(dataset=train_set, batch_size=p.batch_size, shuffle=True, drop_last=True)
# val = DataLoader(dataset=eval_set, batch_size=len(eval_set))
# return train, val


def get_data_loaders(training_set, validation_set):
    train = DataLoader(dataset=training_set, batch_size=p.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(dataset=validation_set, batch_size=len(val_set))
    return train, val


def get_model(model_name, verbose=True, load_from_disc=False):
    # model_dict = {
    #     "brogrammers": BrogrammersModel,
    #     "brogrammers_old": BrogrammersSequentialModel,
    # }
    # my_model = model_dict[model_name]().to(device)
    #
    # # print model summary
    # if verbose:
    #     full_input_shape = [p.batch_size]
    #     for dim in my_model.input_size:
    #         full_input_shape.append(dim)
    #     summary(my_model, tuple(full_input_shape))
    #
    # if load_from_disc:
    #     try:
    #         path = f"data/Coswara_processed/models/{model_name}/model.pth"
    #         my_model.load_state_dict(torch.load(path))
    #         print("model weights loaded from disc")
    #     except FileNotFoundError:
    #         print("no saved model parameters found")
    my_model = resnet18(pretrained=True).to(device)
    for param in my_model.parameters():
        param.requires_grad = False
    n_features = my_model.fc.in_features
    my_model.fc = nn.Linear(n_features, 1)

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
        writer.add_scalar(f"04_true_positives_at_95/{mode}", tpr_at_95, epoch)
        writer.add_scalar(f"05_AUC-precision-recall/{mode}", auc_pr, epoch)


# </editor-fold>

# ###############################################  manual setup  #######################################################
n_epochs = 30

parameters = dict(
    batch_size=[128],
    lr=[1e-3],
    weight_decay=[1e-4],
    noise_sigma=[0],
    cyclic_shift=[True],
    pos_class_weighting=[1]
)
transforms = None
augmentations = Compose([AddGaussianNoise(0, 0.05), CyclicTemporalShift()])

# "brogrammers", "resnet"
MODEL_NAME = "resnet"
RUN_COMMENT = "resnet18_logmel_duplicated_channels"
VERBOSE = True
LOAD_FROM_DISC = False
SAVE_TO_DISC = False
TRACK_METRICS = True

# ############################################ setup ###################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
date = datetime.today().strftime("%Y-%m-%d")
RUN_NAME = f"{date}_{MODEL_NAME}_{RUN_COMMENT}"

# data_set = CustomDataset(ToTensor(), verbose=VERBOSE)
train_set, val_set = get_datasets(MODEL_NAME, split_ratio=0.8, transform=transforms,
                                  train_augmentation=augmentations, random_seed=None)

for p in get_parameter_combinations(parameters):

    # if p.noise_sigma > 0:
    #     train_set.augmentations = Compose([AddGaussianNoise(0, p.noise_sigma), CyclicTemporalShift()])
    # else:
    #     train_set.augmentations = None
    if p.cyclic_shift:
        train_set.augmentations = Compose([AddGaussianNoise(0, p.noise_sigma), CyclicTemporalShift()])
    else:
        train_set.augmentations = Compose([AddGaussianNoise(0, p.noise_sigma)])

    if TRACK_METRICS:
        writer = SummaryWriter(log_dir=f"run/{RUN_NAME}/{p}")

    train_loader, eval_loader = get_data_loaders(train_set, val_set)
    my_cnn = get_model(MODEL_NAME, load_from_disc=LOAD_FROM_DISC, verbose=False)

    optimizer = get_optimizer(MODEL_NAME, load_from_disc=LOAD_FROM_DISC)
    tracker = IntraEpochMetricsTracker()
    # adding a weight to the positive class (which is the underrepresented class --> pas_weight > 1)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p.pos_class_weighting])).to(device)
    # loss_func = nn.BCEWithLogitsLoss().to(device)
    # ################################################ training ########################################################

    epoch_start = time.time()
    for epoch in range(n_epochs):
        tracker.reset()
        # data_set.use_augmentations = True
        for i, batch in enumerate(train_loader):
            train_on_batch(my_cnn, batch, loss_func, optimizer, tracker)
        write_metrics_to_tensorboard(mode="train")

        with torch.no_grad():
            # data_set.use_augmentations = False
            tracker.reset()
            for i, batch in enumerate(eval_loader):
                evaluate_batch(my_cnn, batch, loss_func, tracker)
            write_metrics_to_tensorboard(mode="eval")

    # ##################################################################################################################
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
