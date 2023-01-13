from models import BrogrammersModel, BrogrammersSequentialModel, get_resnet18, get_resnet50
from evaluation_and_tracking import IntraEpochMetricsTracker
from utils.augmentations_and_transforms import AddGaussianNoise, CyclicTemporalShift
from datasets import ResnetLogmelDataset, BrogrammersMFCCDataset, ResnetLogmel3Channels, ResnetLogmel1ChannelBreath,\
    BrogrammersMfccHighRes
from datetime import datetime
import time
import numpy as np
import pickle
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Compose
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from itertools import product
import os
import random
from torch.optim.lr_scheduler import ExponentialLR


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
    # accuracy = get_accuracy(prediction, label)
    my_tracker.add_metrics(loss, label, prediction)

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
    # pred_after_sigmoid = torch.sigmoid(prediction)
    # accuracy = get_accuracy(prediction, label)
    my_tracker.add_metrics(loss, label, prediction)


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
        "15_mfccs": {
            "dataset_class": BrogrammersMFCCDataset,
            "participants_file": "participants_validLabelsOnly.pickle",
            "augmented_files": ["participants_oversampledPositives.pickle"],
            # "augmented_files": None
        },
        "15_mfccs_highRes": {
            # higher time resolution and MFCCS calculated from higher frequency resolution
            # "dataset_class": BrogrammersMfccHighRes,
            "dataset_class": BrogrammersMFCCDataset,
            "participants_file": "2022-12-13_MFCCs_original_highTimeRes.pickle",
            "augmented_files": ["2022-12-13_MFCCs_augmented_highTimeRes.pickle"],
            # "augmented_files": None
        },
        "logmel_1_channel": {
            "dataset_class": ResnetLogmelDataset,
            "participants_file": "2022-11-25-added_logmel224x224_no_augmentations.pickle",
            "augmented_files": ["2022-11-25-added_logmel224x224.pickle"]
            # "augmented_files": None
        },
        "logmel_3_channels_512_2048_8192": {
            "dataset_class": ResnetLogmel3Channels,
            "participants_file": "2022-12-08_logmel_3_channel_noAug_noBadAudio.pickle",
            "augmented_files": ["2022-12-08_logmel_3_channel_augmented_noBadAudio.pickle"]
            # "augmented_files": None
        },
        "logmel_3_channels_1024_2048_4096": {
            "dataset_class": ResnetLogmel3Channels,
            "participants_file": "2022-12-10_logmel_3_channel_noAug_1024x2048x4096.pickle",
            "augmented_files": ["2022-12-10_logmel_3_channel_augmented_1024x2048x4096.pickle"]
            # "augmented_files": None
        },
        "logmel_1_channel_breath": {
            "dataset_class": ResnetLogmel1ChannelBreath,
            "participants_file": "2022-12-11_logmel_1_channel_noAug_heavy_breathing.pickle",
            "augmented_files": ["2022-12-11_logmel_1_channel_augmented_heavy_breathing.pickle"]
            # "augmented_files": None
        }
    }
    dataset_dict = dataset_collection[dataset_name]
    DatasetClass = dataset_dict["dataset_class"]

    pos_ids, neg_ids, invalid_ids = get_ids_of(dataset_dict["participants_file"])
    pos_ids_train, pos_ids_val = randomly_split_list_into_two(pos_ids, ratio=split_ratio, random_seed=random_seed)
    neg_ids_train, neg_ids_val = randomly_split_list_into_two(neg_ids, ratio=split_ratio, random_seed=random_seed)
    train_ids = pos_ids_train + neg_ids_train
    validation_ids = pos_ids_val + neg_ids_val

    training_set = DatasetClass(user_ids=train_ids, original_files=dataset_dict["participants_file"],
                                transform=transform, augmented_files=dataset_dict["augmented_files"],
                                augmentations=train_augmentation, verbose=VERBOSE, mode="train")
    validation_set = DatasetClass(user_ids=validation_ids, original_files=dataset_dict["participants_file"],
                                  transform=transform, verbose=VERBOSE, mode="eval")
    return training_set, validation_set


def get_data_loaders(training_set, validation_set):

    # create weighted random sampler
    label_counts = training_set.label_counts()[1]
    label_weights = np.flip(label_counts/np.sum(label_counts))

    # sample_weights = []
    # for (data, label) in training_set:
    #     sample_weights.append(label_weights[int(label)])

    sample_weights = [label_weights[int(label)] for (data, label) in training_set]

    # might actually set num_samples higher because like this not all samples from the dataset are chosen within 1 epoch
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # create dataloaders
    # train = DataLoader(dataset=training_set, batch_size=p.batch, shuffle=True, drop_last=True)
    train = DataLoader(dataset=training_set, batch_size=p.batch, drop_last=True, sampler=sampler)
    val = DataLoader(dataset=validation_set, batch_size=len(val_set))
    return train, val


def get_model(model_name, verbose=True, load_from_disc=False):
    model_dict = {
        "brogrammers": BrogrammersModel,
        "brogrammers_old": BrogrammersSequentialModel,
        "resnet18": get_resnet18,
        "resnet50": get_resnet50
    }
    my_model = model_dict[model_name]().to(device)

    # # print model summary
    if verbose:
        full_input_shape = [p.batch]
        for dim in my_model.input_size:
            full_input_shape.append(dim)
        summary(my_model, tuple(full_input_shape))

    # TODO move the load from disc functionality inside the "models.py" script
    # if load_from_disc:
    #     try:
    #         path = f"data/Coswara_processed/models/{model_name}/model.pth"
    #         my_model.load_state_dict(torch.load(path))
    #         print("model weights loaded from disc")
    #     except FileNotFoundError:
    #         print("no saved model parameters found")

    return my_model


def get_optimizer(model_name, load_from_disc=False):
    my_optimizer = Adam(my_cnn.parameters(), lr=p.lr, weight_decay=p.wd)
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
        # loss, acc, aucroc, tpr_at_95, auc_pr = tracker.get_epoch_metrics()
        metrics = tracker.get_epoch_metrics()
        writer.add_scalar(f"01_loss/{mode}", metrics["loss"], epoch)
        writer.add_scalar(f"02_accuracy/{mode}", metrics["accuracy"], epoch)
        writer.add_scalar(f"03_AUC-ROC/{mode}", metrics["auc_roc"], epoch)
        writer.add_scalar(f"04_f1_score/{mode}", metrics["f1_score"], epoch)
        writer.add_scalar(f"05_AUC-precision-recall/{mode}", metrics["auc_prec_recall"], epoch)
        writer.add_scalar(f"06_TPR_or_Recall_or_Sensitivity/{mode}", metrics["tpr"], epoch)
        writer.add_scalar(f"07_TrueNegativeRate_or_Specificity/{mode}", metrics["tnr"], epoch)
        writer.add_scalar(f"08_Precision_or_PositivePredictiveValue/{mode}", metrics["precision"], epoch)
        writer.add_scalar(f"09_true_positives_at_95/{mode}", metrics["tpr_at_95"], epoch)


def get_online_augmentations(run_parameters):
    if run_parameters.shift:
        augmentation = Compose([AddGaussianNoise(0, run_parameters.sigma), CyclicTemporalShift()])
    else:
        augmentation = Compose([AddGaussianNoise(0, run_parameters.sigma)])
    return augmentation
# </editor-fold>


# ###############################################  manual setup  #######################################################
random_seeds = [123587955, 99468865, 215674, 3213213211, 55555555,
                66445511337, 316497938271, 161094, 191919191, 101010107]

n_epochs = 40
n_cross_validation_runs = 1

parameters = dict(
    rand=random_seeds[:n_cross_validation_runs],
    batch=[128],
    lr=[1e-3],
    wd=[5e-4],
    sigma=[0],
    shift=[True],
    # weight=[1],
    lr_decay=[0.85, 0.9]
)
# lr_decay = 0.95
transforms = None
augmentations = Compose([AddGaussianNoise(0, 0.05), CyclicTemporalShift()])

# "brogrammers", "resnet18", "resnet50"
MODEL_NAME = "brogrammers"
# logmel_3_channels_512_2048_8192, logmel_3_channels_1024_2048_4096, logmel_1_channel, logmel_1_channel_breath
# 15_mfccs, 15_mfccs_highRes
DATASET_NAME = "15_mfccs_highRes"
RUN_COMMENT = f"tests_mixup"

print(f"Dataset used: {DATASET_NAME}")
print(f"model used: {MODEL_NAME}")
date = datetime.today().strftime("%Y-%m-%d")
RUN_NAME = f"{date}_{MODEL_NAME}_{DATASET_NAME}_{RUN_COMMENT}"
VERBOSE = True
LOAD_FROM_DISC = False
SAVE_TO_DISC = False
TRACK_METRICS = True

# ############################################ setup ###################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

for p in get_parameter_combinations(parameters):

    train_set, val_set = get_datasets(DATASET_NAME, split_ratio=0.8, transform=transforms,
                                      train_augmentation=augmentations, random_seed=p.rand)

    train_set.augmentations = get_online_augmentations(p)
    if TRACK_METRICS:
        writer = SummaryWriter(log_dir=f"run/{RUN_NAME}/{p}")

    train_loader, eval_loader = get_data_loaders(train_set, val_set)
    my_cnn = get_model(MODEL_NAME, load_from_disc=LOAD_FROM_DISC, verbose=False)

    optimizer = get_optimizer(MODEL_NAME, load_from_disc=LOAD_FROM_DISC)
    lr_scheduler = ExponentialLR(optimizer, gamma=p.lr_decay)
    tracker = IntraEpochMetricsTracker()
    # loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p.weight])).to(device)
    loss_func = nn.BCEWithLogitsLoss().to(device)
    # ################################################ training ########################################################

    epoch_start = time.time()
    for epoch in range(n_epochs):


        tracker.reset()
        for i, batch in enumerate(train_loader):
            train_on_batch(my_cnn, batch, loss_func, optimizer, tracker)
        write_metrics_to_tensorboard(mode="train")

        with torch.no_grad():
            tracker.reset()
            for i, batch in enumerate(eval_loader):
                evaluate_batch(my_cnn, batch, loss_func, tracker)
            write_metrics_to_tensorboard(mode="eval")
        lr_scheduler.step()
        print(f"current learning rates: {[round(lr, 8) for lr in lr_scheduler.get_last_lr()]}")
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
