import pandas as pd
from audio_processing import FeatureSet
from models import BrogrammersModel, BrogrammersSequentialModel, get_resnet18, get_resnet50, BrogrammersMIL, \
    Resnet18MIL, PredLevelMIL
from evaluation_and_tracking import IntraEpochMetricsTracker
from utils.augmentations_and_transforms import AddGaussianNoise, CyclicTemporalShift
from datasets import ResnetLogmelDataset, BrogrammersMFCCDataset, MultipleInstanceLearningMFCC, MILResnet
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
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import tkinter as tk
from tkinter.messagebox import askyesno
from torch import cuda

dataset_collection = {
    "combined_breaths_12s_FFT4096_fmax5500_50mfccs":{
        "dataset_class": MultipleInstanceLearningMFCC,
        "participants_file": "2023_03_12_mfcc_combined_breaths_12s_FFT4096_fmax5500_50mfccs.pickle",
        "augmented_files": [".pickle"]
    },

    "mfcc_vowel_e_6s_FFT2048_fmax5500": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_03_11_mfcc_vowel-e_6s_FFT2048_fmax5500.pickle",
        "augmented_files": [".pickle"]
    },
    "mfcc_vowels_combined_6s_FFT2048_fmax5500": {
        "dataset_class": MultipleInstanceLearningMFCC,
        "participants_file": "2023_03_11_mfcc_combined_vowels_6s_FFT2048_fmax5500.pickle",
        "augmented_files": ["2023_03_11_mfcc_combined_vowels_6s_FFT2048_fmax5500_x1x7augmented.pickle"]
    },
    "mfcc_vowel_a_6s_FFT2048_fmax5500": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_03_10_mfcc_vowel-a_6s_FFT2048_fmax5500.pickle",
        "augmented_files": [".pickle"]
    },
    "combined_breaths_6s_2048fft_fmax5500": {
        "dataset_class": MultipleInstanceLearningMFCC,
        "participants_file": "2023_03_09_mfcc_combined_breaths_6s_FFT2048_fmax5500.pickle",
        "augmented_files": ["2023_03_10_mfcc_combined_breaths_6s_FFT2048_fmax5500_x1x5augmented.pickle"]
    },
    "combined_breaths_6s_4096fft": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_03_09_mfcc_combined_breaths_6s_FFT4096.pickle",
        "augmented_files": ["2023_03_09_mfcc_combined_breaths_6s_FFT4096_x1x5augmented.pickle"]
    },
    "combined_breaths_6s_2048fft": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_03_08_mfcc_combined_breaths_6s_FFT2048.pickle",
        "augmented_files": ["2023_03_09_mfcc_combined_breaths_6s_FFT2048_x1x5augmented.pickle"]
    },
    "combined_breaths_3s_2048fft": {
        "dataset_class": MultipleInstanceLearningMFCC,
        "participants_file": "2023_03_07_mfcc_combined_breaths_3s_long_FFTwindow.pickle",
        "augmented_files": ["2023_03_07_mfcc_combined_breaths_3s_long_FFTwindow_x1x4augmented.pickle",
                            "2023_03_08_mfcc_combined_breaths_3s_long_FFTwindow_x1x4_v2augmented.pickle"]
    },
    "resnet_mil_combined_cough": {
        "dataset_class": MILResnet,
        "participants_file": "2023_02_25_logmel_combined_coughs_3s.pickle",
        # "augmented_files":  ["2023_02_26_logmel_combined_coughs_3s_7xaugmented.pickle"]
        "augmented_files": ["2023_02_26_logmel_combined_coughs_3s_7xaugmented.pickle",
                            "2023_02_27_logmel_combined_coughs_3s_augmented_x2x2augmented.pickle"]
    },
    "mfcc_mil_combined_cough": {
        "dataset_class": MultipleInstanceLearningMFCC,
        "participants_file": "2023_02_23_mfcc_combined_coughs_3s.pickle",
        # "augmented_files": ["2023_02_23_mfcc_combined_coughs_3s_7xaugmented.pickle"]
        # "augmented_files": ["2023_02_26_mfcc_combined_coughs_3s_x2x2augmented.pickle"]
        "augmented_files": ["2023_02_23_mfcc_combined_coughs_3s_7xaugmented.pickle",
                            "2023_02_26_mfcc_combined_coughs_3s_x2x2augmented.pickle"]
    },
    "mfccs_3s_breathing_deep": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_02_21_mfcc_breathing-deep_3s_22kHz.pickle",
        "augmented_files": ["2023_02_21_mfcc_breathing-deep_3s_22kHz_augmented.pickle"]
    },
    "logmel_3s_combined_coughs": {
        "dataset_class": ResnetLogmelDataset,
        "participants_file": "2023_02_25_logmel_combined_coughs_3s.pickle",
        "augmented_files": ["2023_02_26_logmel_combined_coughs_3s_7xaugmented.pickle",
                            "2023_02_27_logmel_combined_coughs_3s_augmented_x2x2augmented.pickle"]
        # "augmented_files": ["2023_02_27_logmel_combined_coughs_3s_augmented_x2x2augmented.pickle"]


    },
    "mfccs_3s_combined_coughs": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_02_23_mfcc_combined_coughs_3s.pickle",
        "augmented_files": ["2023_02_23_mfcc_combined_coughs_3s_7xaugmented.pickle",
                            "2023_02_26_mfcc_combined_coughs_3s_x2x2augmented.pickle"]
        # "augmented_files": ["2023_02_23_mfcc_combined_coughs_3s_7xaugmented.pickle"]
    },
    "15_mfccs_highres_new": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_02_21_mfcc_cough-heavy_3s_22kHz.pickle",
        "augmented_files": ["2023_02_21_mfcc_cough-heavy_3s_22kHz_augmented.pickle"],
        # "augmented_files": None
    },
    "brogrammers_new": {
        "dataset_class": BrogrammersMFCCDataset,
        "participants_file": "2023_02_20_brogrammers_settings_new.pickle",
        "augmented_files": ["2023_02_20_brogrammers_settings_new_augmented.pickle"],
        # "augmented_files": None
    },
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
        "participants_file": "2023_02_20_logmel_cough_22kHz_new.pickle",
        "augmented_files": ["2023_02_21_logmel_cough_22kHz_new_augmented.pickle"]
        # "augmented_files": None
    },
    # "logmel_3_channels_512_2048_8192": {
    #     "dataset_class": ResnetLogmel3Channels,
    #     "participants_file": "2022-12-08_logmel_3_channel_noAug_noBadAudio.pickle",
    #     "augmented_files": ["2022-12-08_logmel_3_channel_augmented_noBadAudio.pickle"]
    #     # "augmented_files": None
    # },
    # "logmel_3_channels_1024_2048_4096": {
    #     "dataset_class": ResnetLogmel3Channels,
    #     "participants_file": "2022-12-10_logmel_3_channel_noAug_1024x2048x4096.pickle",
    #     "augmented_files": ["2022-12-10_logmel_3_channel_augmented_1024x2048x4096.pickle"]
    #     # "augmented_files": None
    # },
    # "logmel_1_channel_breath": {
    #     "dataset_class": ResnetLogmel1ChannelBreath,
    #     "participants_file": "2022-12-11_logmel_1_channel_noAug_heavy_breathing.pickle",
    #     "augmented_files": ["2022-12-11_logmel_1_channel_augmented_heavy_breathing.pickle"]
    #     # "augmented_files": None
    # }
}
device = "cuda" if cuda.is_available() else "cpu"
TESTING_MODE = not cuda.is_available()


# <editor-fold desc="function definitions">
def summarize_cuda_memory_usage(summarize_device=False, detailed=False):
    if not cuda.is_available():
        return
    current_device_idx = cuda.current_device()
    if summarize_device:
        print("---------------------------------------------------------------------------------------")
        device_count = cuda.device_count()
        device_name = cuda.get_device_name(current_device_idx)
        # device_properties = cuda.get_device_properties(current_device_idx)
        print(f"Number of available devices: {device_count}")
        print("Current Device:")
        print(device_name)
        # print(device_properties)

    print("---------------------------------------------------------------------------------------")
    if detailed:
        print(cuda.memory_summary(current_device_idx))
    else:
        free_mem, total_mem = cuda.mem_get_info(current_device_idx)
        # convert to gigabyte
        free_mem =  round(free_mem/(1024**2))
        total_mem = round(total_mem/(1024**2))
        print(f"{free_mem} MB / {total_mem} MB are still available")


def get_parameter_groups(model, output_lr, input_lr, verbose=True):
    # applies different learning rates for each (parent) layer in the model (for finetuning a pretrained network). The
    # inout layer gets the input_lr, the output layer the output_lr. All layers in between get linearly interpolated.

    # works for resnet architecture and assigns a learning rate for each parent layer and the input and output layers
    # in total there are (for a resnet 18) 61 parameter groups but only 4 parent layers and 3 layers as input/output
    # layers. this means there are only  4+3  different learning rates.

    parent_layer = lambda name: name.split(".")[0]
    layer_names = [name for name, _ in model.named_parameters()]
    layer_names.reverse()
    parent_layers = list(set([parent_layer(layer) for layer in layer_names]))
    n_parent_layers = len(parent_layers)
    lr = output_lr
    last_parent_layer = parent_layer(layer_names[0])
    if verbose:
        print(f'0: lr = {lr:.6f}, {last_parent_layer}')

    lr_mult = np.power(input_lr / output_lr, 1 / (n_parent_layers - 1))
    parameter_groups = []
    for idx, layer in enumerate(layer_names):
        current_parent_layer = parent_layer(layer)
        if last_parent_layer != current_parent_layer:
            lr *= lr_mult
            if verbose:
                print(f'{idx}: lr = {lr:.6f}, {current_parent_layer}')
            last_parent_layer = current_parent_layer
        parameter_groups.append({'params': [p for n, p in model.named_parameters() if n == layer and p.requires_grad],
                                 'lr': lr})
    return parameter_groups


def get_parameter_combinations(param_dict, verbose=True):
    Run = namedtuple("Run", param_dict.keys())
    runs = []
    for run_combination in product(*param_dict.values()):
        runs.append(Run(*run_combination))
    print(f"#Parameter Combinations: {len(runs)}")
    return runs


def train_on_batch(model, current_batch, current_loss_func, current_optimizer, my_tracker):
    model.train()
    input_data, label = current_batch
    input_data, label = input_data.to(device), label.to(device)

    prediction = torch.squeeze(model(input_data))
    if prediction.dim() == 0:
        prediction = torch.unsqueeze(prediction, dim=0)
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
    if prediction.dim() == 0:
        prediction = torch.unsqueeze(prediction, dim=0)
    loss = loss_function(prediction, label)
    # pred_after_sigmoid = torch.sigmoid(prediction)
    # accuracy = get_accuracy(prediction, label)
    my_tracker.add_metrics(loss, label, prediction)


def get_ids_of(participants_filename):
    # get list of positive, negative and invalid ids from a list of participant instances
    path = os.path.join("data", "Coswara_processed", "pickles", participants_filename)
    with open(path, "rb") as f:
        parts = pickle.load(f).participants
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


def load_train_val_and_test_set_ids_from_disc(rand_seed, split_ratio):
    # test set is fixed on disc! with (now) 15% of the whole valid samples (excluded bad covid labels and invalid audio
    # although low quality audio is still included)
    # The remaining training and evaluation set (85% - ~2000 samples) are randomly split in each fold during cross val.
    test_set = pd.read_csv("data/Coswara_processed/test_set_df.csv")
    train_and_validation_set = pd.read_csv("data/Coswara_processed/train_and_validation_set_df.csv")
    test_set_ids = list(test_set["user_id"])

    train_val_pos_ids = train_and_validation_set[train_and_validation_set["covid_label"] == 1.0]
    train_val_neg_ids = train_and_validation_set[train_and_validation_set["covid_label"] == 0.0]
    train_val_pos_ids, train_val_neg_ids = list(train_val_pos_ids["user_id"]), list(train_val_neg_ids["user_id"])

    pos_ids_train, pos_ids_val = randomly_split_list_into_two(train_val_pos_ids, ratio=split_ratio,
                                                              random_seed=rand_seed)
    neg_ids_train, neg_ids_val = randomly_split_list_into_two(train_val_neg_ids, ratio=split_ratio,
                                                              random_seed=rand_seed)
    train_set_ids = pos_ids_train + neg_ids_train
    validation_set_ids = pos_ids_val + neg_ids_val

    # does it help or hurt to shuffle the ids around so that not all pos/neg samples are bunched together?
    # depends on whether shuffle is activated for the data loader?(the weightedrandsampler should always draw randomly?)
    random.Random(rand_seed).shuffle(train_set_ids)
    random.Random(rand_seed).shuffle(validation_set_ids)
    random.Random(rand_seed).shuffle(test_set_ids)

    return train_set_ids, validation_set_ids, test_set_ids


def get_datasets(dataset_name, split_ratio=0.8, transform=None, train_augmentation=None, random_seed=None, params=None):
    dataset_dict = dataset_collection[dataset_name]
    DatasetClass = dataset_dict["dataset_class"]

    if USE_TRAIN_VAL_TEST_SPLIT:
        train_ids, validation_ids, test_ids = load_train_val_and_test_set_ids_from_disc(rand_seed=random_seed,
                                                                                        split_ratio=split_ratio)
    else:
        # deprecated - old version
        pos_ids, neg_ids, invalid_ids = get_ids_of(dataset_dict["participants_file"])
        pos_ids_train, pos_ids_val = randomly_split_list_into_two(pos_ids, ratio=split_ratio, random_seed=random_seed)
        neg_ids_train, neg_ids_val = randomly_split_list_into_two(neg_ids, ratio=split_ratio, random_seed=random_seed)
        train_ids = pos_ids_train + neg_ids_train
        validation_ids = pos_ids_val + neg_ids_val

    if QUICK_TRAIN_FOR_TESTS:
        np.random.shuffle(train_ids)
        train_ids = train_ids[:200]
        np.random.shuffle(validation_ids)
        validation_ids = validation_ids[:50]

    if not params.use_augm_datasets:
        augmented_datasets = None
    else:
        augmented_datasets = dataset_dict["augmented_files"]

    training_set = DatasetClass(user_ids=train_ids, original_files=dataset_dict["participants_file"],
                                transform=transform, augmented_files=augmented_datasets,
                                augmentations=train_augmentation, verbose=VERBOSE, mode="train")
    validation_set = DatasetClass(user_ids=validation_ids, original_files=dataset_dict["participants_file"],
                                  transform=transform, verbose=VERBOSE, mode="eval")

    training_set.mix_up_alpha = params.mixup_a
    training_set.mix_up_probability = params.mixup_p
    return training_set, validation_set


def get_data_loaders(training_set, validation_set, params):
    # create weighted random sampler
    label_counts = training_set.label_counts()[1]
    label_weights = np.flip(label_counts / np.sum(label_counts))

    # sample_weights = []
    # for (data, label) in training_set:
    #     sample_weights.append(label_weights[int(label)])

    # sample_weights = [label_weights[int(label)] for (data, label) in training_set]
    sample_weights = [label_weights[int(label)] for label in training_set.labels]


    # might actually set num_samples higher because like this not all samples from the dataset are chosen within 1 epoch
    # or fix to "lower" num_sumbples so that we always have the same number of samples within an epoch, no matter how
    # many augmented samples will be added to the training data. This adds comparability between runs.

    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    sampler = WeightedRandomSampler(sample_weights, num_samples=512, replacement=True)

    # create dataloaders
    n_workers = 0
    if cuda.is_available():
        n_workers = 0
    if params.weighted_sampler:
        train = DataLoader(dataset=training_set, batch_size=p.batch, drop_last=True, sampler=sampler,
                           num_workers=n_workers)
    else:
        train = DataLoader(dataset=training_set, batch_size=p.batch, shuffle=True, drop_last=True,
                           num_workers=n_workers)


    val = DataLoader(dataset=validation_set, batch_size=p.batch, drop_last=False, num_workers=n_workers)
    return train, val


def get_model(model_name, params, verbose=True, load_from_disc=False):
    model_dict = {
        "brogrammers": BrogrammersModel,
        "brogrammers_old": BrogrammersSequentialModel,
        "resnet18": get_resnet18,
        "resnet50": get_resnet50,
        "MIL_brogrammers": BrogrammersMIL,
        "Resnet18_MIL": Resnet18MIL,
        "PredictionLevelMIL_mfcc": PredLevelMIL
    }
    if model_name in ["MIL_brogrammers", "Resnet18_MIL", "PredictionLevelMIL_mfcc"]:
        my_model = model_dict[model_name](n_hidden_attention=params.n_MIL_Neurons).to(device)
    else:
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


def write_metrics(mode):
    metrics = tracker.get_epoch_metrics()
    if TRACK_METRICS:
        writer.add_scalar(f"01_loss/{mode}", metrics["loss"], epoch)
        writer.add_scalar(f"02_accuracy/{mode}", metrics["accuracy"], epoch)
        writer.add_scalar(f"03_AUC-ROC/{mode}", metrics["auc_roc"], epoch)
        writer.add_scalar(f"04_f1_score/{mode}", metrics["f1_score"], epoch)
        writer.add_scalar(f"05_AUC-precision-recall/{mode}", metrics["auc_prec_recall"], epoch)
        writer.add_scalar(f"06_TPR_or_Recall_or_Sensitivity/{mode}", metrics["tpr"], epoch)
        writer.add_scalar(f"07_TrueNegativeRate_or_Specificity/{mode}", metrics["tnr"], epoch)
        writer.add_scalar(f"08_Precision_or_PositivePredictiveValue/{mode}", metrics["precision"], epoch)
        writer.add_scalar(f"09_true_positives_at_95/{mode}", metrics["tpr_at_95"], epoch)
    return metrics["loss"]


def get_online_augmentations(run_parameters):
    if run_parameters.shift:
        augmentation = Compose([AddGaussianNoise(0, run_parameters.sigma), CyclicTemporalShift()])
    else:
        augmentation = Compose([AddGaussianNoise(0, run_parameters.sigma)])
    return augmentation


# </editor-fold>

random_seeds = [99468865, 215674, 3213213211, 55555555, 66445511337,
                316497938271, 161094, 191919191, 101010107, 123587955]
# random_seeds = [123587955, 99468865, 215674, 3213213211, 55555555,
#                 66445511337, 316497938271, 161094, 191919191, 101010107]
# first seed (123587955) has a very difficult/bad performing validation set

summarize_cuda_memory_usage(summarize_device=True)
# ###############################################  manual setup  #######################################################
USE_TRAIN_VAL_TEST_SPLIT = True  # use a 70/15/15 split instead of an 80/20 split without test set
QUICK_TRAIN_FOR_TESTS = False

n_epochs = 100
n_cross_validation_runs = 1

parameters = dict(
    # rand=random_seeds[:n_cross_validation_runs],
    batch=[64],
    lr=[5e-4],
    wd=[1e-4],  # weight decay regularization
    lr_decay=[.975],
    mixup_a=[0.4],  # alpha value to decide probability distribution of how much of each of the samples is used
    mixup_p=[1.0],  # probability of mix up being used at all
    use_augm_datasets=[False],
    shift=[True],
    sigma=[0.05],
    weighted_sampler=[True],  # whether to use a weighted random sampler to address the class imbalance
    class_weight=[1],  # factor for loss of the positive class to address class imbalance
    bag_size=[8],
    n_MIL_Neurons=[64]
)

transforms = None
augmentations = Compose([AddGaussianNoise(0, 0.05), CyclicTemporalShift()])

# "brogrammers", "resnet18", "resnet50", MIL_brogrammers, Resnet18_MIL, PredictionLevelMIL_mfcc
MODEL_NAME = "Resnet18_MIL"
# logmel_1_channel, 15_mfccs, 15_mfccs_highRes, 15_mfccs_highres_new, brogrammers_new,mfccs_3s_breathing_deep
# mfccs_3s_combined_coughs, logmel_3s_combined_coughs, mfcc_mil_combined_cough, resnet_mil_combined_cough
# combined_breaths_3s_2048fft, combined_breaths_6s_2048fft, combined_breaths_6s_4096fft, combined_breaths_6s_2048fft_fmax5500
# mfcc_vowel_a_6s_FFT2048_fmax5500, mfcc_vowel_e_6s_FFT2048_fmax5500, mfcc_vowels_combined_6s_FFT2048_fmax5500
# combined_breaths_12s_FFT4096_fmax5500_50mfccs
DATASET_NAME = "combined_breaths_12s_FFT4096_fmax5500_50mfccs"
RUN_COMMENT = f"test"

print(f"Dataset used: {DATASET_NAME}")
print(f"model used: {MODEL_NAME}")
date = datetime.today().strftime("%Y-%m-%d")
RUN_NAME = f"{date}_{MODEL_NAME}_{DATASET_NAME}_{RUN_COMMENT}"
VERBOSE = True
LOAD_FROM_DISC = False
SAVE_TO_DISC = False

if device == "cpu":
    window = tk.Tk()
    TRACK_METRICS = askyesno(title='Tracking Settings',
                             message=f'Do you want to track this run?\nIt will be saved as: {RUN_NAME}')
    window.destroy()
else:
    TRACK_METRICS = True

# ############################################ setup ###################################################################
tracker = IntraEpochMetricsTracker(datasets={DATASET_NAME: dataset_collection[DATASET_NAME]}, verbose=TESTING_MODE)
for p in get_parameter_combinations(parameters, verbose=True):
    tracker.setup_run_with_new_params(p)
    for random_seed in random_seeds[:n_cross_validation_runs]:
        summarize_cuda_memory_usage()
        tracker.start_run_with_random_seed(random_seed)
        train_set, val_set = get_datasets(DATASET_NAME, split_ratio=0.8, transform=transforms,
                                          train_augmentation=augmentations, random_seed=random_seed, params=p)


        train_set.n_channels = 3
        train_set.n_timesteps = 120
        val_set.n_channels = 3
        val_set.n_timesteps = 120

        train_set.bag_size = p.bag_size
        val_set.bag_size = p.bag_size


        train_set.augmentations = get_online_augmentations(p)
        if TRACK_METRICS:
            writer = SummaryWriter(log_dir=f"run/tensorboard_saves/{RUN_NAME}/{p}")

        train_loader, eval_loader = get_data_loaders(train_set, val_set, p)
        my_cnn = get_model(MODEL_NAME, p, load_from_disc=LOAD_FROM_DISC, verbose=False)

        optimizer = get_optimizer(MODEL_NAME, load_from_disc=LOAD_FROM_DISC)
        lr_scheduler = ExponentialLR(optimizer, gamma=p.lr_decay)
        # lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.1)
        loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p.class_weight])).to(device)
        # loss_func = nn.BCEWithLogitsLoss().to(device)
        tracker.save_model_and_training_parameters(my_cnn, optimizer, loss_func)
        tracker.types_of_recording = train_set.types_of_recording
        tracker.audio_processing_params = train_set.audio_proc_params
        tracker.augmentations = train_set.predetermined_augmentations
        tracker.augmentations_per_label = train_set.augmentations_per_label
        tracker.train_set_label_counts = f"label '0': {train_set.label_counts()[1][0]}  -  " \
                                         f"label '1': {train_set.label_counts()[1][1]}"

        # , train_set.audio_proc_params, train_set.predetermined_augmentations
        # ################################################ training ####################################################
        epoch_start = time.time()
        for epoch in range(n_epochs):

            tracker.reset(p, mode="train")
            for i, batch in enumerate(train_loader):
                train_on_batch(my_cnn, batch, loss_func, optimizer, tracker)
            epoch_loss = write_metrics(mode="train")

            with torch.no_grad():
                tracker.reset(p, mode="eval")
                for i, batch in enumerate(eval_loader):
                    evaluate_batch(my_cnn, batch, loss_func, tracker)
                epoch_loss = write_metrics(mode="eval")
            lr_scheduler.step()
            # lr_scheduler.step(epoch_loss)

            if TESTING_MODE:
                print(f"current learning rates: {[round(lr, 8) for lr in lr_scheduler.get_last_lr()]}")
        # ##############################################################################################################
        if VERBOSE:
            delta_t = time.time() - epoch_start
            print(f"Run {p} took [{int(delta_t // 60)}min {int(delta_t % 60)}s] to calculate")

    if TRACK_METRICS:
        with open(f"run/tracker_saves/{RUN_NAME}.pickle", "wb") as f:
            pickle.dump(tracker, f)

if TRACK_METRICS:
    writer.close()

if SAVE_TO_DISC:
    print("saving new model!")
    MODEL_PATH = f"data/Coswara_processed/models/{MODEL_NAME}/model.pth"
    torch.save(my_cnn.state_dict(), MODEL_PATH)
    # optimizer.zero_grad()
    OPTIMIZER_PATH = f"data/Coswara_processed/models/{MODEL_NAME}/optimizer.pickle"
    torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
