import random

import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from torchvision import transforms


def evenly_distributed_cyclic_shifts(input_matrix, n_output_timesteps, n=10):
    # from a 2D time frequency/quefrency matrix of m time indices create n matrices with fixed "output_size". These
    # matrices are all shifted by 1/n of the total number of time indices
    total_time_steps = input_matrix.shape[-1]
    if total_time_steps < n_output_timesteps:
        input_matrix = np.pad(input_matrix, ((0, 0), (0, n_output_timesteps - total_time_steps)))
        total_time_steps = n_output_timesteps

    delta_shift = total_time_steps // n
    shifts = [i*delta_shift for i in range(n)]
    # print(shifts)
    output_matrices = np.array([np.roll(input_matrix, shift=shift, axis=-1) for shift in shifts])
    output_matrices = output_matrices[:, :, :n_output_timesteps]
    return output_matrices


class CustomDataset(Dataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train"):
        self.mode = mode
        self.user_ids = user_ids

        self.transform = transform if transform is not None else transforms.ToTensor()
        self.augmentations = augmentations

        with open(f"data/Coswara_processed/pickles/{original_files}", "rb") as f:
            self.participants = pickle.load(f)

        if augmented_files is not None:
            for pickle_file in augmented_files:
                with open(f"data/Coswara_processed/pickles/{pickle_file}", "rb") as f:
                    self.participants += pickle.load(f)

        self.participants = [part for part in self.participants if part.id in user_ids]

        self.drop_invalid_labels()
        self.drop_bad_audio()
        self.labels = np.array([int(participant.get_label()) for participant in self.participants])
        self.mu, self.sigma = self.get_feature_statistics()
        if verbose:
            self.summarize()

        self.mix_up_alpha = 0.2
        self.mix_up_probability = 1.0


    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]

    def drop_bad_audio(self):
        self.participants = [participant for participant in self.participants if
                             participant.meta_data["audio_quality_heavy_cough"] > 0.0]

    def get_input_features(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        input_features = self.get_input_features(idx)
        # input_features = self.transform(input_features)
        if self.augmentations is not None:
            input_features = self.augmentations(input_features)

        n_timesteps = input_features.shape[2]
        if n_timesteps < self.n_timesteps:
            input_features = torch.nn.functional.pad(input_features, (0, self.n_timesteps - n_timesteps))
        elif n_timesteps > self.n_timesteps:
            input_features = input_features[:, :, :self.n_timesteps]

        output_label = self.participants[idx].get_label()

        if self.mode == "train" and np.random.rand() < self.mix_up_probability:
            # only use mixup on every third sample --> hyperparameter (maybe make dependent on epoch/learning rate
            # probability = 1-np.float_power(epoch, -0.3)
            input_features, output_label = self.mix_up(orig_sample=input_features, orig_label=output_label)

        n_channels = input_features.shape[0]
        if n_channels < self.n_channels:
            input_features = input_features.expand(3, -1, -1)
        return input_features, torch.tensor(output_label).float()

    def __len__(self):
        return len(self.participants)

    def get_object(self, idx):
        return self.participants[idx]

    def z_normalize(self, mfccs):
        return (mfccs - self.mu) / self.sigma

    def label_counts(self):
        return np.unique(self.labels, return_counts=True)

    def get_input_shape(self):
        in_features, _ = self.__getitem__(0)
        return in_features.shape

    def summarize(self):
        print(f"Total number of items = {len(self)}")
        print(f"label count = {self.label_counts()}")
        # print(f"shape of input data without batch size: {self.get_input_shape()}")

    @staticmethod
    def get_feature_statistics():
        # why do I have to convert it to float32???
        mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
        mu = np.expand_dims(mu, axis=1).astype("float32")
        sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
        sigma = np.expand_dims(sigma, axis=1).astype("float32")
        return mu, sigma

    def mix_up(self, orig_sample, orig_label):
        # TODO include probability to use mixup at all as input
        mixup_idx = np.random.randint(self.__len__())
        mixup_sample = self.get_input_features(mixup_idx)
        # CyclicShift(mixup_sample) / np.roll() / torch.roll()
        # mixup_sample = mixup_sample[:, :, self.n_timesteps]
        if self.augmentations is not None:
            mixup_sample = self.augmentations(mixup_sample)

        n_timesteps = mixup_sample.shape[2]
        if n_timesteps < self.n_timesteps:
            mixup_sample = torch.nn.functional.pad(mixup_sample, (0, self.n_timesteps - n_timesteps))
        elif n_timesteps > self.n_timesteps:
            mixup_sample = mixup_sample[:, :, :self.n_timesteps]

        # TODO use padding before cyclic shift to avoid bias

        mixup_label = self.participants[mixup_idx].get_label()

        # alpha = 0.2
        lamda = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
        mixed_up_label = lamda * orig_label + (1 - lamda) * mixup_label
        mixed_up_sample = lamda * orig_sample + (1 - lamda) * mixup_sample
        return mixed_up_sample, mixed_up_label



class BrogrammersMFCCDataset(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train"):
        self.n_channels = 1
        self.n_timesteps = 259
        super(BrogrammersMFCCDataset, self).__init__(user_ids, original_files, transform, augmentations,
                                                     augmented_files, verbose)

    def get_input_features(self, idx):
        input_features = self.participants[idx].heavy_cough.MFCCs
        input_features = self.z_normalize(input_features)
        input_features = self.transform(input_features)
        return input_features

    @staticmethod
    def get_feature_statistics():
        # why do I have to convert it to float32???
        mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
        mu = np.expand_dims(mu, axis=1).astype("float32")
        sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
        sigma = np.expand_dims(sigma, axis=1).astype("float32")
        return mu, sigma


class ResnetLogmelDataset(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train"):
        self.n_channels = 3
        self.n_timesteps = 224
        self.frequency_resolution = 224
        super(ResnetLogmelDataset, self).__init__(user_ids, original_files, transform,
                                                  augmentations, augmented_files, verbose, mode)

    def get_input_features(self, idx):
        input_features = self.participants[idx].heavy_cough.logmel
        input_features = self.transform(input_features)
        return input_features

    @staticmethod
    def get_feature_statistics():
        # do nothing - no normalization
        return 0., 1.


class ResnetLogmel3Channels(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train"):
        self.n_channels = 3
        self.n_timesteps = 224
        self.frequency_resolution = 224
        super(ResnetLogmel3Channels, self).__init__(user_ids, original_files, transform,
                                                    augmentations, augmented_files, verbose, mode)

    def get_input_features(self, idx):
        input_features = self.participants[idx].heavy_cough.logmel_3c
        return torch.Tensor(input_features)

    @staticmethod
    def get_feature_statistics():
        # do nothing - no normalization
        return 0., 1.


class ResnetLogmel1ChannelBreath(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train"):
        self.n_channels = 3
        self.n_timesteps = 224
        self.frequency_resolution = 224
        super(ResnetLogmel1ChannelBreath, self).__init__(user_ids, original_files, transform,
                                                         augmentations, augmented_files, verbose, mode)

    def get_input_features(self, idx):
        input_features = self.participants[idx].deep_breath.logmel
        input_features = self.transform(input_features)
        return input_features

    @staticmethod
    def get_feature_statistics():
        # do nothing - no normalization
        return 0., 1.




class BrogrammersMfccHighRes(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train"):
        self.n_channels = 1
        self.n_timesteps = 259
        super(BrogrammersMfccHighRes, self).__init__(user_ids, original_files, transform, augmentations,
                                                     augmented_files, verbose, mode)

    def get_input_features(self, idx):
        input_features = self.participants[idx].heavy_cough.MFCCs
        input_features = self.z_normalize(input_features)
        if self.mode == "train":
            input_features = self.transform(input_features)
        else:
            input_features = evenly_distributed_cyclic_shifts(input_features, n_output_timesteps=self.n_timesteps, n=10)
            input_features = torch.Tensor(input_features)

        return input_features

    @staticmethod
    def get_feature_statistics():
        # why do I have to convert it to float32???
        mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
        mu = np.expand_dims(mu, axis=1).astype("float32")
        sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
        sigma = np.expand_dims(sigma, axis=1).astype("float32")
        return mu, sigma