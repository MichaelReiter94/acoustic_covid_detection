import random
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from torchvision import transforms
from audio_processing import FeatureSet
from utils.utils import audiomentations_repr
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode=None, min_audio_quality=1):

        if mode is None:
            print("no mode (train or eval) specified for dataset")
            raise ValueError
        self.min_audio_quality = min_audio_quality
        self.mode = mode
        self.user_ids = user_ids

        self.transform = transform if transform is not None else transforms.ToTensor()
        self.augmentations = augmentations
        self.predetermined_augmentations = []
        self.augmentations_per_label = []

        with open(f"data/Coswara_processed/pickles/{original_files}", "rb") as f:
            feature_set = pickle.load(f)
            self.audio_proc_params = feature_set.audio_parameters
            self.types_of_recording = feature_set.types_of_recording
            self.participants = feature_set.participants
            # self.feature_set = pickle.load(f)

        if augmented_files is not None:
            for pickle_file in augmented_files:
                with open(f"data/Coswara_processed/pickles/{pickle_file}", "rb") as f:
                    feature_set = pickle.load(f)
                    self.predetermined_augmentations.append(audiomentations_repr(feature_set.augmentations))
                    self.augmentations_per_label.append(feature_set.augmentations_per_label)

                    self.participants += feature_set.participants

        self.participants = [part for part in self.participants if part.id in user_ids]


        self.drop_invalid_labels()
        self.drop_bad_audio()
        self.labels = np.array([int(participant.get_label()) for participant in self.participants])
        self.mu, self.sigma = self.get_feature_statistics()
        if verbose:
            self.summarize()

        self.mix_up_alpha = 0.2
        self.mix_up_probability = 1.0
        self.bag_size = 1

    def evenly_distributed_cyclic_shifts(self, input_matrix, n_output_timesteps, shift_std=0.25):
        # from a 2D time frequency/quefrency matrix of m time indices create n matrices with fixed "output_size". These
        # matrices are all shifted by 1/n of the total number of time indices
        # TODO implement better:
        # n = np.random.randint(4, 32)
        n = self.bag_size
        total_time_steps = input_matrix.shape[-1]
        if total_time_steps < n_output_timesteps:
            input_matrix = np.pad(input_matrix, ((0, 0), (0, n_output_timesteps - total_time_steps)))
            total_time_steps = n_output_timesteps

        delta_shift = total_time_steps // n

        if self.mode == "eval":
            shift_std = 0.0

        shifts = [i * delta_shift for i in range(n)]
        shifts += (np.random.randn(n) * shift_std * delta_shift).astype("int")  #

        # print(shifts)
        output_matrices = np.array([np.roll(input_matrix, shift=shift, axis=-1) for shift in shifts])
        output_matrices = output_matrices[:, :, :n_output_timesteps]
        return output_matrices

    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]



    def drop_bad_audio(self):
        # TODO do the same thing for other types of recording
        df = pd.read_csv("data/Coswara_processed/full_meta_data.csv")
        manually_identified_bad_ids = list(pd.read_excel
                                           (r"data/Coswara_processed/bad ids from listening and analysis.xlsx",
                                           sheet_name=self.types_of_recording, usecols=["ID"]).ID)
        audio_quality_thresh = self.min_audio_quality
        low_audio_quality_ids = []
        if self.types_of_recording == "combined_breaths":
            # there are 3 categories of audio qulaity with 0 being the worst and 2 being the best
            low_audio_quality_ids = list(df[(df["audio_quality_breathing-shallow"] < audio_quality_thresh) |
                                            (df["audio_quality_breathing-deep"] < audio_quality_thresh)]["user_id"])
        elif self.types_of_recording == "combined_coughs":
            low_audio_quality_ids = list(df[(df["audio_quality_cough-heavy"] < audio_quality_thresh) |
                                            (df["audio_quality_cough-shallow"] < audio_quality_thresh)]["user_id"])
        elif self.types_of_recording == "combined_vowels":
            low_audio_quality_ids = list(df[(df["vowel-a"] < audio_quality_thresh) |
                                            (df["vowel-e"] < audio_quality_thresh) |
                                            (df["vowel-o"] < audio_quality_thresh)]["user_id"])
        else:
            print("You are not using combined sound recordings. There is no code for checcking for the audio quality "
                  "of those separate recordings")
        all_bad_ids = manually_identified_bad_ids + low_audio_quality_ids
        self.participants = [part for part in self.participants if part.id not in all_bad_ids]



    def get_input_features(self, idx, for_mix_up=False):
        input_features = self.participants[idx].recordings[self.types_of_recording].features
        input_features = self.transform(input_features)
        if self.augmentations is not None:
            input_features = self.augmentations(input_features)
        return input_features
        # raise NotImplementedError

    def __getitem__(self, idx):
        input_features = self.get_input_features(idx)
        # input_features = self.transform(input_features)
        # if self.augmentations is not None:
        #     input_features = self.augmentations(input_features)

        n_timesteps = input_features.shape[-1]
        if n_timesteps < self.n_timesteps:
            input_features = torch.nn.functional.pad(input_features, (0, self.n_timesteps - n_timesteps))
        elif n_timesteps > self.n_timesteps:
            input_features = input_features[:, :, :self.n_timesteps]

        output_label = self.participants[idx].get_label()

        if self.mode == "train" and np.random.rand() < self.mix_up_probability:
            # only use mixup on every third sample --> hyperparameter (maybe make dependent on epoch/learning rate
            # probability = 1-np.float_power(epoch, -0.3)
            input_features, output_label = self.mix_up(orig_sample=input_features, orig_label=output_label)

        n_channels = input_features.shape[-3]
        if n_channels < self.n_channels:
            if input_features.dim() == 3:
                input_features = input_features.expand(3, -1, -1)
            elif input_features.dim() == 4:
                input_features = input_features.expand(-1, 3, -1, -1)

        return input_features, torch.tensor(output_label).float()

    def __len__(self):
        return len(self.participants)

    def get_object(self, idx):
        return self.participants[idx]

    def z_normalize(self, mfccs):
        return (mfccs - self.mu) / self.sigma

    def label_counts(self):
        label_count = np.unique(self.labels, return_counts=True)
        return label_count

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
        target_label = random.choice([0, 1])
        mixup_label = None
        while mixup_label != target_label:
            # make sure, that mixup chooses positive and negative samples with 50% each
            # (not depending on the occurences in the dataset)
            mixup_idx = np.random.randint(self.__len__())
            mixup_label = self.participants[mixup_idx].get_label()

        mixup_sample = self.get_input_features(mixup_idx, for_mix_up=True)
        if self.augmentations is not None:
            mixup_sample = self.augmentations(mixup_sample)

        n_timesteps = mixup_sample.shape[-1]
        if n_timesteps < self.n_timesteps:
            mixup_sample = torch.nn.functional.pad(mixup_sample, (0, self.n_timesteps - n_timesteps))
        elif n_timesteps > self.n_timesteps:
            mixup_sample = mixup_sample[:, :, :self.n_timesteps]

        lamda = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
        mixed_up_label = lamda * orig_label + (1 - lamda) * mixup_label
        mixed_up_sample = lamda * orig_sample + (1 - lamda) * mixup_sample
        return mixed_up_sample, mixed_up_label


class BrogrammersMFCCDataset(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train", min_audio_quality=1):
        self.n_channels = 1
        self.n_timesteps = 259
        super(BrogrammersMFCCDataset, self).__init__(user_ids, original_files, transform, augmentations,
                                                     augmented_files, verbose, mode, min_audio_quality)

    def get_input_features(self, idx, for_mix_up=False):
        input_features = self.participants[idx].recordings[self.types_of_recording].features
        #     input_features = self.participants[idx].heavy_cough.MFCCs
        input_features = self.z_normalize(input_features)
        input_features = self.transform(input_features)
        if self.augmentations is not None:
            input_features = self.augmentations(input_features)
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
                 verbose=True, mode=None, min_audio_quality=1):
        self.n_channels = 3
        self.n_timesteps = 224
        self.frequency_resolution = 224
        super(ResnetLogmelDataset, self).__init__(user_ids, original_files, transform,
                                                  augmentations, augmented_files, verbose, mode, min_audio_quality)

    # def get_input_features(self, idx):
    #     input_features = self.participants[idx].heavy_cough.logmel
    #     input_features = self.transform(input_features)
    #     return input_features

    @staticmethod
    def get_feature_statistics():
        # do nothing - no normalization
        return 0., 1.


# class ResnetLogmel3Channels(CustomDataset):
#     def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
#                  verbose=True, mode="train"):
#         self.n_channels = 3
#         self.n_timesteps = 224
#         self.frequency_resolution = 224
#         super(ResnetLogmel3Channels, self).__init__(user_ids, original_files, transform,
#                                                     augmentations, augmented_files, verbose, mode)
#
#     # def get_input_features(self, idx):
#     #     input_features = self.participants[idx].heavy_cough.logmel_3c
#     #     return torch.Tensor(input_features)
#
#     @staticmethod
#     def get_feature_statistics():
#         # do nothing - no normalization
#         return 0., 1.


# class ResnetLogmel1ChannelBreath(CustomDataset):
#     def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
#                  verbose=True, mode="train"):
#         self.n_channels = 3
#         self.n_timesteps = 224
#         self.frequency_resolution = 224
#         super(ResnetLogmel1ChannelBreath, self).__init__(user_ids, original_files, transform,
#                                                          augmentations, augmented_files, verbose, mode)
#
#     # def get_input_features(self, idx):
#     #     input_features = self.participants[idx].deep_breath.logmel
#     #     input_features = self.transform(input_features)
#     #     return input_features
#
#     @staticmethod
#     def get_feature_statistics():
#         # do nothing - no normalization
#         return 0., 1.


class MultipleInstanceLearningMFCC(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode="train", min_audio_quality=1):
        self.n_channels = 1
        self.n_timesteps = 259
        super(MultipleInstanceLearningMFCC, self).__init__(user_ids, original_files, transform, augmentations,
                                                           augmented_files, verbose, mode, min_audio_quality)

    def get_input_features(self, idx, for_mix_up=False):
        input_features = self.participants[idx].recordings[self.types_of_recording].features

        # input_features = self.z_normalize(input_features) TODO uncomment or replace
        if self.augmentations is not None:
            input_features = self.augmentations(input_features)
        # if self.mode == "train":
        #     input_features = self.transform(input_features)
        # else:

        input_features = self.evenly_distributed_cyclic_shifts(input_features, n_output_timesteps=self.n_timesteps)
        input_features = np.expand_dims(input_features, 1)

        input_features = torch.Tensor(input_features)

        return input_features

    @staticmethod
    def get_feature_statistics():
        # why do I have to convert it to float32???
        # because float64 is double and float and double don't mix in torch?
        mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
        mu = np.expand_dims(mu, axis=1).astype("float32")
        sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
        sigma = np.expand_dims(sigma, axis=1).astype("float32")
        return mu, sigma


class MILResnet(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True, mode=None, min_audio_quality=1):
        self.n_channels = 3
        self.n_timesteps = 224
        self.frequency_resolution = 224
        super(MILResnet, self).__init__(user_ids, original_files, transform, augmentations, augmented_files, verbose,
                                        mode, min_audio_quality)


    def get_input_features(self, idx, for_mix_up=False):
        input_features = self.participants[idx].recordings[self.types_of_recording].features

        # input_features = self.z_normalize(input_features)
        if self.augmentations is not None:
            input_features = self.augmentations(input_features)
        # if self.mode == "train":
        #     input_features = self.transform(input_features)
        # else:

        # if not for_mix_up:
        #     self.bag_size = np.random.randint(4, 16)
        # if self.mode == "eval":
        #     self.bag_size = 8

        input_features = self.evenly_distributed_cyclic_shifts(input_features, n_output_timesteps=self.n_timesteps)
        input_features = np.expand_dims(input_features, 1)

        input_features = torch.Tensor(input_features)

        return input_features


    @staticmethod
    def get_feature_statistics():
        # do nothing - no normalization
        return 0., 1.
