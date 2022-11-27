import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from torchvision import transforms



class CustomDataset(Dataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True):

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

    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]

    def drop_bad_audio(self):
        self.participants = [participant for participant in self.participants if
                             participant.meta_data["audio_quality_heavy_cough"] > 0.0]

    def get_input_features(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        input_features = self.get_input_features(idx)
        input_features = self.transform(input_features)
        if self.augmentations is not None:
            input_features = self.augmentations(input_features)

        n_timesteps = input_features.shape[2]
        if n_timesteps < self.n_timesteps:
            input_features = torch.nn.functional.pad(input_features, (0, self.n_timesteps - n_timesteps))
        elif n_timesteps > self.n_timesteps:
            input_features = input_features[:, :, :self.n_timesteps]

        n_channels = input_features.shape[0]
        if n_channels < self.n_channels:
            input_features = input_features.expand(3, -1, -1)
        output_label = self.participants[idx].get_label()
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
        print(f"shape of input data without batch size: {self.get_input_shape()}")

    @staticmethod
    def get_feature_statistics():
        # why do I have to convert it to float32???
        # TODO make individual for each specific subclass and ignore if there are no statistics
        mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
        mu = np.expand_dims(mu, axis=1).astype("float32")
        sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
        sigma = np.expand_dims(sigma, axis=1).astype("float32")
        return mu, sigma


class BrogrammersMFCCDataset(CustomDataset):
    def __init__(self, user_ids, original_files, transform=None, augmentations=None, augmented_files=None,
                 verbose=True):
        self.n_channels = 1
        self.n_timesteps = 259
        super(BrogrammersMFCCDataset, self).__init__(user_ids, original_files, transform, augmentations,
                                                     augmented_files, verbose)

    def get_input_features(self, idx):
        input_features = self.participants[idx].heavy_cough.MFCCs
        input_features = self.z_normalize(input_features)
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
                 verbose=True):
        self.n_channels = 3
        self.n_timesteps = 224
        self.frequency_resolution = 224
        super(ResnetLogmelDataset, self).__init__(user_ids, original_files, transform,
                                                  augmentations, augmented_files, verbose)

    def get_input_features(self, idx):
        input_features = self.participants[idx].heavy_cough.logmel
        return input_features

    @staticmethod
    def get_feature_statistics():
        # do nothing - no normalization
        return 0., 1.
