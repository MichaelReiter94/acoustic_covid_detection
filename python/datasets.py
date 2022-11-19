import numpy as np
from torch.utils.data import Dataset
import pickle
import torch


def get_feature_statistics():
    # why do I have to convert it to float32???
    mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
    mu = np.expand_dims(mu, axis=1).astype("float32")
    sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
    sigma = np.expand_dims(sigma, axis=1).astype("float32")
    return mu, sigma


class CustomDataset(Dataset):
    def __init__(self, transform=None, augmentations=None, verbose=True):
        self.transform = transform
        self.augmentations = augmentations
        self.use_augmentations = True
        # with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "rb") as f:
        with open("data/Coswara_processed/pickles/participant_objects.pickle", "rb") as f:
            self.participants = pickle.load(f)
        self.drop_invalid_labels()
        self.drop_bad_audio()
        self.labels = np.array([int(participant.get_label()) for participant in self.participants])
        self.mu, self.sigma = get_feature_statistics()
        if verbose:
            self.summarize()

    def drop_invalid_labels(self):
        self.participants = [participant for participant in self.participants if participant.get_label() is not None]

    def drop_bad_audio(self):
        self.participants = [participant for participant in self.participants if
                             participant.meta_data["audio_quality_heavy_cough"] > 0.0]

    def __getitem__(self, idx):
        input_features = self.participants[idx].heavy_cough.MFCCs
        input_features = self.z_normalize(input_features)
        if self.transform:
            input_features = self.transform(input_features)
        if self.use_augmentations and self.augmentations is not None:
            input_features = self.augmentations(input_features)

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

