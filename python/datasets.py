import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from torchvision import transforms


def get_feature_statistics():
    # why do I have to convert it to float32???
    mu = np.load("data/Coswara_processed/pickles/mean_cough_heavy_15MFCCs.npy")
    mu = np.expand_dims(mu, axis=1).astype("float32")
    sigma = np.load("data/Coswara_processed/pickles/stds_cough_heavy_15MFCCs.npy")
    sigma = np.expand_dims(sigma, axis=1).astype("float32")
    return mu, sigma


class CustomDataset(Dataset):
    def __init__(self, user_ids, transform=None, augmentations=None, augmented_files=None, verbose=True):
        self.user_ids = user_ids
        self.n_timesteps = 259

        # if transform is not None:
        #     self.transform = transform
        # else:
        #     self.transform = transforms.ToTensor()

        self.transform = transform if transform is not None else transforms.ToTensor()
        self.augmentations = augmentations
        # self.use_augmentations = True
        # with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "rb") as f:
        # with open("data/Coswara_processed/pickles/participant_objects_new.pickle", "rb") as f:
        with open("data/Coswara_processed/pickles/participants_validLabelsOnly.pickle", "rb") as f:
            self.participants = pickle.load(f)
        if augmented_files is not None:
            for pickle_file in augmented_files:
                with open(f"data/Coswara_processed/pickles/{pickle_file}", "rb") as f:
                    self.participants += pickle.load(f)

        self.participants = [part for part in self.participants if part.id in user_ids]

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
        # standard is just ToTensor() --> do not specify "None" as transform. Otherwise, the program will crash
        input_features = self.transform(input_features)

        if self.augmentations is not None:
            input_features = self.augmentations(input_features)

        # add padding if there are too few timesteps. Although there should not be... this is probably because of
        # timestretch in the time domain data augmentation
        # TODO fix this data augmentation error (or just leave it with the fix here)

        n_timesteps = input_features.shape[2]
        if n_timesteps < self.n_timesteps:
            input_features = torch.nn.functional.pad(input_features, (0, self.n_timesteps - n_timesteps))

        input_features = input_features[:, :, :self.n_timesteps]
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

