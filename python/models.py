import torch
from torch import nn
import torch.nn.functional as F
TIMESTEPS = 259
MFCC_BINS = 15


class BrogrammersModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (TIMESTEPS, MFCC_BINS, 1)
        n_filters = 64

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2)
        self.batch_norm2d = nn.BatchNorm2d(n_filters)
        self.dense1 = nn.Linear(in_features=(TIMESTEPS-4) * (MFCC_BINS - 4) * n_filters, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=1)


    def forward(self, input_data):
        x = F.relu(self.conv1(input_data))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm2d(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=0.3)
        x = torch.sigmoid(self.dense3(x))
        return x



class BrogrammersSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (TIMESTEPS, MFCC_BINS, 1)
        n_filters = 64

        self.model = nn.Sequential(
            # valid padding means that it is NOT padded --> it looses size
            # a 3x3 kernel with a stride of 1 removes 2 "pixels" from each dimension
            # --> 431x15 shrink down to 429x13
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, padding="valid"),
            nn.ReLU(),
            # standard for MaxPool2d = no padding
            # kernel size = 2 = 2x2 and stride = 1 shrinks down each dimension by 1
            # --> 429x13 shrink down to 428x12
            nn.MaxPool2d(kernel_size=2, stride=1),
            # kernel size = 2 = 2x2 and stride = 1 shrinks down each dimension by 1
            # --> 428x12 shrink down to 427x11
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, padding="valid"),
            nn.ReLU(),
            # learnable parameters = "scale and shift"
            # --> n_filters are scaled with one parameter and shifted by a learnable bias term
            # --> n_weights = n_kernels (input from last layer) * 2 (bias for each kernel) = 64 * 2 = 128
            nn.BatchNorm2d(num_features=n_filters),
            # start_dim=1 means we do not flatten the "batch" dimension and start flattening the data for each sample
            # 32x64x11x429 --> 32x(64*11*429) = 32x300608 because 32 is the batch size meaning we have
            # 32 samples we do not want to combine
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=(TIMESTEPS-4) * (MFCC_BINS - 4) * n_filters, out_features=256, bias=True),
            nn.ReLU(),
            # TODO add kernel, bias and activity regularizers???
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        prediction = self.model(input_data)
        return prediction
