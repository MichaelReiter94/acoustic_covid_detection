import scipy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from utils.utils import ResidualInstanceNorm2d, ResidualBatchNorm2d
from torchinfo import summary
from tkinter.filedialog import askopenfilename
import tkinter as tk
from torch import cuda


def get_bag_statistics(y, batch_size, bag_size):
    y = y.view(batch_size, bag_size)
    print(f"min: {round(float(y.min(dim=1)[0].mean().detach()), 1)}  |  "
          f"max: {round(float(y.max(dim=1)[0].mean().detach()), 1)}")
    # eps = 1e-6
    eps = 0
    mu = y.mean(dim=1)
    diff = y.t() - mu
    sigma = torch.pow(torch.mean(torch.pow(diff, 2.0), dim=0), 0.5) + eps
    # var = torch.mean(torch.pow(diff, 2.0), dim=0)
    z_scores = diff / sigma
    # print(f"sigma: {sigma.min()}  --  z_scores: {z_scores.max()}")
    skew = torch.mean(torch.pow(z_scores, 3), dim=0)
    kurtoses = torch.mean(torch.pow(z_scores, 4), dim=0)
    median, _ = y.median(dim=1)
    minimum, _ = y.min(dim=1)
    maximum, _ = y.max(dim=1)
    pos_area = torch.relu(y).sum(dim=1)
    neg_area = torch.relu(y * -1.0).sum(dim=1)

    # bag_statistics = torch.stack([mu, median, sigma, minimum, maximum, skew, kurtoses]).t()
    bag_statistics = torch.stack([mu, median, sigma, minimum, maximum, skew, kurtoses, pos_area, neg_area]).t()
    # bag_statistics = torch.stack([mu, median, var, minimum, maximum]).t()
    # bag_statistics = torch.nan_to_num(bag_statistics, 0.0, 0.0, 0.0)  # replaces nan, and infinities with zeros
    if torch.isnan(bag_statistics).sum() > 0:
        print("bag statistics contains nan:")
        print(bag_statistics)
    return bag_statistics


class PredictionLevelMILSingleGatedLayer(nn.Module):
    def __init__(self, n_neurons, dropout, last_layer):
        super().__init__()
        self.n_bag_statistics = 9
        self.n_hidden_attention = n_neurons
        self.dropout = dropout
        if last_layer is None:
            self.resnet_out_features = 512
            self.binary_classification_layer = nn.Sequential(
                nn.Linear(self.resnet_out_features, 1),
            )
        else:
            self.binary_classification_layer = last_layer

        self.attention_V = nn.Sequential(
            nn.BatchNorm1d(self.n_bag_statistics),
            nn.Linear(self.n_bag_statistics, self.n_hidden_attention),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.BatchNorm1d(self.n_bag_statistics),
            nn.Linear(self.n_bag_statistics, self.n_hidden_attention),
            nn.Sigmoid(),
        )
        self.attention_out = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.n_hidden_attention, 1)
        )

    def forward(self, y, batch_size, bag_size):
        if torch.isnan(y).sum() > 0:
            print("prediction y includes nan")
            print(y)
        y = self.binary_classification_layer(y.squeeze())

        bag_statistics = get_bag_statistics(y, batch_size, bag_size)
        A_V = self.attention_V(bag_statistics)
        A_U = self.attention_U(bag_statistics)
        y_pred = self.attention_out(A_V * A_U)  # element wise multiplication
        return y_pred


class PredictionLevelMILDoubleDenseLayer(nn.Module):
    def __init__(self, n_neurons, dropout, last_layer):
        super().__init__()
        self.n_bag_statistics = 9
        self.n_hidden_attention = n_neurons
        self.dropout = dropout
        if last_layer is None:
            self.resnet_out_features = 512
            self.binary_classification_layer = nn.Sequential(
                nn.Linear(self.resnet_out_features, 1),
            )
        else:
            self.binary_classification_layer = last_layer

        self.mil_net = nn.Sequential(
            nn.BatchNorm1d(self.n_bag_statistics),
            nn.Linear(self.n_bag_statistics, self.n_hidden_attention),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.n_hidden_attention, self.n_hidden_attention),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.n_hidden_attention, 1)
        )

    def forward(self, y, batch_size, bag_size):
        y = self.binary_classification_layer(y.squeeze())
        bag_statistics = get_bag_statistics(y, batch_size, bag_size)
        y_pred = self.mil_net(bag_statistics)  # element wise multiplication
        return y_pred


class FeatureLevelMIL(nn.Module):
    def __init__(self, n_neurons, dropout, last_layer):
        super().__init__()
        self.n_hidden_attention = n_neurons
        self.dropout = dropout
        self.resnet_out_features = 512

        self.attention_V = nn.Sequential(
            nn.Linear(self.resnet_out_features, self.n_hidden_attention),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.resnet_out_features, self.n_hidden_attention),
            nn.Sigmoid(),
        )
        self.attention_out = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.n_hidden_attention, 1)
        )

        if last_layer is None:
            # self.feature_layer = nn.Sequential(
            #     nn.Linear(self.resnet_out_features, 1),
            # )
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=self.resnet_out_features, out_features=1)
                # nn.Linear(in_features=128, out_features=1)
            )
        else:
            self.output_layer = last_layer

    def forward(self, y, batch_size, bag_size):
        # y = self.feature_layer(y.squeeze())

        y = y.squeeze()
        # batchsize*bagsize x 512
        A_V = self.attention_V(y)
        A_U = self.attention_U(y)
        attention_coef = self.attention_out(A_V * A_U)  # element wise multiplication
        attention_coef = attention_coef.view(batch_size, bag_size, 1)
        attention_coef = F.softmax(attention_coef, dim=1)
        # batchsize x bagsize x 1
        print(f"min: {round(float(attention_coef.min(dim=1)[0].mean().detach()) * 100, 1)}  "
              f"|  max: {round(float(attention_coef.max(dim=1)[0].mean().detach()) * 100, 1)}")

        x_combined_bag = y.view(batch_size, bag_size, self.resnet_out_features) * attention_coef
        # y = y.view(batch_size, bag_size, 1)
        # [batchsize x bagsize x 512] * [batchsize x bagsize x 1]
        x_combined_bag = x_combined_bag.mean(dim=1)
        # [batch_size x 512]
        y_pred = self.output_layer(x_combined_bag)
        # [batch_size x 1]
        return y_pred


class FeatureLevelMILExtraFeatureLayer(nn.Module):
    def __init__(self, n_features, n_neurons, dropout):
        super().__init__()
        self.n_hidden_attention = n_neurons
        self.dropout = dropout
        self.resnet_out_features = 512
        self.n_features = n_features

        self.feature_layer = nn.Sequential(
            nn.Linear(self.resnet_out_features, self.n_features),
            nn.Dropout(p=self.dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.n_features, self.n_hidden_attention),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.n_features, self.n_hidden_attention),
            nn.Sigmoid(),
        )
        self.attention_out = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.n_hidden_attention, 1)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.n_features, 1)
        )

    def forward(self, y, batch_size, bag_size):
        y = self.feature_layer(y.squeeze())
        # batchsize*bagsize x 512

        A_V = self.attention_V(y)
        A_U = self.attention_U(y)
        attention_coef = self.attention_out(A_V * A_U)  # element wise multiplication
        attention_coef = attention_coef.view(batch_size, bag_size, 1)
        attention_coef = F.softmax(attention_coef, dim=1)
        # batchsize x bagsize x 1
        print(f"min: {round(float(attention_coef.min(dim=1)[0].mean().detach()) * 100, 1)}  "
              f"|  max: {round(float(attention_coef.max(dim=1)[0].mean().detach()) * 100, 1)}")

        x_combined_bag = y.view(batch_size, bag_size, self.n_features) * attention_coef
        # y = y.view(batch_size, bag_size, 1)
        # [batchsize x bagsize x 512] * [batchsize x bagsize x 1]
        x_combined_bag = x_combined_bag.mean(dim=1)
        # [batch_size x 512]
        y_pred = self.output_layer(x_combined_bag)
        # [batch_size x 1]
        return y_pred


class BrogrammersModel(nn.Module):
    def __init__(self):
        super().__init__()
        TIMESTEPS = 259
        MFCC_BINS = 15
        self.input_size = (1, MFCC_BINS, TIMESTEPS)

        n_filters1 = 64
        n_filters2 = 32
        torch.manual_seed(9876543210)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters1, kernel_size=3)
        # TODO the stride of the max-pool layer might be wrong... (not specified in paper but no arguments means
        #  possibly that the stride = the kernel size
        # self.max-pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=n_filters1, out_channels=n_filters2, kernel_size=2)
        self.batch_norm2d = nn.BatchNorm2d(n_filters2)
        self.dense1 = nn.Linear(in_features=((TIMESTEPS - 2) // 2 - 1) * ((MFCC_BINS - 2) // 2 - 1) * n_filters2,
                                out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=1)
        print("loading brogrammers module/class based model")

    def forward(self, input_data):
        x = F.relu(self.conv1(input_data))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm2d(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dense1(x))
        # TODO check if dropout is ignored/turned off during model.eval() mode
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.dense3(x)
        # no sigmoid activation because the now used BCELossWithLogits class has the activation function included (
        # which improves numerical stability/precision) Also this class has the possibility to add a weighting to the
        # two classes to address class imbalance!! x = torch.sigmoid(x)
        return x


class BrogrammersSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        TIMESTEPS = 259
        MFCC_BINS = 15
        self.input_size = (1, MFCC_BINS, TIMESTEPS)
        n_filters = 64

        torch.manual_seed(9876543210)
        self.model = nn.Sequential(
            # valid padding means that it is NOT padded --> it looses size
            # a 3x3 kernel with a stride of 1 removes 2 "pixels" from each dimension
            # --> 431x15 shrink down to 429x13
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, padding="valid"),
            nn.ReLU(),
            # standard for MaxPool2d = no padding
            # kernel size = 2 = 2x2 and stride = 1 shrinks down each dimension by 1
            # --> 429x13 shrink down to 428x12

            # or with stride=kernel size:
            # 429//2 x 13//2 = 214x6
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
            nn.Linear(in_features=(TIMESTEPS - 4) * (MFCC_BINS - 4) * n_filters, out_features=256, bias=True),
            nn.ReLU(),
            # TODO add kernel, bias and activity regularizers???
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        print("loading brogrammers sequential model")

    def forward(self, input_data):
        prediction = self.model(input_data)
        return prediction


def get_resnet18(dropout_p=0.0, add_residual_layers=False, FREQUNCY_BINS=224, TIMESTEPS=224, N_CHANNELS=1,
                 load_from_disc=False):
    my_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    my_model.input_size = (N_CHANNELS, FREQUNCY_BINS, TIMESTEPS)
    device = "cuda" if cuda.is_available() else "cpu"

    # my_model = resnet18()  # not pretrained
    # for param in my_model.parameters():
    #     param.requires_grad = False

    ########################### change number of finput channels from 3 to 1 #################################
    # (either use the pretrained weights of 1 channel or average them over all 3)

    if N_CHANNELS == 1:
        weights = my_model.conv1.weight
        # weights_mean = weights.mean(dim=1).unsqueeze(dim=1)
        weights_single_color = weights[:, 0, :, :].unsqueeze(dim=1)

        my_model.conv1 = nn.Conv2d(N_CHANNELS, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        # my_model.conv1.weight = nn.Parameter(weights_mean)
        my_model.conv1.weight = nn.Parameter(weights_single_color)

    ############################  add residual normalization layers to resnet  ################################
    gamma = 0.5
    if add_residual_layers:
        layers = summary(my_model, input_size=(1, N_CHANNELS, FREQUNCY_BINS, TIMESTEPS))
        layers = str(layers).split("\n")
        layers = [layer for layer in layers if "BatchNorm2d: 3" in layer]
        fdims = [int(layer.split("[")[1].split("]")[0].split(",")[2]) for layer in layers]

        # for every major layer (which is hardcoded for resnet18) we iterate over each BasicBlock in this layer and add
        # a residual normalization layer after the batch norm. such a basic block (for resnet18) has 2 batch norm layers
        # might need changes for resnet50 etc
        layers = [my_model.layer1, my_model.layer2, my_model.layer3, my_model.layer4]
        counter = 0
        for layer in layers:
            for i in range(len(layer)):
                layer[i].bn1 = nn.Sequential(
                    layer[i].bn1,
                    # ResidualInstanceNorm2d(num_features=layer[i].conv1.out_channels, gamma=gamma,
                    #                        gamma_is_learnable=True)
                    ResidualInstanceNorm2d(num_features=fdims[counter], gamma=gamma,
                                           affine=True, track_running_stats=False, gamma_is_learnable=True)
                    # ResidualBatchNorm2d(num_features=fdims[counter], gamma=gamma,
                    #                     affine=True, track_running_stats=True, gamma_is_learnable=True)
                )
                counter += 1
                # layer[i].bn2 = nn.Sequential(
                #     layer[i].bn2,
                #     # ResidualInstanceNorm2d(num_features=layer[i].conv2.out_channels, gamma=gamma,
                #     #                        gamma_is_learnable=True)
                #     ResidualInstanceNorm2d(num_features=fdims[counter], gamma=gamma,
                #                            affine=True, track_running_stats=False, gamma_is_learnable=True)
                # )
                counter += 1

    ############################  add dropouts (spatial and regular) to resnet  ################################
    if dropout_p > 0:
        my_model.layer1 = nn.Sequential(*my_model.layer1, nn.Dropout2d(p=dropout_p))
        my_model.layer2 = nn.Sequential(*my_model.layer2, nn.Dropout2d(p=dropout_p))
        my_model.layer3 = nn.Sequential(*my_model.layer3, nn.Dropout2d(p=dropout_p))
        my_model.layer4 = nn.Sequential(*my_model.layer4, nn.Dropout2d(p=dropout_p))
        my_model.avgpool = nn.Sequential(my_model.avgpool, nn.Dropout(p=dropout_p))

    n_features = my_model.fc.in_features
    my_model.fc = nn.Linear(n_features, 1)

    if isinstance(load_from_disc, bool):
        if load_from_disc:
            try:
                # TODO if load_from_disc is a path, directly load it. no browsing for a file (if the path exists)
                # if the path does not exist probably raise an error
                window = tk.Tk()

                path = askopenfilename(initialdir=f"data/Coswara_processed/models")
                print(f"Path to pretrained model:\n{path}")
                # path = f"data/Coswara_processed/models/{model_name}/model.pth"
                my_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
                window.destroy()

                # for param in my_model.parameters():
                #     param.requires_grad = False

                print("model weights loaded from disc")
            except FileNotFoundError:
                print("no saved model parameters found")
    elif isinstance(load_from_disc, str):
        path = load_from_disc
        print(f"Path to pretrained model:\n{path}")
        my_model.load_state_dict(torch.load(path, map_location=torch.device(device)))

    return my_model


def get_resnet50():
    TIMESTEPS = 224
    FREQUNCY_BINS = 224
    N_CHANNELS = 3
    my_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    my_model.input_size = (N_CHANNELS, FREQUNCY_BINS, TIMESTEPS)
    # for param in my_model.parameters():
    #     param.requires_grad = False
    n_features = my_model.fc.in_features
    my_model.fc = nn.Linear(n_features, 1)
    return my_model


class BrogrammersMIL(nn.Module):
    def __init__(self, n_hidden_attention=32):
        super().__init__()
        TIMESTEPS = 259
        MFCC_BINS = 15
        self.input_size = (1, MFCC_BINS, TIMESTEPS)

        n_filters1 = 64
        n_filters2 = 32
        torch.manual_seed(9876543210)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=n_filters1, out_channels=n_filters2, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_filters2),
            nn.Flatten(start_dim=1)
        )

        n_linear_params = ((TIMESTEPS - 2) // 2 - 1) * ((MFCC_BINS - 2) // 2 - 1) * n_filters2
        self.n_hidden_attention = n_hidden_attention
        n_linear_out = self.n_hidden_attention

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=n_linear_params, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=self.n_hidden_attention),
            nn.ReLU(),
        )

        # # regular version of the MIL attention mechanism
        # self.mil_attention = nn.Sequential(
        #     nn.Linear(n_linear_out, self.n_hidden_attention),
        #     nn.Tanh(),
        #     nn.Linear(self.n_hidden_attention, 1),
        # )

        # gated attention mechanism as seen in Attention-based Deep Multiple Instance Learning by
        # Maximilian Ilse, Jakub M. Tomczak, Max Welling
        self.attention_V = nn.Sequential(
            nn.Linear(n_linear_out, self.n_hidden_attention),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(n_linear_out, self.n_hidden_attention),
            nn.Sigmoid()
        )
        self.attention_out = nn.Linear(self.n_hidden_attention, 1)

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=n_linear_out, out_features=1)
            # nn.Linear(in_features=128, out_features=1)
        )
        print("loading Multiple Instance Learning model based on brogrammers")

    def forward(self, x):
        # # first dimenstion will be the batch size which will be set to 1. This is why it can be eliminated.
        # # the elements within a bag (dimension 1) will instead be kind of treated as batch size
        # x = x.squeeze(0)
        batch_size, bag_size = x.shape[0], x.shape[1]
        feature_size = x.shape[2:]
        x = x.view(batch_size * bag_size, *feature_size)

        x = self.conv_layers(x)
        x = self.linear_layers(x)

        ###################################################     MIL    #################################################
        # # regular version of the MIL attention mechanism
        # attention_coef = self.mil_attention(x)
        # gated attention mechanism
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        attention_coef = self.attention_out(A_V * A_U)  # element wise multiplication

        # needed for both MIL mechanisms
        n_features = x.shape[-1]
        attention_coef = attention_coef.view(batch_size, bag_size, 1)
        attention_coef = F.softmax(attention_coef, dim=1)
        ################################################################################################################
        x_combined_bag = x.view(batch_size, bag_size, n_features) * attention_coef
        x_combined_bag = x_combined_bag.mean(dim=1)

        y_pred = self.output_layer(x_combined_bag)
        # no sigmoid activation because the now used BCELossWithLogits class has the activation function included (
        # which improves numerical stability/precision) Also this class has the possibility to add a weighting to the
        # two classes to address class imbalance!! if BCELossWithLogits is not used uncomment the following line:
        # y_pred = torch.sigmoid(y_pred)
        return y_pred


class Resnet18MILOld(nn.Module):
    def __init__(self, n_hidden_attention=32, add_dropouts=True, add_residual_layers=False, F=224, T=224, C=1):
        super().__init__()
        # self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = get_resnet18(dropout_p=add_dropouts, add_residual_layers=add_residual_layers,
                                   FREQUNCY_BINS=F, TIMESTEPS=T, N_CHANNELS=C)
        # TIMESTEPS = 224
        # FREQUNCY_BINS = 224
        # N_CHANNELS = 3
        # my_model.input_size = (N_CHANNELS, FREQUNCY_BINS, TIMESTEPS)
        n_features = self.resnet.fc.in_features
        n_linear_out = 64
        self.resnet.fc = nn.Linear(n_features, 1)

        self.n_hidden_attention = n_hidden_attention
        n_bag_statistics = 7
        #
        # # regular version of the MIL attention mechanism
        # self.mil_attention = nn.Sequential(
        #     nn.Linear(n_bag_statistics, self.n_hidden_attention),
        #     nn.ReLU(),
        #     nn.Linear(self.n_hidden_attention, self.n_hidden_attention),
        #     nn.ReLU(),
        #     nn.Linear(self.n_hidden_attention, 1),
        # )

        # gated attention mechanism as seen in Attention-based Deep Multiple Instance Learning by
        # Maximilian Ilse, Jakub M. Tomczak, Max Welling
        self.attention_V = nn.Sequential(
            nn.Linear(n_bag_statistics, self.n_hidden_attention),
            nn.Tanh(),
            nn.Dropout(p=0.25)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(n_bag_statistics, self.n_hidden_attention),
            nn.Sigmoid(),
        )
        self.attention_out = nn.Linear(self.n_hidden_attention, 1)

    def forward(self, x):
        batch_size, bag_size = x.shape[0], x.shape[1]
        feature_size = x.shape[2:]
        x = x.view(batch_size * bag_size, *feature_size)
        y = self.resnet(x)
        y = y.view(batch_size, bag_size)

        mu = y.mean(dim=1)
        diff = y.t() - mu
        sigma = torch.pow(torch.mean(torch.pow(diff, 2.0), dim=0), 0.5)
        z_scores = diff / sigma
        skew = torch.mean(torch.pow(z_scores, 3), dim=0)
        kurtoses = torch.mean(torch.pow(z_scores, 4), dim=0)
        median, _ = y.median(dim=1)
        minimum, _ = y.min(dim=1)
        maximum, _ = y.max(dim=1)
        bag_statistics = torch.stack([mu, median, sigma, minimum, maximum, skew, kurtoses]).t()

        ###################################################     MIL    #################################################
        # # regular version of the MIL attention mechanism
        # y_pred = self.mil_attention(bag_statistics)
        # gated attention mechanism
        A_V = self.attention_V(bag_statistics)
        A_U = self.attention_U(bag_statistics)
        y_pred = self.attention_out(A_V * A_U)  # element wise multiplication

        return y_pred


class Resnet18MIL(nn.Module):
    def __init__(self, n_hidden_attention=32, dropout_p=0.0, add_residual_layers=False, F=224, T=224, C=1,
                 load_from_disc=False):
        super().__init__()
        self.resnet = get_resnet18(dropout_p=dropout_p, add_residual_layers=add_residual_layers,
                                   FREQUNCY_BINS=F, TIMESTEPS=T, N_CHANNELS=C, load_from_disc=load_from_disc)
        # trim off the last dense layer [512 --> 1] to be able to add the MIL networks in all variations

        #  # uncomment this if you want to have the last dense layer from the resnet in the mil network instead.
        #  # this loses all pretrained weights
        last_layer = list(self.resnet.children())[-1]
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # self.mil_net = PredictionLevelMILSingleGatedLayer(n_neurons=n_hidden_attention, dropout=dropout_p,
        #                                                   last_layer=last_layer)
        # self.mil_net = PredictionLevelMILDoubleDenseLayer(n_neurons=n_hidden_attention, dropout=dropout_p,
        #                                                   last_layer=last_layer)
        self.mil_net = FeatureLevelMIL(n_neurons=n_hidden_attention, dropout=dropout_p, last_layer=last_layer)
        # self.mil_net = FeatureLevelMILExtraFeatureLayer(n_features=n_hidden_attention, n_neurons=n_hidden_attention,
        #                                                 dropout=dropout_p)
        self.batch_size, self.bag_size, self.feature_size = None, None, None

    def forward(self, x):
        x = self.mil_reshape(x)
        y = self.resnet(x)
        y = self.mil_net(y, self.batch_size, self.bag_size)
        return y

    def mil_reshape(self, x):
        # gets the shape of batch size and bag size and feature size and returns the data in the shape
        # [batch_size*bag_size x channels x FreqBins x TimeSteps] to be able to process it with the resnet
        self.batch_size, self.bag_size, self.feature_size = x.shape[0], x.shape[1], x.shape[2:]
        x = x.view(self.batch_size * self.bag_size, *self.feature_size)
        return x


class PredLevelMIL(nn.Module):  # brogrammers
    def __init__(self, n_hidden_attention=32):
        super().__init__()
        TIMESTEPS = 259
        MFCC_BINS = 15
        self.input_size = (1, MFCC_BINS, TIMESTEPS)

        n_filters1 = 64
        n_filters2 = 32
        torch.manual_seed(9876543210)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=n_filters1, out_channels=n_filters2, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_filters2),
            nn.Flatten(start_dim=1)
        )

        n_linear_params = ((TIMESTEPS - 2) // 2 - 1) * ((MFCC_BINS - 2) // 2 - 1) * n_filters2

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=n_linear_params, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1)
        )

        self.n_hidden_attention = n_hidden_attention
        n_bag_statistics = 7

        # # regular version of the MIL attention mechanism
        # self.mil_attention = nn.Sequential(
        #     nn.Linear(n_bag_statistics, self.n_hidden_attention),
        #     nn.ReLU(),
        #     # nn.Linear(self.n_hidden_attention, self.n_hidden_attention),
        #     # nn.ReLU(),
        #     nn.Linear(self.n_hidden_attention, 1),
        # )

        # gated attention mechanism as seen in Attention-based Deep Multiple Instance Learning by
        # Maximilian Ilse, Jakub M. Tomczak, Max Welling
        self.attention_V = nn.Sequential(
            nn.Linear(n_bag_statistics, self.n_hidden_attention),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(n_bag_statistics, self.n_hidden_attention),
            nn.Sigmoid()
        )
        self.attention_out = nn.Linear(self.n_hidden_attention, 1)

        print("loading Multiple Instance Learning model based on brogrammers")

    def forward(self, x):
        # x = x.squeeze(0)

        batch_size, bag_size = x.shape[0], x.shape[1]
        feature_size = x.shape[2:]
        x = x.view(batch_size * bag_size, *feature_size)

        # first dimenstion will be the batch size which will be set to 1. This is why it can be eliminated.
        # the elements within a bag (dimension 1) will instead be kind of treated as batch size

        x = self.conv_layers(x)
        y = self.linear_layers(x)

        y = y.view(batch_size, bag_size)

        # y = torch.sigmoid(y)
        # mu = y.mean()
        # sigma = torch.pow(torch.mean(torch.pow(y - mu, 2.0)), 0.5)
        # z_scores = (y - mu) / sigma
        # skew = torch.mean(torch.pow(z_scores, 3))
        # kurtoses = torch.mean(torch.pow(z_scores, 4))
        # bag_statistics = torch.stack([mu, y.median(), sigma, y.min(), y.max(), skew, kurtoses])

        mu = y.mean(dim=1)
        diff = y.t() - mu
        sigma = torch.pow(torch.mean(torch.pow(diff, 2.0), dim=0), 0.5)
        z_scores = diff / sigma
        skew = torch.mean(torch.pow(z_scores, 3), dim=0)
        kurtoses = torch.mean(torch.pow(z_scores, 4), dim=0)

        # bag_statistics = torch.stack([mu, y.median(), sigma, y.min(), y.max()])
        median, _ = y.median(dim=1)
        minimum, _ = y.min(dim=1)
        maximum, _ = y.max(dim=1)
        bag_statistics = torch.stack([mu, median, sigma, minimum, maximum, skew, kurtoses]).t()

        # y_pred = torch.stack([y.max()])
        ###################################################     MIL    #################################################
        # # regular version of the MIL attention mechanism
        # y_pred = self.mil_attention(bag_statistics)
        # gated attention mechanism
        A_V = self.attention_V(bag_statistics)
        A_U = self.attention_U(bag_statistics)
        y_pred = self.attention_out(A_V * A_U)  # element wise multiplication

        # needed for both MIL mechanisms
        # attention_coef = torch.transpose(attention_coef, 1, 0)
        # attention_coef = F.softmax(attention_coef, dim=1)
        ################################################################################################################

        # x_combined_bag = torch.mm(attention_coef, x)
        # y_pred = self.output_layer(x_combined_bag)
        # no sigmoid activation because the now used BCELossWithLogits class has the activation function included (
        # which improves numerical stability/precision) Also this class has the possibility to add a weighting to the
        # two classes to address class imbalance!! if BCELossWithLogits is not used uncomment the following line:
        # y_pred = torch.sigmoid(y_pred)
        return y_pred
