import scipy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights



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
        self.dense1 = nn.Linear(in_features=((TIMESTEPS-2)//2-1) * ((MFCC_BINS - 2)//2-1) * n_filters2, out_features=256)
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
        print("loading brogrammers sequential model")


    def forward(self, input_data):
        prediction = self.model(input_data)
        return prediction


def get_resnet18():
    TIMESTEPS = 224
    FREQUNCY_BINS = 224
    N_CHANNELS = 3
    my_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # my_model = resnet18()
    my_model.input_size = (N_CHANNELS, FREQUNCY_BINS, TIMESTEPS)
    # for param in my_model.parameters():
    #     param.requires_grad = False
    n_features = my_model.fc.in_features
    my_model.fc = nn.Linear(n_features, 1)
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

        n_linear_params = ((TIMESTEPS - 2)//2 - 1) * ((MFCC_BINS - 2)//2 - 1) * n_filters2
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
        # attentation_coef = self.mil_attention(x)
        # gated attention mechanism
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        attentation_coef = self.attention_out(A_V * A_U)  # element wise multiplication

        # needed for both MIL mechanisms
        n_features = x.shape[-1]
        attentation_coef = attentation_coef.view(batch_size, bag_size, 1)
        attentation_coef = F.softmax(attentation_coef, dim=1)
        ################################################################################################################
        x_combined_bag = x.view(batch_size, bag_size, n_features) * attentation_coef
        x_combined_bag = x_combined_bag.mean(dim=1)

        y_pred = self.output_layer(x_combined_bag)
        # no sigmoid activation because the now used BCELossWithLogits class has the activation function included (
        # which improves numerical stability/precision) Also this class has the possibility to add a weighting to the
        # two classes to address class imbalance!! if BCELossWithLogits is not used uncomment the following line:
        # y_pred = torch.sigmoid(y_pred)
        return y_pred



class Resnet18MIL(nn.Module):
    def __init__(self, n_hidden_attention=32):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
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
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(n_bag_statistics, self.n_hidden_attention),
            nn.Sigmoid()
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




class PredLevelMIL(nn.Module):
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

        n_linear_params = ((TIMESTEPS - 2)//2 - 1) * ((MFCC_BINS - 2)//2 - 1) * n_filters2

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
        # attentation_coef = torch.transpose(attentation_coef, 1, 0)
        # attentation_coef = F.softmax(attentation_coef, dim=1)
        ################################################################################################################

        # x_combined_bag = torch.mm(attentation_coef, x)
        # y_pred = self.output_layer(x_combined_bag)
        # no sigmoid activation because the now used BCELossWithLogits class has the activation function included (
        # which improves numerical stability/precision) Also this class has the possibility to add a weighting to the
        # two classes to address class imbalance!! if BCELossWithLogits is not used uncomment the following line:
        # y_pred = torch.sigmoid(y_pred)
        return y_pred

