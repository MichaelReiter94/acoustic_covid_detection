import smtplib
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch
from torch import nn
from torch.nn import InstanceNorm2d


def send_mail(to_address, text, subject="This is no spam"):
    gmail = "hundreddaysofcodemr@gmail.com"
    # noinspection SpellCheckingInspection
    # gmail_app_pw = "fvgxkqbqzeeogjkv"
    gmail_app_pw = "htefvcehowtfqmie"  # new pw form 11.05.2023
    # print(gmail_app_pw)

    with smtplib.SMTP("smtp.gmail.com", port=587) as connection:
        connection.starttls()  # encryption
        connection.login(user=gmail, password=gmail_app_pw)
        connection.sendmail(from_addr=gmail,
                            to_addrs=to_address,
                            msg=f"Subject:{subject}\n\n{text}")


def jupyter_setup():
    import os

    # change working directory to main folder
    root_dir = "python"
    _, current_folder = os.path.split(os.getcwd())
    if current_folder != root_dir:
        os.chdir("../")

    # add path variables to avoid "audio module not found"-error
    path = os.environ.get("PATH")
    min_additional_path = "C:\\Users\\Michi\\Anaconda3\\envs\\python_v3-8\\Library\\bin;" \
                          "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\bin;"
    combined_path = min_additional_path + path
    os.environ["PATH"] = combined_path


def audiomentations_repr(audiomentation_compose):
    representation = ""
    if audiomentation_compose is None:
        return {}
    representation = {}
    for audiomentation in audiomentation_compose.transforms:
        name = audiomentation.__repr__().split(".")[-1].split(" ")[0]
        params = {"probability": audiomentation.p}
        if name == "AddGaussianNoise":
            params["min_amplitude"] = audiomentation.min_amplitude
            params["max_amplitude"] = audiomentation.max_amplitude
        elif name == "PitchShift":
            params["min_semitones"] = audiomentation.min_semitones
            params["max_semitones"] = audiomentation.max_semitones

        elif name == "TimeStretch":
            params["min_rate"] = audiomentation.min_rate
            params["max_rate"] = audiomentation.max_rate

        elif name == "Gain":
            params["min_gain_in_db"] = audiomentation.min_gain_in_db
            params["max_gain_in_db"] = audiomentation.max_gain_in_db

        # representation = f"{representation}\n{name} {params}"
        param_string = str(params).replace("{", "").replace("}", "")
        representation[name] = param_string
        # print(f"{name}{params}")
    return representation


class FocalLoss(nn.Module):
    def __init__(self, gamma, pos_weight=1, reduction: str = "mean", exclude_outliers=1):
        super(FocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        if self.pos_weight == 1:
            self.alpha = -1
        else:
            self.alpha = self.pos_weight / (self.pos_weight + 1)
        self.reduction = reduction
        self.exclude_n = exclude_outliers  # always exclude the outliers with the highest loss
        # self.norm_coef = self._calculate_normalizaiton_coef()
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002. 
        
        alpha (float): 
            Weighting factor to balance positive vs negative examples - negative label gets weighted with factor 1, 
            the positive labels with the factor alpha. So if you want the model to learn more from the positive 
            samples than you have to choose alpha > 1 (different to the original implementation of the focal loss but 
            consistent with BCELossWithLogits implementation of this label weighting) 
        gamma (float): 
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: ``2``. 
        reduction (string): 'none' | 'mean' | 'sum' 
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged. 
            'sum': The output will be summed. Default: 'none'. 
        Returns: Loss tensor with the reduction applied. 
        """

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        loss = sigmoid_focal_loss(predictions, targets, alpha=self.alpha, gamma=self.gamma, reduction="none")
        # loss = sigmoid_focal_loss(predictions, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        # loss = torch.nan_to_num(loss, 0.0, 0.0, 0.0)  # replaces nan, and infinities with zeros
        loss = self._exclude_outliers(loss)
        # print(loss)
        if self.reduction == "mean":
            loss = torch.nanmean(loss)
        elif self.reduction == "sum":
            loss = torch.nansum(loss)
        # return loss * self.norm_coef
        loss = torch.nan_to_num(loss, 0.0, 0.0, 0.0)  # replaces nan, and infinities with zeros
        return loss

    def __str__(self):
        return str(f"sigmoid_focal_loss(gamma={self.gamma}, pos_weight={self.pos_weight})")

    def __repr__(self):
        return self.__str__()

    def _exclude_outliers(self, loss: torch.Tensor):
        n = len(loss)
        if n >= 16:
            loss, _ = loss.sort()
            loss = loss[:n-self.exclude_n]
        return loss
    # def _calculate_normalizaiton_coef(self):
    #     # regular BCE loss has an area under curve of 1.412898063659668 (in the range -5, 5 before sigmoid)
    #     # to keep the results comparable the area under the resulting loss curve is normalized. Otherwise,
    #     # you automatially have a lower loss when choosing a higher gamma. this would also result in a pseudo higher
    #     # learning rate (weight update is proportional to lr and loss)
    #     predictions = torch.arange(-5, 5, 0.001)
    #     labels_pos = torch.ones(len(predictions))
    #     auc_loss = sigmoid_focal_loss(predictions, labels_pos, alpha=self.alpha, gamma=self.gamma, reduction="mean")
    #     return 1.412898063659668/auc_loss


# class ResidualInstanceNorm2d(nn.Module):
#     """
#     gamma: out = normalized(input) + gamma*input
#     Applies instance normalization over all time steps and channels of one instance. meaning for one sample in a
#     batch, each frequency gets normalized separately. To not lose too much information from normalizing each
#     frequency, there is a residual term. This means, the original input is added to the normalized input with a
#     factor gamma.
#     """
#     def __init__(self, gamma=1.0, eps=1e-5):
#         super(ResidualInstanceNorm2d, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#
#     def forward(self, x):
#         # for each sample, each frequency bin gets an average over all time steps and channels of this instance
#         x_mean = x.mean(dim=[-3, -1], keepdim=True)
#         x_var = x.var(dim=[-3, -1], keepdim=True)
#         x_normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
#
#         out = x_normalized + self.gamma * x  # add residual connection
#         return out

class ResidualInstanceNorm2d(nn.InstanceNorm2d):
    # using the pytorch implementation of the instance norm. this way, we can make use of affine and track running stats
    def __init__(self, num_features, gamma=1.0, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False,
                 gamma_is_learnable=False):
        super(ResidualInstanceNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum,
                                                     affine=affine, track_running_stats=track_running_stats)
        if gamma_is_learnable:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.gamma = torch.tensor(float(gamma))

    def forward(self, x):
        # permute dimensions because we want to apply the normalization across the time and channel dimensions
        x = x.permute(0, 2, 3, 1).contiguous()
        # apply instance normalization
        x = super(ResidualInstanceNorm2d, self).forward(x)
        # permute dimensions back
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + self.gamma * x
        return x


if __name__ == "__main__":
    send_mail("michael.reiter94@gmail.com", "this is a test", subject="This is no spam")
