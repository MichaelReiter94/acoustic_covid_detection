import smtplib
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch


def send_mail(to_address, text, subject="This is no spam"):
    gmail = "hundreddaysofcodemr@gmail.com"
    # noinspection SpellCheckingInspection
    gmail_app_pw = "fvgxkqbqzeeogjkv"
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


class FocalLoss:
    def __init__(self, gamma, alpha=1, reduction: str = "none"):
        self.gamma = gamma
        if alpha == 1:
            self.alpha = -1
        else:
            self.alpha = alpha / (alpha + 1)
        self.reduction = reduction
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

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor):
        loss = sigmoid_focal_loss(predictions, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss

    def __str__(self):
        return str(f"sigmoid_focal_loss(gamma={self.gamma}, alpha={self.alpha})")

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    send_mail("michael.reiter94@gmail.com", "this is a test", subject="This is no spam")
