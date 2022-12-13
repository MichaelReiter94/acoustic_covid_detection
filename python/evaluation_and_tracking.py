import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd


class ModelEvaluator:
    def __init__(self):
        self.training_loss = []
        self.evaluation_loss = []

        self.batch_training_loss = np.array([])
        self.batch_evaluation_loss = np.array([])
        self.mode = "train"
        self.epoch = 0
        # add "verbose" attribute which can print something everytime a batch loss or epoch loss was added

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def track_loss(self, current_loss: float, mode=None):
        if mode is not None:
            self.mode = mode

        if self.mode == "train":
            self.batch_training_loss = np.append(self.batch_training_loss, current_loss)
        elif self.mode == "eval":
            self.batch_evaluation_loss = np.append(self.batch_evaluation_loss, current_loss)
        else:
            raise ValueError

    def epoch_has_finished(self):
        if len(self.batch_training_loss) > 0:
            self.training_loss.append(self.batch_training_loss)
        if len(self.batch_evaluation_loss) > 0:
            self.evaluation_loss.append(self.batch_evaluation_loss)
        self.batch_training_loss = np.array([])
        self.batch_evaluation_loss = np.array([])
        self.epoch += 1

    def get_loss(self, granularity="epoch", mode=None):
        if mode is not None:
            self.mode = mode

        temp_loss = self.training_loss if self.mode == "train" else self.evaluation_loss

        if granularity == "epoch":
            loss = np.array([np.mean(epoch) for epoch in temp_loss])
        elif granularity == "batch":
            loss = np.array([])
            for epoch in temp_loss:
                loss = np.append(loss, epoch)
        else:
            raise ValueError
        return loss


def get_accuracy(labels, predictions, threshold=0.5):
    labels_bool = labels > threshold
    # print(f"\n\nactual positive labels: {np.sum(labels_bool)}")
    predicted_labels = predictions > threshold
    # print(f"predicted as positive: {np.sum(predicted_labels)}")
    n_correctly_predicted = np.sum(predicted_labels == labels_bool) / len(predictions)
    # print(f"correctly predicted: {round(n_correctly_predicted*100, 2)}%")
    return n_correctly_predicted


def get_confusion_matrix_parameters(labels, predictions, threshold=0.5):
    predictions_bool = predictions > threshold
    labels_bool = labels > threshold
    confusion_mat = confusion_matrix(labels_bool, predictions_bool)
    # print(pd.DataFrame(np.flip(confusion_mat), columns=["Pred. [+]", "Pred. [-]"], index=["Actual [+]", "Actual [-]"]))
    # returns parameters in the following order: tn, fp, fn, tp
    return confusion_mat.ravel()


def get_rates_from_confusion_matrix(confusion_mat):
    TN, FP, FN, TP = confusion_mat
    total_negatives = TN+FP
    total_positives = FN+TP
    tpr = round(TP/total_positives*100, 2) # also known as recall
    tnr = round(TN/total_negatives*100, 2)
    fnr = round(FN/total_positives*100, 2)
    fpr = round(FP/total_negatives*100, 2)
    precision = round(TP/(TP+FP)*100, 2)
    # print(pd.DataFrame(dict(tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, precision=precision), index=[0]))
    return tpr, fpr, tnr, fnr, precision



class IntraEpochMetricsTracker:
    def __init__(self):
        # self.aucroc = None
        # self.auc_preision_recall = None
        self.metrics = ["loss", "accuracy", "tpr", "fpr", "tnr", "fnr", "precision", "recall", "F1", "auc-roc",
                        "auc-prec-recall"]
        self.loss = np.array([])
        # self.accuracy = np.array([])
        self.labels = np.array([])
        self.predictions = np.array([])

    def reset(self):
        self.loss = np.array([])
        # self.accuracy = np.array([])
        self.labels = np.array([])
        self.predictions = np.array([])

    def add_metrics(self, loss, labels, predictions):
        predictions = torch.sigmoid(predictions.cpu().detach()).numpy()
        labels = labels.cpu().detach().numpy()
        self.predictions = np.append(self.predictions, predictions)
        self.loss = np.append(self.loss, float(loss))
        self.labels = np.append(self.labels, labels)
        # self.accuracy =  np.append(self.accuracy, get_accuracy(labels, predictions))
        # get_accuracy(labels, predictions)
        # conf_mat = get_confusion_matrix_parameters(labels, predictions)
        # get_rates_from_confusion_matrix(conf_mat)


    def get_epoch_metrics(self):
        # print("##########################################################################\n")
        confusion_mat = get_confusion_matrix_parameters(self.labels, self.predictions)
        tpr, fpr, tnr, fnr, precision = get_rates_from_confusion_matrix(confusion_mat)
        recall = tpr
        f1_score = 2*(precision*recall)/(precision+recall)
        metric_dict = dict(loss=np.mean(self.loss),
                           accuracy=get_accuracy(self.labels, self.predictions),
                           confusion_mat=confusion_mat,
                           tpr=tpr,
                           fpr=fpr,
                           tnr=tnr,
                           fnr=fnr,
                           auc_roc=self.get_aucroc(),
                           tpr_at_95=self.get_tpr_at_sensitivity(0.95),
                           auc_prec_recall=self.get_auc_precision_recall(),
                           precision=precision,
                           f1_score=f1_score)
        # print("\n##########################################################################")
        return metric_dict


    def get_aucroc(self):
        fpr, tpr, thresh = roc_curve(self.labels, self.predictions)
        aucroc = auc(fpr, tpr)
        return aucroc

    def get_auc_precision_recall(self):
        precision, recall, _ = precision_recall_curve(self.labels, self.predictions)
        auc_preision_recall = auc(recall, precision)
        return auc_preision_recall

    def get_tpr_at_sensitivity(self, sensitivity_target=0.95):
        fpr, tpr, thresh = roc_curve(self.labels, self.predictions)
        sensitivity = 1 - fpr
        distance_to_target_sensitivity = np.abs(sensitivity - sensitivity_target)
        closest_index = np.argmin(distance_to_target_sensitivity)
        return tpr[closest_index]
