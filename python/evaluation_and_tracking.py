import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


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


class IntraEpochMetricsTracker:
    def __init__(self):
        # self.aucroc = None
        # self.auc_preision_recall = None
        self.loss = np.array([])
        self.accuracy = np.array([])
        self.labels = np.array([])
        self.predictions = np.array([])

    def reset(self):
        self.loss = np.array([])
        self.accuracy = np.array([])
        self.labels = np.array([])
        self.predictions = np.array([])

    def add_metrics(self, loss, accuracy, labels, predictions):
        self.loss = np.append(self.loss, float(loss))
        self.accuracy = np.append(self.accuracy, float(accuracy))
        self.labels = np.append(self.labels, labels.cpu().detach().numpy())
        self.predictions = np.append(self.predictions, predictions.cpu().detach().numpy())

    def get_epoch_metrics(self):
        epoch_loss = np.mean(self.loss)
        epoch_accuracy = np.mean(self.accuracy)
        return epoch_loss, epoch_accuracy, self.get_aucroc(), self.get_tpr_at_sensitivity(0.95), \
               self.get_auc_precision_recall()

    def get_aucroc(self):
        """Get AUC-ROC and the AUC of the precision-recall curve"""
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
        # return sensitivity[closest_index], tpr[closest_index]
        return tpr[closest_index]
