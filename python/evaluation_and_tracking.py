import numpy as np


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
