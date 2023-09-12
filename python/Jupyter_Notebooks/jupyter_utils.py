import tkinter as tk
from tkinter import filedialog
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
import torch


def jupyter_setup():
    import os

    # change working directory to main folder
    root_dir = "python"
    _, current_folder = os.path.split(os.getcwd())
    if current_folder != root_dir:
        os.chdir("../")
    print(os.getcwd())
    # add path variables to avoid "audio module not found"-error
    path = os.environ.get("PATH")
    min_additional_path = "C:\\Users\\Michi\\Anaconda3\\envs\\python_v3-8\\Library\\bin;" \
                          "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\bin;"
    combined_path = min_additional_path + path
    os.environ["PATH"] = combined_path


def load_tracker(verbosity=None):
    root = tk.Tk()
    path = filedialog.askopenfilename(initialdir="run/tracker_saves", title="Select a File",
                                      filetypes=[("Pickled Tracker Files", "*.pickle*")])
    root.destroy()
    print(f"Loading Tracker from: {path}")
    with open(path, "rb") as f:
        tracker = pickle.load(f)

    tracker.compute_overall_metrics(smooth_n_samples=5, ignore_first_n_epochs=5,
                                    metric_used_for_performance_analysis="auc_roc")
    if verbosity is not None:
        tracker.summarize(verbosity)

    # print(f"\nTracker contains:\n>>  {tracker.n_hyperparameter_runs} <<  parameter runs\n"
    #       f">>  {tracker.k_folds_for_cross_validation} <<  folds per run\n"
    #       f">> {tracker.n_epochs} <<  epochs")
    return tracker


#######################################   for the linear regression notebook   #########################################


class Participant:
    def __init__(self, identifier, df, allow_n_missing_recordings=2):
        self.id = identifier
        self.cough = self.get_single_prediction("combined_coughs", df)
        self.speech = self.get_single_prediction("combined_speech", df)
        self.breath = self.get_single_prediction("combined_breaths", df)
        self.vowels = self.get_single_prediction("combined_vowels", df)
        # try:
        self.label = df[df.ID == identifier].label.values[0]
        # except IndexError:
        # self.label = df[df.ID == identifier].label.values

        no_recording = 0
        if self.cough is None:
            no_recording += 1
            self.cough = 0
        if self.speech is None:
            no_recording += 1
            self.speech = 0
        if self.breath is None:
            no_recording += 1
            self.breath = 0
        if self.vowels is None:
            no_recording += 1
            self.vowels = 0
        if no_recording > allow_n_missing_recordings:
            raise ValueError("there is at least 1 recording not present")

    def get_single_prediction(self, rec_type, df):
        idx = np.logical_and(df.ID == self.id, df.rec_type == rec_type)
        n_entries = len(df[idx])
        if n_entries == 1:
            try:
                prediction = df[idx].prediction.values[0][-1]
            except IndexError:
                prediction = df[idx].prediction.values[0]

        elif n_entries == 0:
            # print("error")
            # raise ValueError("No Entry for this")
            # prediction = 0
            prediction = None
        else:
            raise ValueError("there cannot be more than one entry with the same ID and rec type")
        # add sigmoid???
        # print(prediction)
        return prediction

    def get_all_predictions(self):
        return np.array([self.cough, self.speech, self.breath, self.vowels])
    # def calculate AUCROC, accuracy, loss for one category and after linear regression?


def get_linregr_matrices(eval_ids, data, allow_n_missing_recordings=2):
    predictions_matrix = np.array([])
    labels = np.array([])
    ids = []
    for i, participant_id in enumerate(eval_ids):
        try:
            participant = Participant(participant_id, data, allow_n_missing_recordings=allow_n_missing_recordings)
            ids.append(participant_id)
        except ValueError:
            # print("error")
            continue
        # print(participant_id)
        if i == 0 or len(predictions_matrix) == 0:
            predictions_matrix = participant.get_all_predictions()
            labels = np.array([participant.label])
        else:
            predictions = participant.get_all_predictions()
            # print(predictions)
            predictions_matrix = np.vstack([predictions_matrix, predictions])
            labels = np.append(labels, participant.label)
        # print(participant_id)
    return predictions_matrix, labels, ids


def get_confusion_matrix_parameters(labels, predictions, threshold=0.5, verbose=False):
    predictions_bool = predictions > threshold
    labels_bool = labels > threshold
    confusion_mat = confusion_matrix(labels_bool, predictions_bool)
    mat = np.flip(confusion_mat)
    mat = np.concatenate([mat, np.expand_dims(mat.sum(axis=1), 1)], axis=1)
    mat = np.concatenate([mat, np.expand_dims(mat.sum(axis=0), 0)], axis=0)
    if verbose:
        # print("##########################################################################\n")
        print(pd.DataFrame(mat, columns=["Pred. [+]", "Pred. [-]", "True Total"],
                           index=["True [+]", "True [-]", "Pred. Total"]))

    # returns parameters in the following order: tn, fp, fn, tp
    return confusion_mat.ravel()


def get_rates_from_confusion_matrix(confusion_mat, verbose=False):
    TN, FP, FN, TP = confusion_mat
    total_negatives = TN + FP
    total_positives = FN + TP
    tpr = round(TP / total_positives, 4)  # also known as recall
    tnr = round(TN / total_negatives, 4)
    fnr = round(FN / total_positives, 4)
    fpr = round(FP / total_negatives, 4)
    precision = round(TP / (TP + FP), 4)
    if verbose:
        print(pd.DataFrame(dict(tpr=tpr * 100, fpr=fpr * 100, tnr=tnr * 100, fnr=fnr * 100), index=[0]))
    return tpr, fpr, tnr, fnr, precision


def get_auc_prec_recall(labels, predictions):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    auc_preision_recall = auc(recall, precision)
    return auc_preision_recall


def extend_linregr_matrx(A):
    # include further components for the linear regression, like a constant, square of each component, inverse squre, square root,...)
    bias = np.ones((A.shape[0], 1))
    sign = np.sign(A)
    absolute = np.abs(A)
    squares = np.power(A, 2)
    cubes = np.power(A, 3)
    roots = sign*np.power(absolute, 1 / 2)
    cuberoots = np.power(absolute, 1 / 3)
    power_four = np.power(A, 4)
    power_five = np.power(A, 5)
    return np.concatenate((A, squares, roots), axis=1)
    # return np.concatenate((A, squares, roots, cubes, cuberoots), axis=1)
    # return np.concatenate((A, bias), axis=1)
    # return A


def get_aucroc(labels, predictions):
    # using mixup, the resulting labels are no longer binary but continous between 0 and 1
    # we round to get any kind of result but for the training data, the auc-roc is not quite meaningful
    # labels = np.round(self.labels)
    # try:
    fpr, tpr, thresh = roc_curve(labels, predictions)
    aucroc = auc(fpr, tpr)
    # except ValueError:
    #     # fpr, tpr, thresh = 0, 0, 0
    #     aucroc = 0.0
    # return np.round(aucroc * 100, 1)
    return aucroc


def get_accuracy(labels, predictions, threshold=0.5, verbose=False):
    labels_bool = labels > threshold
    # predictions = torch.sigmoid(torch.Tensor(predictions))
    predicted_labels = predictions > threshold
    n_correctly_predicted = np.sum(predicted_labels == labels_bool) / len(predictions)
    # return np.round(n_correctly_predicted * 100, 1)
    return n_correctly_predicted


def sigmoid(A):
    return torch.sigmoid(torch.Tensor(A)).numpy()


def get_tpr_at_sensitivity(labels, predictions, sensitivity_target=0.95):
    # using mixup, the resulting labels are no longer binary but continous between 0 and 1
    # we round to get any kind of result but for the training data, the auc-roc is not quite meaningful
    # labels = np.round(self.labels)
    try:
        fpr, tpr, thresh = roc_curve(labels, predictions)
        sensitivity = 1 - fpr
        distance_to_target_sensitivity = np.abs(sensitivity - sensitivity_target)
        closest_index = np.argmin(distance_to_target_sensitivity)
        return tpr[closest_index]
    except ValueError:
        return 0.0


def only_use_last_prediction_and_loss(input_df):
    df = input_df.copy()
    df['prediction'] = df['prediction'].apply(lambda x: x[-1] if x.size > 1 else x)
    df['loss'] = df['loss'].apply(lambda x: x[-1] if x.size > 1 else x)
    return df


def get_mean_prediction_and_loss(input_df):
    df = input_df.copy()
    df['prediction'] = df['prediction'].apply(lambda x: x.mean() if x.size > 1 else x)
    df['loss'] = df['loss'].apply(lambda x: x.mean() if x.size > 1 else x)
    return df

def apply_sigmoid(input_df):
    df = input_df.copy()
    df['prediction'] = df['prediction'].apply(lambda x: sigmoid(np.array(x)))
    return df
