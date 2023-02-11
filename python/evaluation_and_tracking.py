import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd
import plotly.graph_objects as go

# class ModelEvaluator:
#     def __init__(self):
#         self.training_loss = []
#         self.evaluation_loss = []
#
#         self.batch_training_loss = np.array([])
#         self.batch_evaluation_loss = np.array([])
#         self.mode = "train"
#         self.epoch = 0
#         # add "verbose" attribute which can print something everytime a batch loss or epoch loss was added
#
#     def train(self):
#         self.mode = "train"
#
#     def eval(self):
#         self.mode = "eval"
#
#     def track_loss(self, current_loss: float, mode=None):
#         if mode is not None:
#             self.mode = mode
#
#         if self.mode == "train":
#             self.batch_training_loss = np.append(self.batch_training_loss, current_loss)
#         elif self.mode == "eval":
#             self.batch_evaluation_loss = np.append(self.batch_evaluation_loss, current_loss)
#         else:
#             raise ValueError
#
#     def epoch_has_finished(self):
#         if len(self.batch_training_loss) > 0:
#             self.training_loss.append(self.batch_training_loss)
#         if len(self.batch_evaluation_loss) > 0:
#             self.evaluation_loss.append(self.batch_evaluation_loss)
#         self.batch_training_loss = np.array([])
#         self.batch_evaluation_loss = np.array([])
#         self.epoch += 1
#
#     def get_loss(self, granularity="epoch", mode=None):
#         if mode is not None:
#             self.mode = mode
#
#         temp_loss = self.training_loss if self.mode == "train" else self.evaluation_loss
#
#         if granularity == "epoch":
#             loss = np.array([np.mean(epoch) for epoch in temp_loss])
#         elif granularity == "batch":
#             loss = np.array([])
#             for epoch in temp_loss:
#                 loss = np.append(loss, epoch)
#         else:
#             raise ValueError
#         return loss
color_cycle = [(31, 119, 180, 1),
               (255, 127, 14, 1),
               (44, 160, 44, 1),
               (214, 39, 40, 1),
               (148, 103, 189, 1),
               (140, 86, 75, 1),
               (227, 119, 194, 1),
               (127, 127, 127, 1),
               (188, 189, 34, 1),
               (23, 190, 207, 1)]
metrics_to_minimize = ("loss", "fpr", "fnr")
metrics_to_maximize = ("auc_roc", "accuracy", "f1_score", "auc_prec_recall", "precision", "tpr_at_95", "tpr", "tnr")


def pretty_print_dict(dictionary):
    for k, v in dictionary.items():
        offset = 30 - len(k)
        offset = " " * offset
        print(f"           {k}:{offset}{v}")


def highlight_cells(x):
    return 'background-color: ' + x.map({'True': "#08306b", 'False': 'white'})


def set_font_color(x):
    return 'color: ' + x.map({'True': 'white', 'False': 'black'})


def rgba_plotly(c, opacity=1.0):
    return f"rgba({c[0]},{c[1]},{c[2]},{opacity})"


def smooth_function(data: np.array, kernel_size=1, kernel_mode="linear"):
    if kernel_mode == "linear":
        kernel = np.arange(kernel_size) + 1
        kernel = kernel / np.sum(kernel)
        kernel = np.flip(kernel)
        # print(kernel, np.sum(kernel))
    else:
        kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='full')

    # data_convolved[:kernel_size] = data[:kernel_size]
    for idx in range(kernel_size):
        data_convolved[idx] = np.mean(data[:idx + 1])

    return data_convolved[:len(data)]


class TrackedRun:
    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.metrics = {
            "train": dict(auc_roc=[], loss=[], accuracy=[], f1_score=[], auc_prec_recall=[],
                          precision=[], tpr_at_95=[], tpr=[], tnr=[], fpr=[], fnr=[]),
            "eval": dict(auc_roc=[], loss=[], accuracy=[], f1_score=[], auc_prec_recall=[],
                         precision=[], tpr_at_95=[], tpr=[], tnr=[], fpr=[], fnr=[])
        }
        self.best_performance_parameters = dict(
            epoch=None,
            smoothing=None,
            metric_used=None
        )
        self.best_performances = {}
        # self.ignore_first_n_epochs = 5

    def add_epoch_metrics(self, metric_dict, mode):
        for metric, value in metric_dict.items():
            if metric in self.metrics[mode].keys():
                self.metrics[mode][metric].append(value)

    def convert_to_numpy(self):
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].keys():
                self.metrics[mode][metric] = np.array(self.metrics[mode][metric])
                # # delete metrics in list (lazy fix... make better later pls)
                # if metric in ["confusion_mat"]:
                #     self.metrics[mode].pop(metric)

    def get_metric(self, mode="eval", metric="auc_roc", n_samples_for_smoothing=1):
        # n_samples_for_smoothing meaning the size of the kernel/filter (weighted moving average) in samples
        if n_samples_for_smoothing > 1:
            smoothed_metric = smooth_function(self.metrics[mode][metric], kernel_size=n_samples_for_smoothing)
        else:
            smoothed_metric = self.metrics[mode][metric]
        return smoothed_metric

    def get_epoch_at_minmax(self, metric="auc_roc", smoothing=1, ignore_first_n_epochs=5):
        eval_metric_over_epochs = self.get_metric(metric=metric, n_samples_for_smoothing=smoothing)
        if metric in metrics_to_minimize:
            eval_metric_over_epochs = -1 * eval_metric_over_epochs

        epoch_at_minmax = np.argmax(eval_metric_over_epochs[ignore_first_n_epochs:])
        epoch_at_minmax += ignore_first_n_epochs

        # self.best_performance_parameters["epoch"] = epoch_at_minmax
        # self.best_performance_parameters["smoothing"] = smoothing
        # self.best_performance_parameters["metric_used"] = metric
        return epoch_at_minmax

    def compute_best_performances(self, metric="auc_roc", smoothing=5, ignore_first_n_epochs=5):
        epoch = self.get_epoch_at_minmax(metric=metric, smoothing=smoothing,
                                         ignore_first_n_epochs=ignore_first_n_epochs)
        self.best_performances = {}
        for mode in self.metrics.keys():
            self.best_performances[mode] = {}
            for metric in self.metrics[mode].keys():
                metric_data = self.get_metric(mode=mode, metric=metric, n_samples_for_smoothing=smoothing)
                extremum = metric_data[epoch]
                self.best_performances[mode][metric] = extremum
        return self.best_performances


class CrossValRuns:
    def __init__(self, parameter_set, index, performance_eval_params):
        self.parameters = self.named_run_tuple_to_dict(parameter_set)
        self.runs = []
        self.mean_run = TrackedRun(None)
        self.std_cross_val = TrackedRun(None)
        self.n_folds = None
        self.index = index
        self.color = color_cycle[index % len(color_cycle)]
        self.test = {}
        self.performance_eval_params = performance_eval_params
        # self.performance_eval_params = dict(
        #     smoothing=5,
        #     metric_used="auc_roc",
        #     ignore_first_n_epochs=5
        # )
        # self.best_performances = None
        self.best_performance_mean = {}
        self.best_performance_std = {}

        # self.ignore_first_n_epochs = 5
        # self.metric_for_performance_analysis = "auc_roc"
        # self.smoothing = 5
        self.best_performances = {}

    def named_run_tuple_to_dict(self, p):
        p = str(p)
        p = p.replace("Run(", "")
        p = p.replace(")", "")
        p = p.split(", ")
        param_dict = {}
        for element in p:
            param = element.split("=")
            param_dict[param[0]] = param[1]
        return param_dict

    def add_run(self, run: TrackedRun):
        self.runs.append(run)

    def get_cross_val_statistics(self, mode="eval", metric="auc_roc"):
        # list_of_runs = [run.metrics[mode][metric] for run in self.runs]
        list_of_runs = [run.get_metric(mode=mode, metric=metric) for run in self.runs]
        cross_val_mean = np.mean(np.stack(list_of_runs), axis=0)
        cross_val_std = np.std(np.stack(list_of_runs), axis=0)
        return cross_val_mean, cross_val_std

    def process(self):
        # after finishing the tracking this should be called. It will:
        # (- calculate the min/max of each metric before/after smoothing)
        # - save all the metrics that were used in a  list/dict

        # - save number of cross val runs
        self.n_folds = len(self.runs)

        # - transform the metric lists to np.arrays()
        for run in self.runs:
            run.convert_to_numpy()
            run.compute_best_performances(metric=self.performance_eval_params["metric_used"],
                                          smoothing=self.performance_eval_params["smoothing"],
                                          ignore_first_n_epochs=self.performance_eval_params["ignore_first_n_epochs"])

            # run.compute_best_performances(metric=self.metric_for_performance_analysis,
            #                               smoothing=5,
            #                               ignore_first_n_epochs=self.ignore_first_n_epochs)

        # - calculate the mean over all folds for each metric and saves the curves
        self.best_performances = {}
        for mode in self.mean_run.metrics.keys():
            self.best_performances[mode] = {}
            for metric in self.mean_run.metrics[mode].keys():
                self.best_performances[mode][metric] = []

                mu, std = self.get_cross_val_statistics(mode=mode, metric=metric)
                self.mean_run.metrics[mode][metric] = mu
                self.std_cross_val.metrics[mode][metric] = std

        for fold in self.runs:
            for mode in self.mean_run.metrics.keys():
                for metric in self.mean_run.metrics[mode].keys():
                    self.best_performances[mode][metric].append(fold.best_performances[mode][metric])

        self.best_performance_mean, self.best_performance_std = {}, {}
        for mode in self.best_performances.keys():
            self.best_performance_mean[mode], self.best_performance_std[mode] = {}, {}
            for metric in self.best_performances[mode].keys():
                self.best_performance_mean[mode][metric] = np.nanmean(self.best_performances[mode][metric])
                self.best_performance_std[mode][metric] = np.nanstd(self.best_performances[mode][metric])

    def plot_mean_run(self, mode, metric, n_samples_for_smoothing=1, show_separate_folds=False,
                      show_std_area_plot=True, fig=None):
        # note, that neither mean nor std of the cross validation is calculated from smoothed curves of the folds, but
        # it is rather smoothed AFTER calculations. This results usually (always?) in a higher std

        mean_curve = self.mean_run.get_metric(mode=mode, metric=metric, n_samples_for_smoothing=n_samples_for_smoothing)
        std = self.std_cross_val.get_metric(mode=mode, metric=metric, n_samples_for_smoothing=n_samples_for_smoothing)
        if fig is None:
            fig = go.Figure()

        if show_std_area_plot:
            self.area_plot(mean_curve + std, mean_curve - std, figure=fig)

        if show_separate_folds:
            for run in self.runs:
                fold = run.get_metric(mode=mode, metric=metric, n_samples_for_smoothing=n_samples_for_smoothing)
                ignore_n_first_epochs = 5
                epoch = run.get_epoch_at_minmax(
                    metric=self.performance_eval_params["metric_used"],
                    smoothing=n_samples_for_smoothing,  # smoothing parameter can be changed dynamically by the user
                    ignore_first_n_epochs=self.performance_eval_params["ignore_first_n_epochs"])

                fig.add_trace(go.Scatter(x=np.array(epoch), y=np.array(fold[epoch]), showlegend=False,
                                         name="fold extremum", legendgroup=str(self.parameters),
                                         marker=dict(size=15, color=rgba_plotly(self.color))))

                fig.add_trace(go.Scatter(y=fold, opacity=0.35, showlegend=False,
                                         name="fold", legendgroup=str(self.parameters),
                                         line=dict(color=rgba_plotly(self.color), width=2, dash='dash')))

        fig.add_trace(go.Scatter(y=mean_curve,
                                 name=str(self.parameters),
                                 legendgroup=str(self.parameters),
                                 # showlegend=False,
                                 line=dict(color=rgba_plotly(self.color))))
        return fig

    def area_plot(self, upper_bound, lower_bound, figure=None, x_axis=None):
        if figure is None:
            figure = go.Figure()
        if x_axis is None:
            x_axis = np.arange(len(upper_bound))

        x_path = np.concatenate([x_axis, np.flip(x_axis)], axis=0)
        y_path = np.concatenate([upper_bound, np.flip(lower_bound)], axis=0)
        figure.add_trace(go.Scatter(x=x_path, y=y_path,
                                    name="area of std", legendgroup=str(self.parameters),
                                    fill='toself', showlegend=False, hoverinfo='none',
                                    fillcolor=rgba_plotly(self.color, opacity=0.2),
                                    # opacity=0.2,
                                    line=dict(color="rgba(255,255,255,0)")))
        # line_color='rgba(255,255,255,0)'))
        # figure.update_traces(mode='lines')
        return figure


def get_accuracy(labels, predictions, threshold=0.5, verbose=False):
    labels_bool = labels > threshold
    predicted_labels = predictions > threshold
    n_correctly_predicted = np.sum(predicted_labels == labels_bool) / len(predictions)
    if verbose:
        print(f"\nactual positive labels: {np.sum(labels_bool)}")
        print(f"predicted as positive: {np.sum(predicted_labels)}")
        print(f"correctly predicted: {round(n_correctly_predicted * 100, 2)}%")
    return n_correctly_predicted


def get_confusion_matrix_parameters(labels, predictions, threshold=0.5, verbose=False):
    predictions_bool = predictions > threshold
    labels_bool = labels > threshold
    confusion_mat = confusion_matrix(labels_bool, predictions_bool)
    if verbose:
        print("##########################################################################\n")
        print(pd.DataFrame(np.flip(confusion_mat), columns=["Pred. [+]", "Pred. [-]"],
                           index=["Actual [+]", "Actual [-]"]))
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
        print(pd.DataFrame(dict(tpr=tpr * 100, fpr=fpr * 100, tnr=tnr * 100, fnr=fnr * 100, precision=precision * 100),
                           index=[0]))
    return tpr, fpr, tnr, fnr, precision


class IntraEpochMetricsTracker:
    def __init__(self, datasets, verbose):
        self.datasets = datasets
        self.metrics_used = ["auc_roc", "loss", "accuracy", "f1_score", "auc_prec_recall",
                             "precision", "tpr_at_95", "tpr", "tnr", "fpr", "fnr"]
        self.loss = np.array([])
        self.labels = np.array([])
        self.predictions = np.array([])
        self.mode = ""
        self.crossval_runs = []
        self.current_run = None
        self.performance_eval_params = dict(
            smoothing=5,
            metric_used="auc_roc",
            ignore_first_n_epochs=5
        )
        self.verbose = verbose
        self.model_and_training_parameters = {}
        self.n_epochs = None
        self.n_hyperparameter_runs = None
        self.k_folds_for_cross_validation = None
        self.full_metric_performance_df = None  # only for evaluation set/validation set
        self.compact_metric_performance_df = None
        self.run_ids = None
        self.hyperparameters = None

    def compute_overall_metrics(self, metric_used_for_performance_analysis="auc_roc",
                                smooth_n_samples=5, ignore_first_n_epochs=5):
        self.performance_eval_params = dict(
            metric_used=metric_used_for_performance_analysis,
            smoothing=smooth_n_samples,
            ignore_first_n_epochs=ignore_first_n_epochs
        )

        for run in self.crossval_runs:
            run.performance_eval_params = self.performance_eval_params
            run.process()

        metric_performance_df = pd.DataFrame()
        # pd.set_option('display.max_colwidth', None)  # or 199
        pd.set_option('display.max_columns', None)

        self.full_metric_performance_df = self.get_metric_performance_df(include=("std", "params"),
                                                                         remove_columns=("fnr", "fpr"))
        self.compact_metric_performance_df = self.get_metric_performance_df(include="params",
                                                                            remove_columns=("fnr", "fpr", "tpr_at_95"))

        self.metrics_used = list(self.crossval_runs[0].runs[0].metrics["eval"].keys())
        self.n_epochs = len(self.crossval_runs[0].runs[0].metrics["train"]["loss"])
        self.n_hyperparameter_runs = len(self.crossval_runs)
        self.k_folds_for_cross_validation = len(self.crossval_runs[0].runs)
        self.run_ids = [str(run.parameters) for run in self.crossval_runs]
        self.hyperparameters = list(self.crossval_runs[0].parameters.keys())
        print(f"\nParameters used for finding the best performing epoch:")
        pretty_print_dict(self.performance_eval_params)

    def get_metric_performance_df(self, include=("std", "params", "combined_params"), remove_columns=()):
        df = pd.DataFrame()
        for run in self.crossval_runs:
            row = dict(run.best_performance_mean["eval"])
            if "std" in include:
                row_std = dict(run.best_performance_std["eval"])
                row_temp = {}
                for key, val in row.items():
                    row_temp[f"{key}"] = row[key]
                    row_temp[f"{key}-σ"] = row_std[key]
                row = row_temp

            if "params" in include or "parameters" in include or "param" in include:
                row.update(run.parameters)

            if "combined_params" in include or "combined_parameters" in include:
                row.update({"combined_params": str(run.parameters)})
            keys_to_be_removed = []
            for key in row.keys():
                for column_to_be_removed in remove_columns:
                    if column_to_be_removed in key:
                        keys_to_be_removed.append(key)
            for key in keys_to_be_removed:
                row.pop(key)

            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
        return df

    def style_metric_performance_df(self, df, sort_by="auc_roc", metrics_to_min=metrics_to_minimize,
                                    metrics_to_max=metrics_to_maximize, highlight_hyperparameters=True,
                                    filter_thresholds={}):

        df = pd.DataFrame(df)
        # remove all rows where the performance is so bad for a certain metric that it drops below a given threshold
        for metric, threshold in filter_thresholds.items():
            if metric in metrics_to_minimize:
                df = df[df[metric] < threshold]
            else:
                df = df[df[metric] > threshold]

        low_is_good_cols = [col for col in metrics_to_minimize if col in df.columns]
        # metrics_to_maximize = [metric for metric in self.metrics_used if metric not in metrics_to_minimize]
        high_is_good_cols = [col for col in metrics_to_maximize if col in df.columns]

        ascending = sort_by in metrics_to_minimize
        style = df.sort_values(sort_by, ascending=ascending).style.background_gradient(cmap="RdYlGn",
                                                                                       subset=high_is_good_cols)
        style = style.background_gradient(cmap="RdYlGn_r", subset=low_is_good_cols)
        style.set_table_styles([
            {
                "selector": "",
                "props": [("border", "3px solid grey")]},
            {
                "selector": "tbody td",
                "props": [("border", "3px solid black")]},
            {
                "selector": "th",
                "props": [("border", "3px solid grey")]}
        ])

        if highlight_hyperparameters:
            for col in df:
                if len(df[col].unique()) > 1 and col not in self.metrics_used and "σ" not in col:
                    try:  # check if the values in the column are numeric
                        _ = df[col].astype("float")
                        style = style.background_gradient(cmap="Blues", subset=[col])
                    except ValueError:  # otherwise it is assumed to be a binary column which is styled differently
                        style.apply(highlight_cells, subset=[col])
                        style.apply(set_font_color, subset=[col])

        return style

    def setup_run_with_new_params(self, parameter_set):
        index = len(self.crossval_runs)
        self.crossval_runs.append(CrossValRuns(parameter_set, index, self.performance_eval_params))

    def start_run_with_random_seed(self, random_seed):
        self.current_run = TrackedRun(random_seed)
        self.crossval_runs[-1].add_run(self.current_run)

    def reset(self, run_id, mode):
        self.mode = mode
        # self.current_run_id = str(run_id)
        # if self.runs.get(self.current_run_id) is None:
        #     self.runs[self.current_run_id] = {}
        # if self.runs[self.current_run_id].get(self.mode) is None:
        #     self.runs[self.current_run_id][self.mode] = dict(loss=[],
        #                                                      accuracy=[],
        #                                                      confusion_mat=[],
        #                                                      tpr=[],
        #                                                      fpr=[],
        #                                                      tnr=[],
        #                                                      fnr=[],
        #                                                      auc_roc=[],
        #                                                      tpr_at_95=[],
        #                                                      auc_prec_recall=[],
        #                                                      precision=[],
        #                                                      f1_score=[])
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
        confusion_mat = get_confusion_matrix_parameters(self.labels, self.predictions, verbose=self.verbose)
        tpr, fpr, tnr, fnr, precision = get_rates_from_confusion_matrix(confusion_mat, verbose=self.verbose)
        recall = tpr
        f1_score = 2 * (precision * recall) / (precision + recall)
        metric_dict = dict(
            auc_roc=self.get_aucroc(),
            loss=np.mean(self.loss),
            accuracy=get_accuracy(self.labels, self.predictions, verbose=self.verbose),
            f1_score=f1_score,
            auc_prec_recall=self.get_auc_precision_recall(),
            precision=precision,
            tpr_at_95=self.get_tpr_at_sensitivity(0.95),
            tpr=tpr,
            tnr=tnr,
            fpr=fpr,
            fnr=fnr,
            confusion_mat=confusion_mat,
        )
        self.current_run.add_epoch_metrics(metric_dict, mode=self.mode)
        return metric_dict

    def get_aucroc(self):
        # using mixup, the resulting labels are no longer binary but continous between 0 and 1
        # we round to get any kind of result but for the training data, the auc-roc is not quite meaningful
        labels = np.round(self.labels)
        fpr, tpr, thresh = roc_curve(labels, self.predictions)
        aucroc = auc(fpr, tpr)
        return aucroc

    def get_auc_precision_recall(self):
        # using mixup, the resulting labels are no longer binary but continous between 0 and 1
        # we round to get any kind of result but for the training data, the auc-roc is not quite meaningful
        labels = np.round(self.labels)
        precision, recall, _ = precision_recall_curve(labels, self.predictions)
        auc_preision_recall = auc(recall, precision)
        return auc_preision_recall

    def get_tpr_at_sensitivity(self, sensitivity_target=0.95):
        # using mixup, the resulting labels are no longer binary but continous between 0 and 1
        # we round to get any kind of result but for the training data, the auc-roc is not quite meaningful
        labels = np.round(self.labels)
        fpr, tpr, thresh = roc_curve(labels, self.predictions)
        sensitivity = 1 - fpr
        distance_to_target_sensitivity = np.abs(sensitivity - sensitivity_target)
        closest_index = np.argmin(distance_to_target_sensitivity)
        return tpr[closest_index]

    def show_all_runs(self, mode="eval", metric="auc_roc", n_samples_for_smoothing=None, show_separate_folds=False,
                      show_std_area_plot=True, show_n_best_runs="all"):

        if show_n_best_runs == "all":
            show_n_best_runs = self.n_hyperparameter_runs

        if n_samples_for_smoothing is None:
            n_samples_for_smoothing = self.performance_eval_params["smoothing"]

        fig = go.Figure()
        for idx, run in enumerate(self.crossval_runs):
            if idx in self.get_indices_of_best_n_runs(n=show_n_best_runs,
                                                      sort_by=self.performance_eval_params["metric_used"]):
                run.plot_mean_run(mode=mode, metric=metric, n_samples_for_smoothing=n_samples_for_smoothing,
                                  show_separate_folds=show_separate_folds, fig=fig,
                                  show_std_area_plot=show_std_area_plot)
        fig.update_layout(autosize=False, width=1300, height=600, showlegend=True,
                          margin=dict(l=20, r=20, t=50, b=20),
                          title_text=f"<b>{mode} - {metric}</b>", title_x=0.3, title_y=0.97, titlefont=dict(size=24),
                          legend={"orientation": 'h'}
                          )
        # fig.show()
        return fig

    def boxplot_run_statistics(self, metric, show_n_best_runs="all", color_by_hyperparameter=None,
                               sort_by_current_metric=True):
        if show_n_best_runs == "all":
            show_n_best_runs = self.n_hyperparameter_runs

        df = pd.DataFrame()
        for run in self.crossval_runs:
            row = dict(run.best_performances["eval"])
            row.update({"combined_params": str(run.parameters)})
            df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

        black = (0, 0, 0, 1)
        fig = go.Figure()
        if sort_by_current_metric:
            sort_by_metric = metric
        else:
            sort_by_metric = self.performance_eval_params["metric_used"]
        best_indices = self.get_indices_of_best_n_runs(n=show_n_best_runs, sort_by=sort_by_metric)

        for idx in best_indices:
            run = self.crossval_runs[idx]
            if color_by_hyperparameter is not None:
                hyperparams = list(self.full_metric_performance_df[color_by_hyperparameter].unique())
                color = color_cycle[hyperparams.index(run.parameters[color_by_hyperparameter])]
                name = f": {color_by_hyperparameter}: {run.parameters[color_by_hyperparameter]}"
            else:
                color = run.color
                name = str(run.parameters)

            temp = df[df["combined_params"] == str(run.parameters)]
            fig.add_trace(go.Box(y=temp[metric], boxmean=True, x=[metric] * len(temp), name=name,
                                 jitter=0.3, pointpos=-0, boxpoints='all', marker_color=f"rgba{black}",
                                 line_color=f"rgba{color}")
                          )

        fig.update_layout(width=1300, height=600, margin=dict(l=150, r=20, t=10, b=20), boxmode='group',
                          legend={"orientation": 'h'})
        return fig

    def show_single_run(self, run_id, mode="eval", metric="auc_roc", n_samples_for_smoothing=None,
                        show_separate_folds=True,
                        show_std_area_plot=True):

        if n_samples_for_smoothing is None:
            n_samples_for_smoothing = self.performance_eval_params["smoothing"]

        fig = go.Figure()

        run_idx = self.run_ids.index(run_id)
        run = self.crossval_runs[run_idx]

        run.plot_mean_run(mode=mode, metric=metric, n_samples_for_smoothing=n_samples_for_smoothing,
                          show_separate_folds=show_separate_folds, fig=fig, show_std_area_plot=show_std_area_plot)
        fig.update_layout(autosize=False, width=1300, height=600, showlegend=True,
                          margin=dict(l=20, r=20, t=50, b=20),
                          title_text=f"<b>{mode} - {metric}</b>", title_x=0.3, title_y=0.97, titlefont=dict(size=24),
                          legend={"orientation": 'h'})
        # fig.show()
        return fig

    def summarize(self, detail="compact"):
        # print(f"Datasets used:\n{self.datasets}")
        print(f"Datasets used:")
        key = list(self.datasets.keys())[0]
        print(key)
        pretty_print_dict(self.datasets[key])

        try:
            if detail == "compact":
                pretty_print_dict(self.model_and_training_parameters[detail])
            else:
                for key, val in self.model_and_training_parameters["full"].items():
                    print(f"______________________________________________________________\n{key}:\n{val}")
                    # print(val)
        except:  # eliminate after a while
            print(f"\nModel used:")
            print(self.model_name)

        print(f"\nTracker contains:\n>>  {self.n_hyperparameter_runs} <<  parameter runs\n"
              f">>  {self.k_folds_for_cross_validation} <<  folds per run\n"
              f">> {self.n_epochs} <<  epochs")

    def get_indices_of_best_n_runs(self, n, sort_by):
        if self.n_hyperparameter_runs < n:
            n = self.n_hyperparameter_runs
        sort_ascending = sort_by in metrics_to_minimize

        sorted_df = self.full_metric_performance_df.sort_values(sort_by, ascending=sort_ascending)
        indices = list(sorted_df[:n].index)
        return indices

    def save_model_and_training_parameters(self, model_info, optimizer, loss_function):
        self.model_and_training_parameters = {
            "full": {
                "model": str(model_info),
                "optimizer": str(optimizer),
                "loss_function": str(loss_function)
            },
            "compact": {
                "model": str(model_info).split("(")[0],
                "optimizer": str(optimizer).split("(")[0],
                "loss_function": str(loss_function).split("(")[0]
            }

        }
