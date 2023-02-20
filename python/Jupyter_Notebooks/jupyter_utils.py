import tkinter as tk
from tkinter import filedialog
import pickle


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

    tracker.compute_overall_metrics(smooth_n_samples=10, ignore_first_n_epochs=10,
                                    metric_used_for_performance_analysis="auc_roc")
    if verbosity is not None:
        tracker.summarize(verbosity)

    # print(f"\nTracker contains:\n>>  {tracker.n_hyperparameter_runs} <<  parameter runs\n"
    #       f">>  {tracker.k_folds_for_cross_validation} <<  folds per run\n"
    #       f">> {tracker.n_epochs} <<  epochs")
    return tracker
