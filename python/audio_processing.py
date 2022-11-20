# <editor-fold desc="Imports">
import os
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, Gain, Shift, TimeStretch, Trim

path = os.environ.get("PATH")
additional_path = "C:\\Users\\micha\\anaconda3\\envs\\ai38;" \
                  "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\mingw-w64\\bin;" \
                  "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\usr\\bin;" \
                  "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\bin;" \
                  "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Scripts;" \
                  "C:\\Users\\micha\\anaconda3\\envs\\ai38\\bin;" \
                  "C:\\Users\\micha\\anaconda3\\condabin;"
min_additional_path = "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\bin;"
os.environ["PATH"] = min_additional_path + path
import librosa
from participant import Participant
import pickle
from tqdm import tqdm
import pandas as pd

# </editor-fold>


def create_participant_objects(save_to: str, augmentations=None, augmentations_per_label=(1, 1),
                               UPDATE_INVALID_RECORDINGS=False):
    # no oversampling when there are no augmentations specified
    if augmentations is None:
        augmentations_per_label = (1, 1)

    error_ids_zero_length, error_ids_all_zeros, error_ids_unknown, participants, errors = [], [], [], [], {}
    for ID in tqdm(participant_ids):
        participant_metadata = metadata[metadata["user_id"] == ID]
        try:
            label = int(participant_metadata.covid_label)
            for _ in range(augmentations_per_label[label]):
                participants.append(Participant(ID, augmentations=augmentations))
        except ValueError as e:
            # length of recording is 0
            # or no valid covid label
            error_ids_zero_length.append(ID)
            errors[type(e).__name__] = {"error_description": e.args[0],
                                        "error_meaning": "length of the audio file = 0 or the user does not have a "
                                                         "valid covid label/test",
                                        "id_list": error_ids_zero_length}
        except librosa.util.exceptions.ParameterError as e:
            # all values in the audio file are 0 (but the length is > 0) resulting in NaN when normalizing with 0 as max
            error_ids_all_zeros.append(ID)
            errors[type(e).__name__] = {"error_description": e.args[0],
                                        "error_meaning": "All values in audio file are 0",
                                        "id_list": error_ids_all_zeros}

    if UPDATE_INVALID_RECORDINGS:
        with open("data/Coswara_processed/pickles/invalid_recordings.pickle", "wb") as f:
            pickle.dump(errors, f)

    print(f"saving {len(participants)} instances\n")
    with open(f"data/Coswara_processed/pickles/{save_to}.pickle", "wb") as f:
        pickle.dump(participants, f)


time_domain_augmentations = Compose([
    AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.01, p=0.8),
    PitchShift(min_semitones=-1, max_semitones=1, p=0.8),
    TimeStretch(min_rate=0.85, max_rate=1.15, leave_length_unchanged=False, p=0.8),
    Gain(min_gain_in_db=-30, max_gain_in_db=6, p=0.8)
])

participant_ids = os.listdir("data/Coswara_processed/Recordings")
metadata = pd.read_csv("data/Coswara_processed/full_meta_data.csv")

create_participant_objects(save_to="participants_oversampledCovidPositives_validLabelsOnly",
                           augmentations=time_domain_augmentations,
                           augmentations_per_label=(1, 4))
