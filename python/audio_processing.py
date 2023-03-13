# <editor-fold desc="Imports">
import os
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain, TimeStretch

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
from datetime import datetime
from utils.utils import audiomentations_repr

# </editor-fold>
# def create_participant_objects(save_to: str,
#                                augmentations=None,
#                                augmentations_per_label=(1, 1),
#                                UPDATE_INVALID_RECORDINGS=False):
#     # no oversampling when there are no augmentations specified
#     if augmentations is None:
#         augmentations_per_label = (1, 1)
#
#     error_ids_zero_length, error_ids_all_zeros, error_ids_unknown, participants, errors = [], [], [], [], {}
#     for ID in tqdm(participant_ids):
#         participant_metadata = metadata[metadata["user_id"] == ID]
#         # print(participant_metadata["audio_quality_heavy_cough"].item())
#         # if participant_metadata["audio_quality_deep_breathing"].item() > 0:
#         if participant_metadata["audio_quality_heavy_cough"].item() > 0:
#             try:
#                 label = int(participant_metadata.covid_label)
#                 for _ in range(augmentations_per_label[label]):
#                     participants.append(Participant(ID, augmentations=augmentations))
#             except ValueError as e:
#                 # length of recording is 0, or no valid covid label
#                 error_ids_zero_length.append(ID)
#                 errors[type(e).__name__] = {"error_description": e.args[0],
#                                             "error_meaning": "length of the audio file = 0 or the user does not have a "
#                                                              "valid covid label/test",
#                                             "id_list": error_ids_zero_length}
#             except librosa.util.exceptions.ParameterError as e:
#                 # all values in the audio file are 0 (but the length is > 0) resulting in NaN when normalizing with 0
#                 # as max
#                 error_ids_all_zeros.append(ID)
#                 errors[type(e).__name__] = {"error_description": e.args[0],
#                                             "error_meaning": "All values in audio file are 0",
#                                             "id_list": error_ids_all_zeros}
#
#     if UPDATE_INVALID_RECORDINGS:
#         with open("data/Coswara_processed/pickles/invalid_recordings.pickle", "wb") as f:
#             pickle.dump(errors, f)
#
#     print(f"saving {len(participants)} instances\n")
#     with open(f"data/Coswara_processed/pickles/{save_to}.pickle", "wb") as f:
#         pickle.dump(participants, f)


def contains_only_good_audio(participant_metadata, types_of_recording):
    only_good_audio = True

    combined_recordings = {
        "combined_coughs": ["cough-heavy", "cough-shallow"],
        "combined_breaths": ["breathing-deep", "breathing-shallow"],
        "combined_vowels": ["vowel-a", "vowel-e", "vowel-o"]
    }

    if isinstance(types_of_recording, str):
            types_of_recording = [types_of_recording]

    for rec_type in types_of_recording:
        recording_quality = 0

        if rec_type in combined_recordings:
            # allow for recordings that have at least 1 of the combined recording types to have above quality 0
            # if there is any nan in any of the recordings, it will be dismissed
            combined_recordings = combined_recordings.get(rec_type)
            for combined_rec_type in combined_recordings:

                recording_quality += participant_metadata[f"audio_quality_{combined_rec_type}"].item()
        else:
            recording_quality += participant_metadata[f"audio_quality_{rec_type}"].item()
        if recording_quality > 0:
            pass
        else:
            only_good_audio = False
    return only_good_audio


def pretty_print_dict(dictionary):
    return_string = ""
    for k, v in dictionary.items():
        offset = 20 - len(k)
        offset = " " * offset
        line = f"{k}:{offset}{v}"
        return_string = f"{return_string}\n{line}"
    return return_string



class FeatureSet:
    def __init__(self, type_of_recording, audio_params):
        audio_params["hop_size_ms"] =  round(audio_params["hop_size"]/audio_params["sample_rate"]*1000, 2)
        audio_params["window_length_ms"] =  round(audio_params["window_length"]/audio_params["sample_rate"]*1000, 2)
        audio_params["duration_seconds"] =  round(audio_params["hop_size_ms"]*audio_params["n_time_steps"]/1000, 2)
        audio_params["fft_res_hz"] =  round(audio_params["sample_rate"]/audio_params["n_fft"], 2)

        self.types_of_recording = type_of_recording  # cough-heavy | breathing-deep | ... or a list of them!
        self.audio_parameters = audio_params
        self.augmentations = None
        self.augmentations_per_label = None  # labels(0,1) meaning (neg, pos)
        self.participants = []
        self.is_augmented = None

    def create_participant_objects(self, augmentations=None,
                                   augmentations_per_label=(1, 1),
                                   UPDATE_INVALID_RECORDINGS=False):
        self.augmentations = augmentations
        participant_ids = os.listdir("data/Coswara_processed/Recordings")
        metadata = pd.read_csv("data/Coswara_processed/full_meta_data.csv")
        self.participants = []

        # no oversampling when there are no augmentations specified
        if augmentations is None:
            augmentations_per_label = (1, 1)
            self.is_augmented = False
        else:
            self.is_augmented = True
        self.augmentations_per_label = augmentations_per_label

        error_ids_zero_length, error_ids_all_zeros, error_ids_unknown, errors = [], [], [], {}
        for ID in tqdm(participant_ids):
            participant_metadata = metadata[metadata["user_id"] == ID]

            audio_quailty_is_good_enough = contains_only_good_audio(participant_metadata, self.types_of_recording)
            if audio_quailty_is_good_enough:
                try:
                    label = int(participant_metadata.covid_label)
                    for _ in range(self.augmentations_per_label[label]):
                        self.participants.append(Participant(participant_id=ID,
                                                             types_of_recording=self.types_of_recording,
                                                             audio_params=self.audio_parameters,
                                                             augmentations=augmentations))


                except ValueError as e:
                    # length of recording is 0, or no valid covid label
                    error_ids_zero_length.append(ID)
                    errors[type(e).__name__] = {"error_description": e.args[0],
                                                "error_meaning": "length of the audio file = 0 or the user does not "
                                                                 "have a "
                                                                 "valid covid label/test",
                                                "id_list": error_ids_zero_length}
                except librosa.util.exceptions.ParameterError as e:
                    # all values in the audio file are 0 (but the length is > 0) resulting in NaN when normalizing with
                    # 0 as max
                    error_ids_all_zeros.append(ID)
                    errors[type(e).__name__] = {"error_description": e.args[0],
                                                "error_meaning": "All values in audio file are 0",
                                                "id_list": error_ids_all_zeros}

        if UPDATE_INVALID_RECORDINGS:
            with open("data/Coswara_processed/pickles/invalid_recordings.pickle", "wb") as f:
                pickle.dump(errors, f)

    def save_to(self, save_to):
        print(f"saving {len(self.participants)} participant instances\n")
        date = datetime.today().strftime("%Y_%m_%d")
        append = ""
        if self.augmentations is not None:
            append += "augmented"

        with open(f"data/Coswara_processed/pickles/{date}_{self.audio_parameters['type_of_features']}_"
                  f"{self.types_of_recording}_{save_to}{append}.pickle", "wb") as f:
            pickle.dump(self, f)

    def __str__(self):
        representation = f"\nFeature Set - #Participants: {self.__len__()}"
        representation += "\n----------------------------------------------------"
        representation += pretty_print_dict(self.audio_parameters)
        representation += "\n-------------------Augmentations--------------------"

        representation += pretty_print_dict(audiomentations_repr(self.augmentations))
        representation += "\n----------------------------------------------------"

        return representation

    def __repr__(self):
        return f"Feature Set - #Participants: {self.__len__()}"

    def __len__(self):
        return len(self.participants)



time_domain_augmentations = Compose([
    AddGaussianNoise(min_amplitude=0.0003, max_amplitude=0.01, p=0.8),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.8),
    TimeStretch(min_rate=0.95, max_rate=1.05, leave_length_unchanged=False, p=0.8),
    Gain(min_gain_in_db=-18, max_gain_in_db=12, p=0.8)
])
all_types_of_recording = ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow", "counting-fast",
                          "counting-normal", "vowel-a", "vowel-e", "vowel-o"]
combined_recordings = {
    "combined_coughs": ["cough-heavy", "cough-shallow"],
    "combined_breaths": ["breathing-deep", "breathing-shallow"],
    "combined_vowels": ["vowel-a", "vowel-e", "vowel-o"]
}

audio_parameters = dict(
    type_of_features="mfcc",  # logmel | mfcc
    n_time_steps=259,  # 259 | 224
    n_features=50,  # 15 | 224
    sample_rate=22050,
    n_fft=512 * 16,
    window_length=512 * 8,
    hop_size=512*2,
    fmin=0,
    fmax=22050 // 4
)

if __name__ == "__main__":
    feature_set = FeatureSet("combined_breaths", audio_parameters)
    feature_set.create_participant_objects(augmentations=time_domain_augmentations,
                                           augmentations_per_label=(1, 5))
    feature_set.save_to("12s_FFT4096_fmax5500_50mfccs_x1x5")
