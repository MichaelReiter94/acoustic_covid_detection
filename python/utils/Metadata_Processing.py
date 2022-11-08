import os
import pandas as pd
import numpy as np
from IPython.display import clear_output


# <editor-fold desc="Librosa Error Workaround">
# adding to PATH environment variable To get librosa and other audio libraries to run. don't know why but jupyter
# does not load the necessary environment pth variables by itself
path = os.environ.get("PATH")
min_additional_path = "C:\\Users\\Michi\\Anaconda3\\envs\\python_v3-8\\Library\\bin;"
combined_path = min_additional_path + path
os.environ["PATH"] = combined_path
import librosa
# </editor-fold>

# <editor-fold desc="Function Definitions">

def rename_columns(metadata_df):
    meta_data_labels = {
        "id": "user_id",
        "a": "age",
        "covid_status": "covid_health_status",
        "record_date": "record_date",
        "ep": "english_proficiency",
        "g": "gender",
        "l_c": "country",
        "l_l": "local_region",
        "l_s": "state",
        "rU": "returning_user",
        "asthma": "asthma",
        "cough": "cough",
        "smoker": "smoker",
        "test_status": "covid_test_result",
        "ht": "hypertension",
        "cold": "cold",
        "diabetes": "diabetes",
        "diarrhoea": "diarrheoa",
        "um": "was_using_mask",
        "ihd": "ischemic_heart_disease",
        "bd": "breathing_difficulties",
        "st": "sore_throat",
        "fever": "fever",
        "ftg": "fatigue",
        "mp": "muscle_pain",
        "loss_of_smell": "loss_of_smell",
        "cld": "chronic_lung_disease",
        "pneumonia": "pneumonia",
        "ctScan": "has_taken_ct_scan",
        "testType": "type_of_covid_test",
        "test_date": "covid_test_date",
        "vacc": "vaccination_status",  # (y->both doses, p->one dose(partially vaccinated), n->no doses)
        "ctDate": "date_of_ct_scan",
        "ctScore": "ct_score",
        "others_resp": "other_respiratory_illness",
        "others_preexist": "other_preexisting_condition"
    }
    metadata_df.rename(meta_data_labels, axis="columns", inplace=True)


def create_labels(metadata_df, verbose=False):
    # 0 meaning healthy, 1 meaning covid-infection
    NEGATIVE_LABELS = ["healthy", "resp_illness_not_identified", "no_resp_illness_exposed"]
    POSITIVE_LABELS = ["positive_mild", "positive_moderate", "positive_asymp"]
    UNKNOWN_LABELS =  ["under_validation", "recovered_full"]

    metadata_df["covid_label"] = np.nan
    negative_idx = metadata_df["covid_health_status"].str.contains("|".join(NEGATIVE_LABELS))
    positive_idx = metadata_df["covid_health_status"].str.contains("|".join(POSITIVE_LABELS))
    metadata_df.loc[negative_idx, "covid_label"] = 0
    metadata_df.loc[positive_idx, "covid_label"] = 1
    if verbose:
        print(metadata_df["covid_label"].value_counts())


def get_audio_quality_annotations():
    PATH = "data/Coswara-Data/annotations/"
    recording_quality_files = {"audio_quality_deep_breathing":  PATH + "breathing-deep_labels_pravinm.csv",
                               "audio_quality_shallow_breathing": PATH + "breathing-shallow_labels_pravinm.csv",
                               "audio_quality_heavy_cough": PATH + "cough-heavy_labels_debottam.csv",
                               "audio_quality_shallow_cough": PATH + "cough-shallow_labels_debarpan.csv",
                               "audio_quality_counting_fast": PATH + "counting-fast_labels_pravinm.csv",
                               "audio_quality_counting_normal": PATH + "counting-normal_labels_pravinm.csv",
                               "audio_quality_vowel_a": PATH + "vowel-a_labels_debarpan.csv",
                               "audio_quality_vowel_e": PATH + "vowel-e_labels_debottam.csv",
                               "audio_quality_vowel_o": PATH + "vowel-o_labels_updated_neeraj.csv"}

    df = pd.DataFrame({"user_id":participant_ids})
    for (feature_name, file_path) in recording_quality_files.items():
        audio_quality_annotations = pd.read_csv(file_path)

        rename_dict = {"FILENAME": "user_id",
                       " QUALITY": feature_name}

        audio_quality_annotations.rename(rename_dict, axis="columns", inplace=True)
        audio_quality_annotations.user_id = audio_quality_annotations.user_id.str.split("_").str[0]
        df = pd.merge(df, audio_quality_annotations, on="user_id", how="outer")
    return df


def get_recording_durations(id_participant, recordings_path="data/Coswara_processed/Recordings"):
    # returns original duration and duration after trimming trailing/leading silence for each recording type as a dict
    # returns 0 for both if an error occured/the audio is invalid
    recording_types = ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow", "counting-fast",
                       "counting-normal", "vowel-a", "vowel-e", "vowel-o"]
    path = os.path.join(recordings_path, id_participant)
    duration_original = {}
    duration_trimmed = {}

    for rec_type in recording_types:
        file_path = f"{os.path.join(path, rec_type)}.wav"
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
        except FileNotFoundError:
            duration_original[rec_type], duration_trimmed[rec_type] = 0, 0
        if len(audio) == 0:
            duration_original[rec_type], duration_trimmed[rec_type] = 0, 0
        elif max(audio) == 0:
            duration_original[rec_type], duration_trimmed[rec_type] = 0, 0
        else:
            duration_original[rec_type] = len(audio)/sample_rate
            audio, _ = librosa.effects.trim(audio, top_db=54)
            duration_trimmed[rec_type] = len(audio)/sample_rate


    return duration_original, duration_trimmed


def get_recording_duration_df(participant_ids, LOAD_FROM_DISC=True):
    from time import sleep
    recording_types = ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow", "counting-fast",
                       "counting-normal", "vowel-a", "vowel-e", "vowel-o"]
    if LOAD_FROM_DISC:
        try:
            audio_recording_metadata = pd.read_csv("data/Coswara_processed/duration_df.csv")
            print("loaded audio_recording_metadata dataframe from disk!")
        except FileNotFoundError:
            LOAD_FROM_DISC=False
            print("File not found, computing durations anew")
            sleep(3)

    if not LOAD_FROM_DISC:
        durations_original, durations_trimmed = [], []
        for idx, participant in enumerate(participant_ids):
            clear_output(wait=True)
            print(f"{idx+1} / {len(participant_ids)}")
            original, trimmed = get_recording_durations(participant)
            durations_original.append(original), durations_trimmed.append(trimmed)

        all_durations_orig = {}
        all_durations_trim = {}
        for rec_type in recording_types:
            all_durations_orig[f"duration_original_{rec_type}"] = [round(recording[rec_type],3) for recording in durations_original]
            all_durations_trim[f"duration_trimmed_{rec_type}"] = [round(recording[rec_type],3) for recording in durations_trimmed]

        duration_dict = {"user_id":participant_ids}
        duration_dict.update(all_durations_orig)
        duration_dict.update(all_durations_trim)
        audio_recording_metadata = pd.DataFrame(duration_dict)

        audio_recording_metadata.to_csv("data/Coswara_processed/duration_df.csv", index=False)
    return audio_recording_metadata


def get_invalid_recordings(audio_recording_metadata, recording_types_used=
                           ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow",
                            "counting-fast","counting-normal", "vowel-a", "vowel-e", "vowel-o"]):

    recording_is_invalid=[]
    for rec_type in recording_types_used:
        recording_is_invalid.append(audio_recording_metadata[f"duration_original_{rec_type}"] == 0)

    n_invalid_recordings = np.array(recording_is_invalid).sum(axis=0)
    contains_invlaid_recording = n_invalid_recordings > 0
    contains_invalid_recording_df = pd.DataFrame({"user_id": participant_ids,
                                                  "recording_invalid": contains_invlaid_recording,
                                                  "n_invalid_recordings": n_invalid_recordings})
    return contains_invalid_recording_df

# </editor-fold>


# get into the correct root directory
root_dir = "python"
_, current_folder = os.path.split(os.getcwd())
if current_folder != root_dir:
    os.chdir("../")

metadata = pd.read_csv("data/Coswara_processed/original_metadata.csv")
participant_ids = os.listdir("data/Coswara_processed/Recordings/")
rename_columns(metadata)

audio_quality_annotations = get_audio_quality_annotations()
create_labels(metadata, verbose=False)
recording_metadata_df = get_recording_duration_df(participant_ids, LOAD_FROM_DISC=True)
invalid_recordings_df = get_invalid_recordings(recording_metadata_df)

# # Merge all partial meta data dataframes
full_meta_data = pd.merge(metadata, audio_quality_annotations, on="user_id", how="outer")
full_meta_data = pd.merge(full_meta_data, recording_metadata_df, on="user_id", how="outer")
full_meta_data = pd.merge(full_meta_data, invalid_recordings_df, on="user_id", how="outer")

# save various csv meta-data files
full_meta_data.to_csv("data/Coswara_processed/full_meta_data.csv", index=False)
metadata.to_csv("data/Coswara_processed/reformatted_metadata.csv", index=False)
invalid_recordings_df.to_csv("data/Coswara_processed/invalid_recordings.csv", index=False)
recording_metadata_df.to_csv("data/Coswara_processed/duration_df.csv", index=False)
audio_quality_annotations.to_csv("data/Coswara_processed/audio_quality_annotations_df.csv", index=False)