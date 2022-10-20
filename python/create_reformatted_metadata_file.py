import pandas as pd
import pickle
import numpy as np

metadata = pd.read_csv("data/Coswara_processed/combined_data.csv")

######################################## Rename Metadata to readable entries ###########################################

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
metadata.rename(meta_data_labels, axis="columns", inplace=True)

#################################### Flag zero-lentgh and empty recordings #############################################
# there are ~ 100 participants who recorded invalid audio:
#       43 of those are of length 0
#       55 only contain 0 within the (non-zero length) audio signal
#       2648 valid participants are left after that

with open("data/Coswara_processed/pickles/unvalid_recordings.pickle", "rb") as f:
    invalid_recordings = pickle.load(f)

metadata['recording_error'] = np.nan
for (error, error_info) in invalid_recordings.items():
    for ID in error_info["id_list"]:
        # metadata["recording_error"][metadata["user_id"] == ID] = error
        metadata.loc[metadata["user_id"] == ID, ["recording_error"]] = error
print(metadata.loc[:, "recording_error"].value_counts())

################################# Add audio quality information from other CSVs ########################################
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

for (feature_name, file_path) in recording_quality_files.items():
    audio_quality_annotations = pd.read_csv(file_path)
    # metadata[feature_name] = np.nan

    rename_dict = {"FILENAME": "user_id",
                   " QUALITY": feature_name}
    audio_quality_annotations.rename(rename_dict, axis="columns", inplace=True)
    audio_quality_annotations.user_id = audio_quality_annotations.user_id.str.split("_").str[0]
    # audio_quality_annotations = audio_quality_annotations.astype({feature_name: int})
    metadata = pd.merge(metadata, audio_quality_annotations, on="user_id", how="outer")



################################################ Save into new CSV###### ###############################################

metadata.to_csv("data/Coswara_processed/reformatted_metadata.csv", index=False)
