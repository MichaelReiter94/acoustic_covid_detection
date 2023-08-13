import torch

from audio_recording import AudioRecording
import os
import pandas as pd

# not necessary anymore... only here to remember which labels were chosen to be positive or negative
# no_resp_illness_exposed means
NEGATIVE_LABELS = ["healthy", "resp_illness_not_identified", "no_resp_illness_exposed"]
POSITIVE_LABELS = ["positive_mild", "positive_moderate", "positive_asymp"]
UNKNOWN_LABELS = ["under_validation", "recovered_full"]
all_types_of_recording = ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow", "counting-fast",
                          "counting-normal", "vowel-a", "vowel-e", "vowel-o"]
transforms_for_metadata = {
    "nan": 0,
    "y": 1,
    "True": 1,
    "n": -1,
    "False": -1,
    "p": 1,
    "rtpcr": 1,
    "rat": 0.5,
    "female": 1,
    "male": -1,
    "other": 0
}
# metadata_for_training = ["smoker", "cold", "hypertension", "diabetes",
#                          "cough", "diarrheoa", "fever", "loss_of_smell", "muscle_pain", "breathing_difficulties",
#                          "other_respiratory_illness", "fatigue", "sore_throat", "ischemic_heart_disease", "asthma",
#                          "other_preexisting_condition", "chronic_lung_disease", "pneumonia",
#                          "gender", "age", "type_of_covid_test", "vaccination_status"]
metadata_for_training = ["smoker", "cold", "hypertension", "diabetes", "cough", "fever", "loss_of_smell",
                         "muscle_pain",  "breathing_difficulties", "fatigue", "sore_throat", "asthma", "gender",
                         "age", "type_of_covid_test", "vaccination_status"]


class Participant:
    def __init__(self, participant_id, types_of_recording, audio_params, augmentations=None):
        self.id = participant_id
        self.is_augmented = augmentations is not None
        data_directory = "data/Coswara_processed/Recordings"
        file_path_participant = os.path.join(data_directory, self.id).replace("\\", "/")
        self.recordings = {}

        data = pd.read_csv("data/Coswara_processed/full_meta_data.csv")
        self.meta_data = data[data["user_id"] == self.id].to_dict("records")[0]

        # for recording_name in all_types_of_recording:
        #     if recording_name in types_of_recording:
        #         self.recordings[recording_name] = AudioRecording(data_path=file_path_participant,
        #                                                          type_of_recording=recording_name,
        #                                                          audio_parameters=audio_params,
        #                                                          augmentations=augmentations)

        if isinstance(types_of_recording, str):
            types_of_recording = [types_of_recording]

        for recording_name in types_of_recording:
            self.recordings[recording_name] = AudioRecording(data_path=file_path_participant,
                                                             type_of_recording=recording_name,
                                                             audio_parameters=audio_params,
                                                             augmentations=augmentations,
                                                             meta_data=self.meta_data)

        # if types_of_recording == "cough-heavy":
        #     self.heavy_cough = AudioRecording(self.file_path_participant, "cough-heavy", audio_params, augmentations)
        #
        # self.shallow_cough = AudioRecording(self.file_path_participant, "cough-shallow", audio_params, augmentations)
        #
        # if type_of_recording == "breathing-deep":
        #     self.deep_breath = AudioRecording(self.file_path_participant, "breathing-deep", audio_params,
        #     augmentations)
        #
        # self.shallow_breath = AudioRecording(self.file_path_participant, "breathing-shallow", audio_params,
        #                                      augmentations)
        # self.counting_fast = AudioRecording(self.file_path_participant, "counting-fast", audio_params, augmentations)
        # self.counting_normal = AudioRecording(self.file_path_participant, "counting-normal", audio_params,
        #                                       augmentations)
        # self.vowel_a = AudioRecording(self.file_path_participant, "vowel-a", audio_params, augmentations)
        # self.vowel_e = AudioRecording(self.file_path_participant, "vowel-e", audio_params, augmentations)
        # self.vowel_o = AudioRecording(self.file_path_participant, "vowel-o", audio_params, augmentations)

        # creates an additional column for pandas dataframe index that is not really needed or wanted

    def get_label(self):
        """returns 0 if participant is considered healthy or 1 if a covid infection was determined\n
        This label is derived from the 'covid_health_status' from the coswara dataset which includes several
        (sub-)categories"""
        try:
            label = int(self.meta_data["covid_label"])
        except ValueError:
            label = None
        return label

    def __str__(self):
        types_of_recs = list(self.recordings.keys())
        durs = [self.recordings[recording_name].original_duration_trimmed_silence for recording_name in types_of_recs]
        if len(durs) == 1:
            durs = durs[0]
        return f"id: {self.id}\n" \
               f"label: {self.get_label()}\n" \
               f"age: {self.meta_data['age']}\n" \
               f"gender: {self.meta_data['gender']}\n" \
               f"duration: {durs}s"

    def __repr__(self):
        return f"class: {self.__class__.__name__} | id: {self.id} | label: {self.get_label()}"

    def get_transformed_metadata(self):
        transformed_metadata = {}
        for item in metadata_for_training:
            if item == "age":
                transformed_metadata[item] = int(self.meta_data[item]) / 100
            else:
                transformed_metadata[item] = transforms_for_metadata[str(self.meta_data[item])]
        return transformed_metadata

    def get_transformed_metadata_tensor(self):
        metadata = self.get_transformed_metadata()
        return torch.Tensor(list(metadata.values()))
