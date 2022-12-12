from audio_recording import AudioRecording
import os
import pandas as pd

# not necessary anymore... only here to remember which labels were chosen to be positive or negative
NEGATIVE_LABELS = ["healthy", "resp_illness_not_identified", "no_resp_illness_exposed"]
POSITIVE_LABELS = ["positive_mild", "positive_moderate", "positive_asymp"]
UNKNOWN_LABELS = ["under_validation", "recovered_full"]


class Participant:

    def __init__(self, participant_id, augmentations=None):
        self.id = participant_id
        data_directory = "data/Coswara_processed/Recordings"
        self.file_path_participant = os.path.join(data_directory, self.id).replace("\\", "/")
        # self.heavy_cough = AudioRecording(self.file_path_participant, "cough-heavy", augmentations)
        # self.shallow_cough = AudioRecording(self.file_path_participant, type_of_recording="cough-shallow")
        self.deep_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-deep")
        # self.shallow_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-shallow")
        # self.counting_fast = AudioRecording(self.file_path_participant, type_of_recording="counting-fast")
        # self.counting_normal = AudioRecording(self.file_path_participant, type_of_recording="counting-normal")
        # self.vowel_a = AudioRecording(self.file_path_participant, type_of_recording="vowel-a")
        # self.vowel_e = AudioRecording(self.file_path_participant, type_of_recording="vowel-e")
        # self.vowel_o = AudioRecording(self.file_path_participant, type_of_recording="vowel-o")

        data = pd.read_csv("data/Coswara_processed/full_meta_data.csv")
        self.meta_data = data[data["user_id"] == self.id].to_dict("records")[0]
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
