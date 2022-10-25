from audio_recording import AudioRecording
import os
import pandas as pd
covid_negative_labels = ["healthy"]
covid_positive_labels = ["positive_mild", "positive_moderate", "positive_asymp"]
covid_unknown_labels = ["under_validation", "no_resp_illness_exposed", "resp_illness_not_identified", "recovered_full"]

class Participant:


    def __init__(self, participant_id):
        self.id = participant_id
        data_directory = "data/Coswara_processed/Recordings"
        # data_directory = ""
        self.file_path_participant = os.path.join(data_directory, self.id)
        self.heavy_cough = AudioRecording(self.file_path_participant, type_of_recording="cough-heavy")
        # self.shallow_cough = AudioRecording(self.file_path_participant, type_of_recording="cough-shallow")
        # self.deep_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-deep")
        # self.shallow_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-shallow")
        # self.counting_fast = AudioRecording(self.file_path_participant, type_of_recording="counting-fast")
        # self.counting_normal = AudioRecording(self.file_path_participant, type_of_recording="counting-normal")
        # self.vowel_a = AudioRecording(self.file_path_participant, type_of_recording="vowel-a")
        # self.vowel_e = AudioRecording(self.file_path_participant, type_of_recording="vowel-e")
        # self.vowel_o = AudioRecording(self.file_path_participant, type_of_recording="vowel-o")

        data = pd.read_csv("data/Coswara_processed/reformatted_metadata.csv")
        self.meta_data = data[data["user_id"] == self.id].to_dict("records")[0]
        # creates an additional column for pandas dataframe index that is not really needed or wanted


    def get_label(self):
        # returns 0 if the participant is considered healthy (including recovered cases)
        # or 1 if a covid infection is assumed (positive PCR or antigen test)
        if self.meta_data["covid_health_status"] in covid_negative_labels:
            label = 0
        elif self.meta_data["covid_health_status"] in covid_positive_labels:
            label = 1
        else:
            label = None
        # TODO raise error if it cannot be said if the participant is positive or negative
        return label

