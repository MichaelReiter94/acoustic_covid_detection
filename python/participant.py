from audio_recording import AudioRecording
import os
import pandas as pd


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
        # TODO maybe have covid test result be a direct attribute of the Participant
        #  class and not within the metadata dict
