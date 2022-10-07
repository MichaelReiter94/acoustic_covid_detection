from audio_recording import AudioRecording
import os
import pandas as pd
import json


class Participant:

    def __init__(self, participant_id):
        self.id = participant_id
        # data_directory = "data/Coswara_processed/Recordings"
        data_directory = ""
        self.file_path_participant = os.path.join(data_directory, self.id)
        self.heavy_cough = AudioRecording(self.file_path_participant, type_of_recording="cough-heavy")
        self.deep_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-deep")
        self.shallow_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-shallow")
        self.shallow_cough = AudioRecording(self.file_path_participant, type_of_recording="cough-shallow")
        self.counting_fast = AudioRecording(self.file_path_participant, type_of_recording="counting-fast")
        self.counting_normal = AudioRecording(self.file_path_participant, type_of_recording="counting-normal")
        self.vowel_a = AudioRecording(self.file_path_participant, type_of_recording="vowel-a")
        self.vowel_e = AudioRecording(self.file_path_participant, type_of_recording="vowel-e")
        self.vowel_o = AudioRecording(self.file_path_participant, type_of_recording="vowel-o")

        with open(os.path.join(self.file_path_participant, "metadata.json")) as f:
            self.meta_data = json.load(f)

        # self.meta_data = pd.read_json(os.path.join(self.file_path_participant, "metadata.json"))
