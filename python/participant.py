from audio_recording import AudioRecording
import os


class Participant:

    def __init__(self, participant_id):
        self.id = participant_id
        # data_directory = "data/Coswara_processed/Recordings"
        data_directory = ""
        self.file_path_participant = os.path.join(data_directory, self.id)
        self.heavy_cough = AudioRecording(self.file_path_participant, type_of_recording="cough-heavy")
        self.deep_breath = AudioRecording(self.file_path_participant, type_of_recording="breathing-deep")
