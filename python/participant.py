from audio_recording import AudioRecording
import os
import pandas as pd

# not necessary anymore... only here to remember which labels were chosen to be positive or negative
NEGATIVE_LABELS = ["healthy", "resp_illness_not_identified", "no_resp_illness_exposed"]
POSITIVE_LABELS = ["positive_mild", "positive_moderate", "positive_asymp"]
UNKNOWN_LABELS = ["under_validation", "recovered_full"]
all_types_of_recording = ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow", "counting-fast",
                          "counting-normal", "vowel-a", "vowel-e", "vowel-o"]


class Participant:
    def __init__(self, participant_id, types_of_recording, audio_params, augmentations=None):
        self.id = participant_id
        self.is_augmented = augmentations is not None
        data_directory = "data/Coswara_processed/Recordings"
        file_path_participant = os.path.join(data_directory, self.id).replace("\\", "/")
        self.recordings = {}
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
                                                             augmentations=augmentations)

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
