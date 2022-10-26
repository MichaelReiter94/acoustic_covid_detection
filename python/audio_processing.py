import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from audio_recording import AudioRecording
from time import time
from participant import Participant
import pickle

error_ids_zero_length = []
error_ids_all_zeros = []
error_ids_unknown = []
errors = {}

UPDATE_INVALID_RECORDINGS = False

DATA_PATH = os.path.join(os.getcwd(), "data/Coswara_processed/Recordings")
participant_ids = os.listdir(DATA_PATH)


participants = []
for counter, ID in enumerate(participant_ids):
    print(f"{counter + 1} / {len(participant_ids)}")
    try:
        participants.append(Participant(ID))
    # <editor-fold desc="Error Codes">
    except ValueError as e:
        # length of recording is 0
        error_ids_zero_length.append(ID)
        errors[type(e).__name__] = {"error_description": e.args[0],
                                    "error_meaning": "length of the audio file = 0 samples",
                                    "id_list": error_ids_zero_length}

    except librosa.util.exceptions.ParameterError as e:
        # all values in the audio file are 0 (but the length is > 0) resulting in NaN when normalizing with 0 as max
        error_ids_all_zeros.append(ID)
        errors[type(e).__name__] = {"error_description": e.args[0],
                                    "error_meaning": "All values in audio file are 0 resulting in NaN when normalizing",
                                    "id_list": error_ids_all_zeros}

    except Exception as e:
        # all values in the audio file are 0 (but the length is > 0) resulting in NaN when normalizing with 0 as max
        error_ids_unknown.append(ID)
        errors[type(e).__name__] = {"error_description": e.args[0],
                                    "error_meaning": "Unknown Error",
                                    "id_list": error_ids_unknown}
    # </editor-fold>



with open("data/Coswara_processed/pickles/participant_objects.pickle", "wb") as f:
    pickle.dump(participants, f)
if UPDATE_INVALID_RECORDINGS:
    with open("data/Coswara_processed/pickles/invalid_recordings.pickle", "wb") as f:
        pickle.dump(errors, f)

