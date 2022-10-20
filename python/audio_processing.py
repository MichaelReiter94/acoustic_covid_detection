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

# set audio playback device
sd.default.device[1] = 3

# get list of all participants (= id's = folder names)
DATA_PATH = os.path.join(os.getcwd(), "data/Coswara_processed/Recordings")
participant_ids = os.listdir(DATA_PATH)
participant_ids = participant_ids[:5]

participants = []

erronous_ids = ['0dE8pniDGCZp0gYMy23fvjpZBDr1', '0dfJArS4xbNhay72hpzOvlHkpIj2', '0Js6ZUZQ9NUnu568Fh7B6mZ1R8o1',
                '0M4YhLM7FwWO5IjCqhi7MlBhvJv2', '0mkAUAbpROMIFjKpdFxlUnYTV262', '1NQvmLMrJyTwrmbNwAm6wDT4wpz2',
                '1R9RcXlOOLhCmAYuqdRSGP3b0OD3', '1Vm44aluw7PMy2mfETHc3MvwQxx1']

error_ids_zero_length = []
error_ids_all_zeros = []
error_ids_unknown = []
errors = {}

for counter, ID in enumerate(participant_ids):
    print(f"{counter + 1} / {len(participant_ids)}")
    try:
        participants.append(Participant(ID))

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



with open("data/Coswara_processed/pickles/participant_objects_subset.pickle", "wb") as f:
    pickle.dump(participants, f)



# with open("data/Coswara_processed/pickles/unvalid_recordings.pickle", "wb") as f:
#     pickle.dump(errors, f)
# # there are ~ 100 participants who recorded invalid audio:
# #       43 of those are of length 0
# #       55 only contain 0 within the (non-zero length) audio signal
# #       2648 valid participants are left after that

