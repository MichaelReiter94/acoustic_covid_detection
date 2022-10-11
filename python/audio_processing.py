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


participants = []
t1 = time()
for ID in participant_ids[:10]:
    participants.append(Participant(ID))
print(time() - t1)
t1 = time()

with open("data/Coswara_processed/pickles/participant_objects.pickle", "wb") as f:
    pickle.dump(participants, f)
print(time() - t1)









