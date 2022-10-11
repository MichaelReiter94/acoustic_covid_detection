import os
import numpy as np
# from participant import Participant
import matplotlib.pyplot as plt
import pickle
import random

with open("data/Coswara_processed/pickles/participant_objects.pickle", "rb") as f:
    participants = pickle.load(f)

sample_participant = random.choice(participants)
print(sample_participant.meta_data)
sample_participant.heavy_cough.play_audio()
sample_participant.heavy_cough.show_waveform()
plt.show()

