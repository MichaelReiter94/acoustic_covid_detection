import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from audio_recording import AudioRecording
from time import time
from participant import Participant

sd.default.device[1] = 3
sd.query_devices()

################################################ Load Audio ############################################################

DATA_PATH = os.path.join(os.getcwd(), "data/Coswara_processed/Recordings")
os.chdir(DATA_PATH)
participant_id = os.listdir()[5]
participant_id = "00xKcQMmcAhX8CODgBBLOe7Dm0T2"
# os.chdir(participant_id)
participant_file_path = os.path.join(DATA_PATH, participant_id)
# audioFilePath = os.path.join(os.listdir()[5], "cough-heavy.wav")


participant = Participant(participant_id)
# participant.heavy_cough.play_audio()
# participant.deep_breath.play_audio()

participant.heavy_cough.show_waveform()
participant.deep_breath.show_waveform()
plt.show()












# random_recording = AudioRecording(participant_file_path, type_of_recording="cough-heavy")
# # t1 = time()
# PLAY_AUDIO = True
# if PLAY_AUDIO:
#     random_recording.play_audio()
# # print(f"duration of audio file: {time() - t1} seconds")
# audio, sr = random_recording.get_audio(processed=True)
# print(audio.shape, sr)
# # audio, sr = random_recording.get_audio(processed=False)
# # print(audio.shape, sr)
#
# random_recording.show_waveform(True)
# random_recording.show_waveform(False)
#
# print(random_recording.show_MFCCs().shape)
# print(random_recording.recording_type)
# print(random_recording.file_path)
# plt.show()



# ################################################ STFT ################################################################
# n_fft = 2048
# hop_size = 512
# stft = librosa.stft(audio, hop_length=hop_size, n_fft=n_fft)
# spectrogram = librosa.amplitude_to_db(np.abs(stft))
#
# # librosa.display.specshow(spectrogram)
# # plt.xlabel("Time")
# # plt.ylabel("Frequency / [Hz]")
# # plt.colorbar()
# # plt.show()

# ############################################## clipping present? #####################################################

