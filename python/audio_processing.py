import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from audio_recording import AudioRecording
from time import time


def play_audio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()


sd.default.device[1] = 3
sd.query_devices()


################################################ Load Audio ############################################################

DATA_PATH = os.path.join(os.getcwd(), "data/Coswara_processed/Recordings")
os.chdir(DATA_PATH)
# audioFilePath = os.path.join(os.getcwd(), os.listdir()[5])
audioFilePath = os.path.join(os.listdir()[5], "cough-heavy.wav")
# audio, sample_rate = librosa.load(audioFilePath)
# librosa.display.waveshow(audio, sr=sample_rate)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()


random_recording = AudioRecording(audioFilePath)
# t1 = time()
# random_recording.play_audio()
# print(f"duration of audio file: {time() - t1} seconds")
audio, sr = random_recording.get_audio(processed=True)
print(audio.shape, sr)
audio, sr = random_recording.get_audio(processed=False)
print(audio.shape, sr)

random_recording.show_waveform(True)
random_recording.show_waveform(False)

plt.show()


#
# ################################################## FFT #################################################################
#
# n_samples = len(audio)
# fft = np.fft.fft(audio)
# spectrum_magn = np.abs(fft)[:n_samples // 2]
# frequency = np.linspace(0, sample_rate, n_samples // 2)
# # plt.plot(frequency, spectrum_magn)
# # plt.xlabel("Frequency / [Hz]")
# # plt.ylabel("Amplitude")
# # plt.show()
#
# ################################################ STFT #################################################################
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
#
# ################################################ MFCCs #################################################################
#
# MFCCs = librosa.feature.mfcc(y=audio, n_fft=n_fft, hop_length=hop_size, n_mfcc=13)
#
# # librosa.display.specshow(MFCCs)
# # plt.xlabel("Time")
# # plt.ylabel("MFCCs / [Hz]")
# # plt.colorbar()
# # plt.show()
#
# ######################################## Audio PLayback from ndarray ###################################################
#
# # sd.play(audio, sample_rate)
# # sd.wait()
#
# ############################################### Cut File to 7s #########################################################
#
# print(audio.shape)
# print(sample_rate)
# duration_s = len(audio) / sample_rate
# print(round(duration_s, 2))
# n_samples_from_s = 2 * sample_rate
# print(n_samples_from_s)
#
# short_audio = audio[:n_samples_from_s]
# sd.play(short_audio, sample_rate)
# sd.wait()
#
# # plt.figure()
# librosa.display.waveshow(short_audio, sr=sample_rate)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# # plt.show()
#
# ############################# Trim leading and trailing silence from an audio signal. ##################################
#
# trimmed_audio, index = librosa.effects.trim(audio, top_db=54)
# plt.figure()
# librosa.display.waveshow(trimmed_audio, sr=sample_rate)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()
#
#
# ############################################## clipping present? #######################################################
#
#
