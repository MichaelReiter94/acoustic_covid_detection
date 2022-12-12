import os
from os import stat_result

import librosa
import sounddevice as sd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, Gain, Shift, TimeStretch, Trim


def get_logmel_spectrum(audio, sr):
    # total duration = 2.6 seconds (512 samples hopsize = 11.6ms --> 11.6ms * 224 (frames for resnet input) = 2.6s
    target_sr = 44100
    if target_sr != sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    n_fft = 1024 * 16  # only for zero padding to increas frequency resolution
    hop_size = 512  # 11.6ms
    n_mels = 224

    window_length = 1024  # 23.2ms
    mel_spect = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
                                               win_length=window_length, n_mels=n_mels,
                                               sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)

    # mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, htk=True, fmin=0, fmax=target_sr/2)
    return librosa.power_to_db(mel_spect)


def get_3_channel_logmel_spectrum(audio, sr):
    # total duration = 2.6 seconds (512 samples hopsize = 11.6ms --> 11.6ms * 224 (frames for resnet input) = 2.6s
    target_sr = 44100
    if target_sr != sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    n_fft = 1024 * 16  # only for zero padding to increas frequency resolution
    hop_size = 512  # 11.6ms
    n_mels = 224
    base_window_legth = 1024

    window_length = base_window_legth  # 11ms
    channel1 = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
                                              win_length=window_length, n_mels=n_mels,
                                              sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)

    window_length = base_window_legth * 2  # 46ms
    channel2 = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
                                              win_length=window_length, n_mels=n_mels,
                                              sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)

    window_length = base_window_legth * 4  # 185ms
    channel3 = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
                                              win_length=window_length, n_mels=n_mels,
                                              sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)

    channel1 = librosa.power_to_db(channel1)
    channel2 = librosa.power_to_db(channel2)
    channel3 = librosa.power_to_db(channel3)

    mel_spect = np.stack([channel1, channel2, channel3])
    # mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, htk=True, fmin=0, fmax=target_sr/2)
    return mel_spect


def get_15_MFCCs(audio, sr):
    target_sr = 22050
    if target_sr != sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

    hop_size = 512
    nfft = 2048
    n_MFCCs = 15
    mfccs = librosa.feature.mfcc(y=audio, n_fft=nfft, hop_length=hop_size, n_mfcc=n_MFCCs, sr=target_sr)
    return mfccs


class AudioRecording:

    def __init__(self, data_path, type_of_recording, augmentations=None):
        # are all files .wav?
        self.file_path = f"{os.path.join(data_path, type_of_recording)}.wav".replace("\\", "/")
        self.recording_type = type_of_recording
        self.augmentations = augmentations
        # self.target_duration_seconds = 6

        # self.n_samples_target = self.target_sample_rate * self.target_duration_seconds
        # self.target_time_steps = int(self.n_samples_target // self.hop_size + 1)

        self.original_duration = None
        self.original_duration_trimmed_silence = None

        audio, self.original_sr = self.get_audio(trim_silence_below_x_dB=48)

        if self.augmentations is not None:
            audio = self.augmentations(audio, self.original_sr)

        # self.MFCCs = get_15_MFCCs(audio, sr=self.original_sr)
        self.logmel = get_logmel_spectrum(audio, self.original_sr)
        # self.logmel_3c = get_3_channel_logmel_spectrum(audio, self.original_sr)

    def get_audio(self, trim_silence_below_x_dB=48):
        audio, sr = librosa.load(self.file_path, sr=None)
        self.original_duration = round(len(audio) / sr, 2)

        if trim_silence_below_x_dB is not None:
            audio, _ = librosa.effects.trim(audio, top_db=trim_silence_below_x_dB)
            self.original_duration_trimmed_silence = round(len(audio) / sr, 2)

        print(f"Duration of the Recording: {round(len(audio) / sr, 2)}")
        return audio, sr

    def show_waveform(self, trim_silence_below_x_dB=48):
        plt.figure()
        audio, sr = self.get_audio(trim_silence_below_x_dB)
        librosa.display.waveshow(audio, sr=sr)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        # plt.show()

    def show_MFCCs(self):
        plt.figure()
        librosa.display.specshow(self.MFCCs)
        plt.xlabel("Time Frame")
        plt.ylabel("MFCC")
        plt.colorbar()

    def play_audio(self, trim_silence_below_x_dB=48):
        data, sr = self.get_audio(trim_silence_below_x_dB)
        sd.play(data, sr)
        sd.wait()

    def play_randomly_augmented_audio(self):
        data, sr = self.get_audio(trim_silence_below_x_dB=48)
        if self.augmentations is not None:
            data = self.augmentations(data, sr)
        else:
            print("There are no augmentations specified for this instance. This recording is original.")
        sd.play(data, sr)
        sd.wait()

    def show_logmel(self):
        target_sr = 44100
        window_length = 1024  # 23.2ms
        n_fft = 1024 * 16  # only for zero padding to increas frequency resolution
        hop_size = 512  # 11.6ms
        n_mels = 224
        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, htk=True, fmin=0, fmax=target_sr / 2)

        plt.figure()
        librosa.display.specshow(self.logmel, x_axis='time', y_axis="log", cmap="magma",
                                 hop_length=hop_size, sr=target_sr, y_coords=mel_frequencies)
        plt.colorbar(format="%+2.f dB")
        print(self.logmel.shape)

    def show_3channel_logmel(self, time_frame=None, frequency_range=None):
        """
        time_frame: tuple of starting time in seconds and end time in seconds to view
        frequency_range: tuple of starting and end frequency you want to be shown
        if None is specified the full spectrum/time will be displaid
        """
        target_sr = 44100
        hop_size = 512  # 11.6ms
        n_mels = 224
        titles = ["512 samples = 11.6ms", "2048 samples = 46.4ms", "8192 samples = 185.6ms"]
        fmin = 0

        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, htk=True, fmin=fmin, fmax=target_sr / 2)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[18, 6])
        for i, ax in enumerate(axes):
            img = librosa.display.specshow(self.logmel_3c[i], x_axis='time', y_axis='log', cmap="magma",
                                           hop_length=hop_size, sr=target_sr, y_coords=mel_frequencies, ax=ax)
            if time_frame is not None:
                ax.set(xlim=[time_frame[0], time_frame[1]])
            if frequency_range is not None:
                ax.set(ylim=[frequency_range[0], frequency_range[1]])
            ax.set(title=f"fft window  length: {titles[i]}")
            fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.tight_layout()
