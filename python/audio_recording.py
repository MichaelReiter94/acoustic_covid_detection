import os
import librosa
import sounddevice as sd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, Gain, Shift, TimeStretch, Trim


# def get_3_channel_logmel_spectrum(audio, sr):
#     # total duration = 2.6 seconds (512 samples hopsize = 11.6ms --> 11.6ms * 224 (frames for resnet input) = 2.6s
#     target_sr = 44100
#     if target_sr != sr:
#         audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
#     n_fft = 1024 * 16  # only for zero padding to increas frequency resolution
#     hop_size = 512  # 11.6ms
#     n_mels = 224
#     base_window_legth = 1024
#
#     window_length = base_window_legth  # 11ms
#     channel1 = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
#                                               win_length=window_length, n_mels=n_mels,
#                                               sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)
#
#     window_length = base_window_legth * 2  # 46ms
#     channel2 = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
#                                               win_length=window_length, n_mels=n_mels,
#                                               sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)
#
#     window_length = base_window_legth * 4  # 185ms
#     channel3 = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, hop_length=hop_size,
#                                               win_length=window_length, n_mels=n_mels,
#                                               sr=target_sr, fmin=0, fmax=target_sr / 2, htk=True)
#
#     channel1 = librosa.power_to_db(channel1)
#     channel2 = librosa.power_to_db(channel2)
#     channel3 = librosa.power_to_db(channel3)
#
#     mel_spect = np.stack([channel1, channel2, channel3])
#     # mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, htk=True, fmin=0, fmax=target_sr/2)
#     return mel_spect


class AudioRecording:
    def __init__(self, data_path, type_of_recording, audio_parameters, augmentations=None):
        combined_recordings = {
            "combined_coughs": ["cough-heavy", "cough-shallow"],
            "combined_breaths": ["breathing-deep", "breathing-shallow"],
            "combined_vowels": ["vowel-a", "vowel-e", "vowel-o"],
            "combined_speech": ["counting-normal", "counting-fast"]
        }

        if type_of_recording in combined_recordings:
            self.file_path = []
            for combined_rec_type in combined_recordings[type_of_recording]:
                self.file_path.append(f"{os.path.join(data_path, combined_rec_type)}.wav".replace("\\", "/"))
        else:
            self.file_path = f"{os.path.join(data_path, type_of_recording)}.wav".replace("\\", "/")

        self.recording_type = type_of_recording
        self.augmentations = augmentations
        self.hop_size = audio_parameters["hop_size"]
        self.window_length = audio_parameters["window_length"]
        self.n_fft = audio_parameters["n_fft"]
        self.n_features = audio_parameters["n_features"]
        self.fmin = audio_parameters["fmin"]
        self.fmax = audio_parameters["fmax"]
        self.target_sr = audio_parameters["sample_rate"]
        self.type_of_features = audio_parameters["type_of_features"]

        self.original_duration = None
        self.original_duration_trimmed_silence = None

        audio, self.original_sr = self.get_audio(trim_silence_below_x_dB=30)

        if self.augmentations is not None:
            audio = self.augmentations(audio, self.original_sr)

        if self.target_sr != self.original_sr:
            audio = librosa.resample(y=audio, orig_sr=self.original_sr, target_sr=self.target_sr)

        if self.type_of_features.lower() == "mfcc":
            self.features = self.get_mfccs(audio)
        elif self.type_of_features.lower() == "logmel":
            self.features = self.get_logmel_spectrum(audio)
        else:
            raise KeyError

    def get_mfccs(self, audio):
        # if self.target_sr != self.original_sr:
        #     audio = librosa.resample(y=audio, orig_sr=self.original_sr, target_sr=self.target_sr)

        mfccs = librosa.feature.mfcc(y=audio, win_length=self.window_length, n_fft=self.n_fft, htk=True, fmax=self.fmax,
                                     fmin=self.fmin, hop_length=self.hop_size, n_mfcc=self.n_features,
                                     sr=self.target_sr, n_mels=224)
        return mfccs

    def get_logmel_spectrum(self, audio):
        # total duration = 2.6 seconds (512 samples hopsize = 11.6ms --> 11.6ms * 224 (frames for resnet input) = 2.6s
        # if self.target_sr != self.original_sr:
        #     audio = librosa.resample(y=audio, orig_sr=self.original_sr, target_sr=self.target_sr)

        mel_spect = librosa.feature.melspectrogram(y=audio, n_fft=self.n_fft, hop_length=self.hop_size,
                                                   win_length=self.window_length, n_mels=self.n_features,
                                                   sr=self.target_sr, fmin=self.fmin, fmax=self.fmax, htk=True)
        return librosa.power_to_db(mel_spect)

    def get_audio(self, trim_silence_below_x_dB=40):

        if isinstance(self.file_path, str):
            audio, sr = librosa.load(self.file_path, sr=None)
            if trim_silence_below_x_dB is not None:
                audio, _ = librosa.effects.trim(audio, top_db=trim_silence_below_x_dB)
                audio = librosa.util.normalize(audio)

        else:
            audio = np.array([]).astype("float32")
            for path in self.file_path:
                audio_temp, sr = librosa.load(path, sr=None)
                if trim_silence_below_x_dB is not None:
                    audio_temp, _ = librosa.effects.trim(audio_temp, top_db=trim_silence_below_x_dB)
                    audio_temp = librosa.util.normalize(audio_temp)

                audio = np.concatenate([audio, audio_temp])
                # trim again after combining... if one of the recordings was silent it only gets removed here
            audio, _ = librosa.effects.trim(audio, top_db=trim_silence_below_x_dB)

        # normalize. I don't know why I removed this again. maybe it was an accident?
        # self.original_duration = round(len(audio) / sr, 2)
        self.original_duration_trimmed_silence = round(len(audio) / sr, 2)
        print(f"Duration of the Recording: {round(len(audio) / sr, 2)}")
        # reintroduce normalization? why is it gone?
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
        librosa.display.specshow(self.features)
        print(self.features.shape)

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
        mel_frequencies = librosa.mel_frequencies(n_mels=self.n_features, htk=True, fmin=self.fmin, fmax=self.fmax)

        plt.figure()
        librosa.display.specshow(self.features, x_axis='time', y_axis="log", cmap="magma",
                                 hop_length=self.hop_size, sr=self.target_sr, y_coords=mel_frequencies)
        plt.colorbar(format="%+2.f dB")
        print(self.features.shape)

    def show_features(self):
        if self.type_of_features.lower() == "mfcc":
            self.show_MFCCs()
        elif self.type_of_features.lower() == "logmel":
            self.show_logmel()
        else:
            raise KeyError

    def __str__(self):
        return f"type: {self.recording_type}\n" \
               f"hop size: {self.hop_size}\n" \
               f"window length: {self.window_length}\n" \
               f"n_features: {self.n_features}\n" \
               f"fmax: {self.fmax}\n" \
               f"sample rate: {self.target_sr}"
        # f"nfft: {self.n_fft}\n" \

    def __repr__(self):
        return f"class: {self.__class__.__name__} | {self.recording_type}"

    # def show_3channel_logmel(self, time_frame=None, frequency_range=None):
    #     """
    #     time_frame: tuple of starting time in seconds and end time in seconds to view
    #     frequency_range: tuple of starting and end frequency you want to be shown
    #     if None is specified the full spectrum/time will be displaid
    #     """
    #     # target_sr = 44100
    #     # hop_size = 512  # 11.6ms
    #     # n_mels = 224
    #     # fmin = 0
    #
    #     # titles = ["512 samples = 11.6ms", "2048 samples = 46.4ms", "8192 samples = 185.6ms"]
    #     mel_frequencies = librosa.mel_frequencies(n_mels=self.n_features, htk=True, fmin=self.fmin, fmax=self.fmax)
    #
    #     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[18, 6])
    #     for i, ax in enumerate(axes):
    #         img = librosa.display.specshow(self.features[i], x_axis='time', y_axis='log', cmap="magma",
    #                                        hop_length=self.hop_size, sr=self.target_sr, y_coords=mel_frequencies, ax=ax)
    #         if time_frame is not None:
    #             ax.set(xlim=[time_frame[0], time_frame[1]])
    #         if frequency_range is not None:
    #             ax.set(ylim=[frequency_range[0], frequency_range[1]])
    #         # ax.set(title=f"fft window  length: {titles[i]}")
    #         fig.colorbar(img, ax=ax, format="%+2.f dB")
    #     plt.tight_layout()
