import os
import librosa
import sounddevice as sd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt



class AudioRecording:


    def __init__(self, data_path, type_of_recording):
        # relativ or absolute filepath ?
        # are all files .wav?
        self.file_path = f"{os.path.join(data_path, type_of_recording)}.wav"
        # self.recording_type = os.path.basename(file_path).split(".")[0]
        self.recording_type = type_of_recording
        self.target_sample_rate = 44100
        self.target_duration_seconds = 2
        self.n_fft = 2048
        self.hop_size = 512
        self.n_MFCCs = 15
        self.n_samples_target = self.target_sample_rate * self.target_duration_seconds
        self.original_sample_rate = None
        self.MFCCs = self.get_MFCCs()



    def get_MFCCs(self):
        audio, sr = self.get_audio(processed=True)
        mfccs = librosa.feature.mfcc(y=audio,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_size,
                                     n_mfcc=self.n_MFCCs)
        return mfccs





    def get_audio(self, processed=True):
        """ Prcessing of original audio if 'processed=True':\n
                - Silent leading and trailing parts are trimmed\n
                - the file is resampled to samplerate defined within the class\n
                - amplitude is normalized\n
                - cut or padded to specified length"""

        audio, file_sample_rate = librosa.load(self.file_path, sr=None)
        self.original_sample_rate = file_sample_rate
        if processed:
            # resample
            audio = librosa.resample(y=audio, orig_sr=file_sample_rate, target_sr=self.target_sample_rate)
            file_sample_rate = self.target_sample_rate
            # trim
            audio, _ = librosa.effects.trim(audio, top_db=54)
            # cut or pad to self.target_duration_seconds
            if len(audio) < self.n_samples_target:
                audio = np.pad(audio, (0, self.n_samples_target - len(audio)))
            elif len(audio) > self.n_samples_target:
                audio = audio[:self.n_samples_target]
            # normalize amplitude
            audio = audio / np.max(np.abs(audio))
        return audio, file_sample_rate


    def show_waveform(self, processed=True):
        plt.figure()
        audio, sr = self.get_audio(processed)
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
        return self.MFCCs
        # plt.show()


    def play_audio(self, processed=True):
        data, sr = self.get_audio(processed)
        sd.play(data, sr)
        sd.wait()
