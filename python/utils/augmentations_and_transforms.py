import torch
import numpy as np


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor + np.random.randn(*tensor.shape) * self.std + self.mean
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CyclicTemporalShift:
    def __init__(self):
        pass
        # self.std = std
        # self.mean = mean

    def __call__(self, tensor):
        # [-1] should be the time step dimension of the tensor
        shift = np.random.randint(0, tensor.shape[-1])
        if isinstance(tensor, np.ndarray):
            return np.roll(tensor, shift=shift, axis=-1)
        else:
            return torch.roll(tensor, shifts=shift, dims=-1)

    def __repr__(self):
        return self.__class__.__name__


class TransferFunctionSim:
    def __init__(self, k=3.0, sinusoid_ratios=[15, 8, 5, 3, 2, 1]):
        """
        k represents the max magnitude shift
        """
        self.k = k
        self.sinusoid_ratios = sinusoid_ratios

    def __call__(self, tensor):
        # F = tensor.shape[-2]

        if isinstance(tensor, np.ndarray):
            # tensor = torch.Tensor(tensor)
            return tensor + self.get_tf(tensor)
        else:
            # tensor = np.array(tensor)
            return tensor + torch.Tensor(self.get_tf(np.array(tensor)))

    def __repr__(self):
        sinusoid_percentages = np.round(self.sinusoid_ratios/np.sum(self.sinusoid_ratios)*100, 0)
        return self.__class__.__name__ + f" k={self.k}, sin_ratios={sinusoid_percentages}"

    def get_tf(self, tensor):
        n_freq_bins = tensor.shape[-2]

        n_sinosoids = len(self.sinusoid_ratios)
        sinosoid_ratios = np.expand_dims(np.array(self.sinusoid_ratios), 1).astype("float")
        sinosoid_ratios /= sinosoid_ratios.sum()
        sinosoid_coefs = sinosoid_ratios * self.k

        freq_multiplicator = np.expand_dims(np.arange(n_sinosoids) + 1, 1)

        argument = np.tile(np.arange(n_freq_bins) / n_freq_bins / 2, (n_sinosoids, 1)).astype("float")
        argument *= freq_multiplicator
        argument += np.random.uniform(size=(n_sinosoids, 1))
        tf = np.sin(2 * np.pi * argument)
        tf *= sinosoid_coefs
        tf = tf.sum(axis=0)
        return np.expand_dims(tf, axis=1)


class RandomGain:
    def __init__(self, min_gain=-9.0, max_gain=3.0):
        self.min_gain = min_gain
        self.max_gain = -min_gain
        self.diff = self.max_gain - self.min_gain

    def __call__(self, tensor):
        gain = np.random.rand() * self.diff + self.min_gain
        tensor += gain
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f" min_gain={self.min_gain}dB, max_gain={self.max_gain}dB"
