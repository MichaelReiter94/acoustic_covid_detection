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

    # def __repr__(self):
    #     return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

