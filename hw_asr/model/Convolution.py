from torch import nn

class Convolution(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

    def forward(self, spectrogram, **batch):
        return spectrogram

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
