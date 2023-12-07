import torch

from processors.autodiff.compressor import Compressor
from processors.autodiff.peq import ParametricEQ
from processors.autodiff.fir import FIRFilter


class AutodiffChannel(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.fc = torch.nn.Linear(22, 18)
        self.sigmoid = torch.nn.Sigmoid()
        self.peq = ParametricEQ(sample_rate)
        # self.comp = Compressor(sample_rate)
        self.ports = [self.peq.ports]
        # self.ports = [self.peq.ports, self.comp.ports]
        self.num_control_params = self.peq.num_control_params
        # self.num_control_params = (self.peq.num_control_params + self.comp.num_control_params)

    def forward(self, x, p, sample_rate=24000, **kwargs):

        # split params between EQ and Comp.
        # p_peq = p[:, : self.peq.num_control_params]
        # p_comp = p[:, self.peq.num_control_params :]

        # y = self.peq(x, p_peq, sample_rate)
        # y = self.comp(y, p_comp, sample_rate)

        p = self.sigmoid(self.fc(p))
        y = self.peq(x, p, sample_rate)

        return y
