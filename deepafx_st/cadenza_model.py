import torch
import torchaudio
from torch import nn

from deepafx_st.utils import DSPMode
from deepafx_st.models.encoder import SpectralEncoder
from deepafx_st.models.controller import StyleTransferController
from processors.autodiff.channel import AutodiffChannel
from system import System


class CadenzaModel(nn.Module):
    def __init__(self, dsp_sample_rate=24000):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dsp_sample_rate = dsp_sample_rate
        self.dsp_mode = DSPMode.NONE

        self.processor = AutodiffChannel(self.dsp_sample_rate)

        # load pre-loaded weights
        checkpoint_path = "../../DeepAFx-ST/checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt"
        system = System.load_from_checkpoint(checkpoint_path)
        self.encoder = system.encoder
        
        # self.encoder = SpectralEncoder(
        #     self.processor.num_control_params,
        #     self.dsp_sample_rate,
        #     encoder_model="efficient_net",
        #     embed_dim=1024,
        #     width_mult=1,
        # )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.controller = StyleTransferController(
            self.processor.num_control_params,
            system.encoder.embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gain: torch.Tensor,
        data_sample_rate: int = 24000,
    ):
        """Forward pass through the system subnetworks.

        Args:
            x (tensor): Input audio tensor with shape (batch x 1 x samples)
            y (tensor): Target audio tensor with shape (batch x 1 x samples)
            e_y (tensor): Target embedding with shape (batch x edim)
            z (tensor): Bottleneck latent.
            dsp_mode (DSPMode): Mode of operation for the DSP blocks.
            analysis_length (optional, int): Only analyze the first N samples.
            data_sample_rate (optional, int): Sampling rate of data.

        You must supply target audio `y`, `z`, or an embedding for the target `e_y`.

        Returns:
            y_hat (tensor): Output audio.
            p (tensor):
            e (tensor):

        """
        bs, chs, samp = x.size()
        
        if data_sample_rate != self.dsp_sample_rate:
            x_enc = torchaudio.transforms.Resample(data_sample_rate, self.dsp_sample_rate).to(x.device)(x)
            if y is not None:
                y_enc = torchaudio.transforms.Resample(data_sample_rate, self.dsp_sample_rate).to(y.device)(y)
        else:
            x_enc = x
            y_enc = y

        e_x = self.encoder(x_enc)  # generate latent embedding for input
        e_y = self.encoder(y_enc)  # generate latent embedding for target

        # learnable comparision
        p = self.controller(e_x, e_y)
        p_with_gain = torch.cat((p, gain), dim=1)
        y_hat = self.processor(x, p_with_gain).requires_grad_()

        # process audio conditioned on parameters
        # if there are multiple channels process them using same parameters
        # y_hat = torch.zeros(x.shape).type_as(x).requires_grad_()
        # for ch_idx in range(chs):
        #     y_hat_ch = self.processor(x[:, ch_idx : ch_idx + 1, :], p_with_gain)
        #     y_hat[:, ch_idx : ch_idx + 1, :] = y_hat_ch

        return y_hat, p_with_gain, e_x
