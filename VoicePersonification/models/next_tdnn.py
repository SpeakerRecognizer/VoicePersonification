import os
import math
import warnings
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from typing import Union

import numpy as np
import torch
import torchaudio
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchaudio.compliance import kaldi as torch_kaldi
from torchaudio.transforms import SlidingWindowCmn
from torchvision.transforms import Compose
from VoicePersonification.models.verification_model import VerificationModel


def kaldi_compatible_fbank(
        num_mel_bins: int,
        low_freq: int,
        high_freq: int,
        sample_frequency: int,
) -> Compose:
    return Compose(
        [
            KaldiCompatibleMelSpectrogram(
                num_mel_bins=num_mel_bins,
                low_freq=low_freq,
                high_freq=high_freq,
                sample_frequency=sample_frequency,
            ),
            SlidingWindowCmn(cmn_window=300, center=True),  # consider to use cmvn instead this one, it can lead to
                                                            # significant acceleration
            # add_channel,
        ]
    )


class KaldiCompatibleMelSpectrogram:
    def __init__(
            self,
            num_mel_bins: int,
            low_freq: int = 20,
            high_freq: int = 7900,
            sample_frequency: int = 16_000,
    ) -> None:
        self.num_mel_bins = num_mel_bins
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sample_frequency = sample_frequency

    def __call__(
            self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype == torch.int16:
            waveform = waveform.float()
        # Temporal solution to combine with torch_audiomentations
        if waveform.shape[0] == 1:
            waveform = waveform.unsqueeze(0)
        if len(waveform.shape) > 2:
            waveform = waveform[0]
        return torch_kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
            window_type="hamming",
            sample_frequency=self.sample_frequency,
            snip_edges=False,
            energy_floor=0.0,
            dither=1.0,
        )


def add_channel(fbank: torch.Tensor) -> torch.Tensor:
    return torch.unsqueeze(fbank.T, dim=0)


def cmvn(fbank: torch.Tensor) -> torch.Tensor:
    fbank = fbank - torch.mean(fbank, dim=0)
    return fbank


class ExtractFbanks80(BaseWaveformTransform):
    def __init__(self, time_last_dim=False) -> None:
        super().__init__(p=1)
        self.pipeline = kaldi_compatible_fbank(
            num_mel_bins=80,
            low_freq=20,
            high_freq=7600,
            sample_frequency=16000,
        )
        self.time_last_dim = time_last_dim

    def forward(
            self, samples: torch.Tensor, sample_rate: int = 16000, **kwargs
    ) -> torch.Tensor:
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).unsqueeze(0)
        if sample_rate != 16000:
            samples = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(samples)
        samples = self.pipeline(samples)
        if not self.time_last_dim:
            samples = torch.swapaxes(samples, 0, 1)
        if len(samples.size()) == 2:
            samples = samples.unsqueeze(0)
        return samples


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


class LayerNorm(nn.Module):  # ⚡
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, T, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, T).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]  # ⚡
            return x


class GRN(nn.Module):  # ⚡
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class NeXtTDNN(nn.Module):  #
    """ NeXt-TDNN / NeXt-TDNN-Light model.

    Args:
        in_chans (int): Number of input channels. Default: 80
        depths (tuple(int)): Number of blocks at each stage. Default: [1, 1, 1]
        dims (int): Feature dimension at each stage. Default: [256, 256, 256]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """

    def __init__(self, in_chans=80,  # in_channels: 3 -> 80
                 depths=[1, 1, 1], dims=[256, 256, 256],
                 drop_path_rate=0.,  #
                 kernel_size: Union[int, List[int]] = 7,
                 ):
        super().__init__()
        self.depths = depths
        self.stem = nn.ModuleList()
        Conv1DLayer = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")  # ⚡
        )
        self.stem.append(Conv1DLayer)

        self.stages = nn.ModuleList()  # 3 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(self.depths)):  # ⚡
            stage = nn.Sequential(
                *[TSConvNeXt(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size) for j in range(depths[i])]
                # ⚡
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

        # MFA layer
        self.MFA = nn.Sequential(  # ⚡
            nn.Conv1d(3 * dims[-1], int(3 * dims[-1]), kernel_size=1),
            LayerNorm(int(3 * dims[-1]), eps=1e-6, data_format="channels_first")  # ⚡
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):  # ⚡
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem[0](x)  # ⚡ stem_tdnn

        mfa_in = []
        for i in range(len(self.depths)):  # ⚡ 4 -> 3(len(self.depths))
            x = self.stages[i](x)
            mfa_in.append(x)

        return mfa_in  # ⚡

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, T).
        Returns:
            Tensor: Output tensor of shape (N, C, T).
        """
        x = self.forward_features(x)  # ⚡

        # MFA layer
        x = torch.cat(x, dim=1)
        x = self.MFA(x)  # Conv1d + LayerNorm_TDNN

        return x


class TSConvNeXt(nn.Module):
    """ TSConvNeXt Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0., kernel_size: List[int] = [7, 65]):
        super().__init__()
        for kernel in kernel_size:
            assert (kernel - 1) % 2 == 0, "kernel size must be odd number"
        self.projection_linear = nn.Conv1d(dim, dim, kernel_size=1)
        self.num_scale = len(kernel_size)
        self.mscconv = nn.ModuleList()
        for i in range(self.num_scale):
            self.mscconv.append(nn.Conv1d(dim // self.num_scale, dim // self.num_scale, kernel_size=kernel_size[i],
                                          padding=((kernel_size[i] - 1) // 2), groups=dim // self.num_scale))

        self.pwconv_1stage = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, T).
        Returns:
            Tensor: Output tensor of shape (N, C, T).
        """

        # MSC module
        input = x
        # Linear projection
        x = self.projection_linear(x)
        x = x.chunk(self.num_scale, dim=1)

        x_msc = []
        for i in range(self.num_scale):
            x_msc.append(self.mscconv[i](x[i]))
        x = torch.cat(x_msc, dim=1)

        x = self.act(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.pwconv_1stage(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = x + input

        # FFN module
        input = x
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        x = input + self.drop_path(x)
        return x


class VAP_BN_FC_BN(nn.Module):
    def __init__(self, channel_size, intermediate_size, embeding_size):
        super(VAP_BN_FC_BN, self).__init__()
        self.channel_size = channel_size
        self.intermediate_size = intermediate_size
        self.embeding_size = embeding_size

        self.conv_1 = nn.Conv1d(self.channel_size, self.intermediate_size, kernel_size=1)
        self.conv_2 = nn.Conv1d(self.intermediate_size, self.channel_size, kernel_size=1)
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(self.channel_size*2)

        self.fc = nn.Linear(self.channel_size*2, self.embeding_size)
        self.bn2 = nn.BatchNorm1d(self.embeding_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, channel_size, T)
        Returns:
            x: (batch_size, embeding_size)
        """
        assert x.dim() == 3, "x.dim() must be 3"

        attn = self.conv_2(self.tanh(self.conv_1(x)))
        # self.conv_1(x).shape : (batch_size, intermediate_size, T)
        # self.conv_2(self.tapn(self.conv_1(x))).shape : (batch_size, channel_size, T)
        attn = F.softmax(attn, dim=2) # (batch_size, channel_size, T)

        mu = torch.sum(x * attn, dim=2) # (batch_size, channel_size)
        rh = torch.sqrt((torch.sum((x**2) * attn, dim=2) - mu**2 ).clamp(min=1e-5))

        x = torch.cat((mu, rh), dim=1) # (batch_size, channel_size*2)
        x = self.bn1(x)

        x = self.fc(x) # (batch_size, embeding_size)
        x = self.bn2(x)

        return x


class NextTDNNVerificationModel(VerificationModel):

    def __init__(
            self,
            next_tdnn_kernel_size,
            next_tdnn_depths,
            next_tdnn_dims,
            aggreagte_chanel_size,
            aggreagte_intermediate_size,
            emb_size

            ) -> None:
        super().__init__()

        self.kaldi_fbank80 = ExtractFbanks80()
        self.feat_extractor = NeXtTDNN(kernel_size=next_tdnn_kernel_size, depths=next_tdnn_depths, dims=next_tdnn_dims)
        self.aggregation = VAP_BN_FC_BN(channel_size=int(aggreagte_chanel_size),
                                        intermediate_size=int(aggreagte_intermediate_size), embeding_size=emb_size)
        
    def forward(self, x: torch.Tensor):

        x = self.kaldi_fbank80(x)
        feats = self.feat_extractor(x)
        output = self.aggregation(feats)

        return output

    @staticmethod
    def from_pretrained(
        name_or_path: str = "itmo_personification_model_fast.ckpt", 
        repo: str = "VoicePersonificationITMO/NIRSIModels"
    ):
        if not os.path.isfile(name_or_path):
            name_or_path = hf_hub_download(repo_id=repo, filename=name_or_path)

        checkpoint = torch.load(name_or_path, map_location="cpu")
        model = NextTDNNVerificationModel(**checkpoint["args"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
