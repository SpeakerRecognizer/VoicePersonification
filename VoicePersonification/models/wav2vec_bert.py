from enum import Enum
import torch
from torch import nn
from transformers import Wav2Vec2BertModel, Wav2Vec2BertConfig, AutoFeatureExtractor
from transformers import AutoFeatureExtractor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from typing import List, Union
from torch.nn.functional import relu
from VoicePersonification.models.verification_model import VerificationModel
from torch.nn import Parameter
import math



# features 
class Wav2Vec2BertFeatures(BaseWaveformTransform):

    def __init__(self, feats_name: str = "facebook/w2v-bert-2.0"):
        super().__init__()
        self.feats_extractor = AutoFeatureExtractor.from_pretrained(feats_name)

    def forward(
        self, samples: torch.Tensor, sample_rate: int = 16_000, **kwargs
    ):
        samples = samples.squeeze()
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)

        feats = self.feats_extractor(samples, sampling_rate=sample_rate,  return_tensors="pt")

        return feats["input_features"].permute(0, 2, 1).squeeze()


# extractor
class Wav2Vec2BertEncoder(nn.Module):

    def __init__(self, model_name: str, n_layer: int) -> None:
        super().__init__()
        config = Wav2Vec2BertConfig.from_pretrained(model_name)
        if n_layer >= 0:
            config.update({
                "num_hidden_layers": n_layer
            })
        self.w2v = Wav2Vec2BertModel.from_pretrained(model_name, config=config)

    def forward(self, x: torch.Tensor):
        input_features = x.permute(0, 2, 1)
        model_output = self.w2v(input_features=input_features)
        
        return model_output["last_hidden_state"].transpose(-1, -2)

# frame level 

class TDNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      padding=padding, 
                                      bias=bias)
    def forward(self, x: torch.Tensor):
        return self.conv1d(x)
          

# pooling
    
class StatPoolMode(Enum):
    M = 0
    V = 1
    MV = 2


class StatPoolLayer(nn.Module):

    @classmethod
    def by_string_mode(cls, mode: str):
        mode = StatPoolMode[mode]
        return cls(mode)

    """ Building of SGPK (Snyder, Garcia-Romero, Povey, Khudanpur) statistics pooling layer's model. Implementation
    is based on work Snyder D. et al. X-vectors: Robust DNN embeddings for speaker recognition // ICASSP, Calgary,
    Canada. – 2018
    """
    def __init__(self, stat_pool_mode: StatPoolMode, dim=-1):
        super(StatPoolLayer, self).__init__()

        ################################################################################################################
        #
        # mode:      0 - average is calculated only;
        #            1 - standard deviation is calculated only,
        #            2 - average and standard deviation is calculated
        #
        ################################################################################################################
        self.mode = stat_pool_mode
        self.dim = dim

    def forward(self, x, **kwargs):
        mean_x = x.mean(self.dim)
        mean_x2 = x.pow(2).mean(self.dim)

        std_x = relu(mean_x2 - mean_x.pow(2)).sqrt()
        
        if self.mode == StatPoolMode.M:
            out = mean_x
        elif self.mode == StatPoolMode.V:
            out = std_x
        elif self.mode == StatPoolMode.MV:
            out = torch.cat([mean_x, std_x], dim=-1)
        else:
            raise ValueError('Operation\'s mode is incorrect')
        out = torch.flatten(out, 1)
        return out


# segment_level 

class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return torch.max(self.linear1(x), self.linear2(x))

class MaxoutSegmentLevel(nn.Module):

    def __init__(self, input_dim: Union[int, List[int]], output_dim: Union[int, List[int]], enable_batch_norm: bool, fixed: bool = False):
        super(MaxoutSegmentLevel, self).__init__()

        if isinstance(input_dim, int) and isinstance(output_dim, int):
            input_dim = [input_dim]
            output_dim = [output_dim]
        self.num_layers = len(input_dim)
        self.enable_batch_norm = enable_batch_norm
        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.fixed = fixed

        for idx in range(self.num_layers):
            self.layers.append(MaxoutLinear(input_dim[idx], output_dim[idx]))

            if self.enable_batch_norm:
                self.bn.append(nn.BatchNorm1d(output_dim[idx], affine=False))

    def forward(self, x, **kwargs):

        for idx in range(self.num_layers):
            x = self.layers[idx](x)

            if self.enable_batch_norm:
                x = self.bn[idx](x)
        return x



# head 
class CurricularAAM(nn.Module):
    def __init__(self, in_features, out_features, sub_centers=1,
                 m=0.35, s=32., fixed=False):
        super(CurricularAAM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_centers = sub_centers
        self.register_buffer('m', torch.tensor(m))
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features*sub_centers))
        self.fixed = fixed
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def update(self, v):
        m = getattr(self, "m")
        setattr(self, "m", torch.tensor(v, dtype=m.dtype, device=m.device))

    def forward(self, embbedings,  label=None,
                add_labels=None,
                lambda_f=None):
        embbedings_norm = torch.nn.functional.normalize(embbedings, dim=1)
        kernel_norm = torch.nn.functional.normalize(self.kernel, dim=0)

        cos_theta = torch.mm(embbedings_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        if self.sub_centers > 1:
            cos_theta = cos_theta.reshape(-1, self.out_features, self.sub_centers)
            cos_theta, _ = cos_theta.max(axis=-1)
        if self.training:
            with torch.no_grad():
                origin_cos = cos_theta.clone()
            target_logit = cos_theta[torch.arange(0, embbedings_norm.size(0)), label].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            cos_theta_m = cos_theta_m.to(target_logit.dtype)
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            cos_theta[mask] = hard_example * (self.t.to(hard_example.dtype) + hard_example)
            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s
            return output, origin_cos * self.s
        else:
            return cos_theta


# model 
class Wav2VecBERTModel(VerificationModel):
    def __init__(self):
        super().__init__()
        self.feat_extractor = Wav2Vec2BertEncoder("VoicePersonificationITMO/NIRSIModels/itmo_personification_model_segmentation.ckpt", 8)
        self.frame_level = torch.nn.Conv1d(1024, 2048, 
                                      kernel_size=1, 
                                      stride=1, 
                                      padding=0, 
                                      bias=True)
        self.pooling = StatPoolLayer(StatPoolMode.MV)
        self.segment_level = MaxoutSegmentLevel(4096, 512, True)
    
    def forward(self, features):
        x = self.feat_extractor(features)
        x = self.frame_level(x)
        x = self.pooling(x)
        x = self.segment_level(x)
        return x

