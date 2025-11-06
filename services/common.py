import io
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _maybe_average_channels(data: np.ndarray) -> np.ndarray:
    """Convert stereo data to mono by averaging channels."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


class HydraAudioService:
    """Utility to bootstrap models defined via Hydra experiment configs."""

    def __init__(
        self,
        config_path: Path,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = OmegaConf.load(str(config_path))
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = instantiate(self.cfg.model)
        self.model.eval()
        self.model.to(torch_device)

        dataset_cfg = self.cfg.dataset
        self.sample_rate = dataset_cfg.get("sample_rate", getattr(self.model, "sample_rate", 16_000))
        self.feature_extractor = None
        if dataset_cfg.get("feature_extractor") is not None:
            self.feature_extractor = instantiate(dataset_cfg.feature_extractor)

        self.device = torch_device

    def _load_waveform(self, payload: bytes) -> torch.Tensor:
        """Read audio payload and return mono waveform tensor at model's sample rate."""
        with io.BytesIO(payload) as buffer:
            waveform, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)

        waveform = _maybe_average_channels(waveform)
        waveform_tensor = torch.from_numpy(waveform)

        if sample_rate != self.sample_rate:
            waveform_tensor = F.resample(
                waveform_tensor.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            ).squeeze(0)

        return waveform_tensor.to(self.device)

    def preprocess(self, payload: bytes) -> torch.Tensor:
        """Basic preprocessing shared across services."""
        waveform = self._load_waveform(payload)
        if self.feature_extractor is not None:
            features = self.feature_extractor(
                waveform.unsqueeze(0),
                sample_rate=self.sample_rate,
            )
            if features.dim() == 2:
                features = features.unsqueeze(0)
            return features.to(self.device)
        return waveform.unsqueeze(0)

    def predict(self, payload: bytes):
        raise NotImplementedError("Implement in subclasses.")
