import torch

from VoicePersonification.models.verification_model import VerificationModel
from speechbrain.inference import EncoderClassifier

class ECAPATDNNModel(VerificationModel):

    def __init__(self, pretrained_model_path: str = "yangwang825/ecapa-tdnn-vox2") -> None:
        super().__init__()
        self.extractor = EncoderClassifier.from_hparams(pretrained_model_path)
        
    def forward(self, wavs: torch.Tensor):
        self.extractor.device = self.device
        return self.extractor.encode_batch(wavs)