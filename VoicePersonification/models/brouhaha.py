import torch
import kaldiio
from lightning import LightningModule
from typing import List, Tuple, Dict, Optional, Union
from pyannote.audio import Model
import os
import numpy as np


class BrouhahaVADModel(LightningModule):
    """Speech detection model using Brouhaha neural network.

    This class implements speech detection, noise level (SNR) estimation,
    and speech quality (C50) prediction using the Brouhaha model.
    The model processes audio in 20-second chunks and saves results in Kaldi format.

    Args:
        model_path (str): Path to the pretrained Brouhaha model.
        chunk_size (float): Size of audio chunks in seconds (default: 20.0 seconds).
    """

    def __init__(self, 
                 chunk_size: float = 20.0,
                 threshold: float = 0.85):
        super().__init__()
        self.model = self._load_model()
        self.chunk_size = chunk_size
        self.threshold = threshold

    def _load_model(self) -> torch.nn.Module:
        """Load the pretrained Brouhaha model."""
        # Implementation depends on how the model is stored
        # This is a placeholder - replace with actual model loading
        model = Model.from_pretrained("pyannote/brouhaha", 
                              use_auth_token=os.environ.get("HF_TOKEN_BROUHAHA", 
                                                            os.environ.get("HF_TOKEN", 
                                                                           "Error")))

        return model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Brouhaha model.
        
        Args:
            x (torch.Tensor): Input audio tensor of shape (batch, samples).
            
        Returns:
            Tuple containing:
                - speech_pred: Binary speech prediction (0 or 1)
                - snr: Estimated SNR value
                - c50: Estimated C50 speech quality metric
        """
        # Model returns three outputs as described
        result = self.model(x.unsqueeze(0)).squeeze()
        return {
                'speech': (result[:, 0].cpu() > self.threshold).float(),
                'snr': result[:, 1].cpu(),
                'c50': result[:, 2].cpu()
            }

    def on_predict_start(self) -> None:
        """Initialize predictions storage at the start of prediction."""
        self.current_predictions = {'keys': [], 'speech': [], 'snr': [], 'c50': []}

    def on_predict_epoch_start(self) -> None:
        """Initialize at the start of predict epoch."""
        self.on_predict_start()

    def predict_step(self, batch: Tuple[List[str], torch.Tensor], batch_idx: int) -> List[Dict[str,Union[str, torch.Tensor]]]:
        """Process a batch of audio data during prediction.
        
        Args:
            batch: Tuple containing:
                - keys: List of utterance IDs
                - audio: Tensor of audio data
            batch_idx: Batch index
        """
        keys, audio = batch
        chunk_size_samples = int(self.chunk_size * self.sample_rate)
        
        final_results = []
        
        for utt_name, audio_tensor in zip(keys, audio):
            # Process audio in chunks
            
            result = {"utt_name": utt_name, 'speech': [], 'snr': [], 'c50': []}
            
            for i in range(0, audio_tensor.shape[0], chunk_size_samples):
                chunk = audio_tensor[i:i+chunk_size_samples]
                results = self(chunk)
                
                for key in results.keys():
                    result[key].append(results[key])
            for key in results.keys():
                result[key] = torch.cat(result[key]) 
            final_results.append(result)
        return final_results

    @property
    def sample_rate(self) -> int:
        """Get the sample rate expected by the model."""
        return 16000  # Typical value for speech processing, adjust if different


def output_handler(results: List[List[Dict[str,Union[str, torch.Tensor]]]],
                   output_dir:str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    vad_scp = {}
    snr_scp = {}
    c50_scp = {}
    for batch in results:
        for utt in batch:
            vad_scp[utt["utt_name"]] = utt["speech"].numpy().astype(np.float32)
            snr_scp[utt["utt_name"]] = utt["snr"].numpy().astype(np.float32)
            c50_scp[utt["utt_name"]] = utt["c50"].numpy().astype(np.float32)
    output_msg = [""]
    for file_name, container in zip(["vad", "snr", "c50"],
                                    [vad_scp, snr_scp, c50_scp]):
        
        with kaldiio.WriteHelper(f'ark,scp:{output_dir}/{file_name}.ark,{output_dir}/{file_name}.scp') as writer:
            for utt_name, value in container.items():
                writer(utt_name, value)
        output_msg += [
            f"{file_name} markup saved in {output_dir}/{file_name}.scp.",
            "You can read this file with 'VoicePersonification.utils.data.read_scp'"
        ]
    return "\n\t".join(output_msg)
