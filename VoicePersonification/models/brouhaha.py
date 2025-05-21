import torch
import kaldiio
from lightning import LightningModule
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from pyannote.audio import Model


class SpeechDetectionModel(LightningModule):
    """Speech detection model using Brouhaha neural network.

    This class implements speech detection, noise level (SNR) estimation,
    and speech quality (C50) prediction using the Brouhaha model.
    The model processes audio in 20-second chunks and saves results in Kaldi format.

    Args:
        model_path (str): Path to the pretrained Brouhaha model.
        chunk_size (float): Size of audio chunks in seconds (default: 20.0).
        device (str): Device to run inference on ('cpu' or 'cuda').
    """

    def __init__(self, chunk_size: float = 20.0, device: str = 'cpu'):
        super().__init__()
        self.model = self._load_model()
        self.chunk_size = chunk_size
        self.model.eval()
        
        # For storing predictions during inference
        self.current_predictions: Dict[str, List[torch.Tensor]] = {}
        self.output_dir: Optional[Path] = None

    def _load_model(self) -> torch.nn.Module:
        """Load the pretrained Brouhaha model."""
        # Implementation depends on how the model is stored
        # This is a placeholder - replace with actual model loading
        model = Model.from_pretrained("pyannote/brouhaha", 
                              use_auth_token="ACCESS_TOKEN_GOES_HERE")

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
        speech_pred, snr, c50 = self.model(x)
        return speech_pred, snr, c50

    def process_audio_chunk(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single audio chunk.
        
        Args:
            audio (torch.Tensor): Audio tensor of shape (samples,).
            
        Returns:
            Dictionary with keys 'speech', 'snr', 'c50' and corresponding values.
        """
        with torch.no_grad():
            speech_pred, snr, c50 = self(audio.unsqueeze(0))
        return {
            'speech': speech_pred.squeeze().cpu(),
            'snr': snr.squeeze().cpu(),
            'c50': c50.squeeze().cpu()
        }

    def on_predict_start(self) -> None:
        """Initialize predictions storage at the start of prediction."""
        self.current_predictions = {'keys': [], 'speech': [], 'snr': [], 'c50': []}

    def on_predict_epoch_start(self) -> None:
        """Initialize at the start of predict epoch."""
        self.on_predict_start()

    def predict_step(self, batch: Tuple[List[str], torch.Tensor], batch_idx: int) -> None:
        """Process a batch of audio data during prediction.
        
        Args:
            batch: Tuple containing:
                - keys: List of utterance IDs
                - audio: Tensor of audio data
            batch_idx: Batch index
        """
        keys, audio = batch
        chunk_size_samples = int(self.chunk_size * self.sample_rate)
        
        for key, audio_tensor in zip(keys, audio):
            # Process audio in chunks
            for i in range(0, audio_tensor.shape[0], chunk_size_samples):
                chunk = audio_tensor[i:i+chunk_size_samples]
                # if chunk.shape[0] < chunk_size_samples:
                    # Pad last chunk if needed
                    # chunk = torch.nn.functional.pad(
                    #     chunk, 
                    #     (0, chunk_size_samples - chunk.shape[0])
                    # )
                
                results = self.process_audio_chunk(chunk)
                
                # Store results with chunk-specific key
                chunk_key = f"{key}_{i//chunk_size_samples}"
                self.current_predictions['keys'].append(chunk_key)
                self.current_predictions['speech'].append(results['speech'])
                self.current_predictions['snr'].append(results['snr'])
                self.current_predictions['c50'].append(results['c50'])

    def on_predict_epoch_end(self) -> None:
        """Save predictions to ark/scp files at the end of prediction epoch."""
        if not self.output_dir:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each output type separately
        for output_type in ['speech', 'snr', 'c50']:
            ark_path = self.output_dir / f"{output_type}.ark"
            scp_path = self.output_dir / f"{output_type}.scp"
            
            with kaldiio.WriteHelper(
                f'ark,scp:{str(ark_path)},{str(scp_path)}'
            ) as writer:
                for key, value in zip(
                    self.current_predictions['keys'],
                    self.current_predictions[output_type]
                ):
                    writer(key, value.numpy())
        
        # Clear predictions after saving
        self.current_predictions = {'keys': [], 'speech': [], 'snr': [], 'c50': []}

    def set_output_dir(self, output_dir: Union[str, Path]) -> None:
        """Set the output directory for prediction results.
        
        Args:
            output_dir: Path to directory where ark/scp files will be saved.
        """
        self.output_dir = Path(output_dir)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate expected by the model."""
        return 16000  # Typical value for speech processing, adjust if different
