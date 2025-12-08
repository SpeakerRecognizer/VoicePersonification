import argparse
import io
from concurrent import futures
from pathlib import Path
from typing import Dict, List

import grpc
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
import whisper

from services.common import _maybe_average_channels
from services.protos import voice_personification_pb2 as pb2
from services.protos import voice_personification_pb2_grpc as pb2_grpc


class WhisperRecognitionService:
    """Serve Whisper Large ASR model over gRPC."""

    def __init__(self, model_name: str = "large", device: str = None) -> None:
        raw_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(raw_device)
        self.model = whisper.load_model(model_name, device=self.device.type)
        self.sample_rate = whisper.audio.SAMPLE_RATE

    def _load_waveform(self, payload: bytes) -> np.ndarray:
        with io.BytesIO(payload) as buffer:
            waveform, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)

        waveform = _maybe_average_channels(waveform)
        tensor = torch.from_numpy(waveform)
        if sample_rate != self.sample_rate:
            tensor = F.resample(
                tensor.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            ).squeeze(0)
        return tensor.cpu().numpy()

    def transcribe(self, payload: bytes) -> Dict[str, object]:
        audio = self._load_waveform(payload)
        result = self.model.transcribe(
            audio,
            fp16=self.device.type == "cuda",
        )

        segments: List[Dict[str, object]] = []
        for seg in result.get("segments", []):
            segments.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", "").strip(),
                }
            )

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language"),
        }


class WhisperRecognitionGrpcServicer(pb2_grpc.WhisperRecognitionGrpcServiceServicer):
    def __init__(self, service: WhisperRecognitionService) -> None:
        self.service = service

    def Transcribe(self, request: pb2.AudioRequest, context: grpc.ServicerContext) -> pb2.TranscriptionResponse:
        result = self.service.transcribe(request.audio)
        return pb2.TranscriptionResponse(
            filename=request.filename,
            transcription=pb2.TranscriptionResult(
                text=result["text"],
                segments=[
                    pb2.TranscriptSegment(
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                    )
                    for segment in result["segments"]
                ],
                language=result["language"] or "",
            ),
        )


def serve(port: int = 50055, model_name: str = "large") -> None:
    service = WhisperRecognitionService(model_name=model_name)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_WhisperRecognitionGrpcServiceServicer_to_server(
        WhisperRecognitionGrpcServicer(service),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Whisper Large ASR gRPC service.")
    parser.add_argument("--port", type=int, default=50055, help="Port to bind the gRPC server.")
    parser.add_argument("--model", type=str, default="large", help="Whisper model variant to load.")
    args = parser.parse_args()
    serve(port=args.port, model_name=args.model)
