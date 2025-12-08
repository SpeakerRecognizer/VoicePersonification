import argparse
from concurrent import futures
from pathlib import Path
from typing import Dict, List

import grpc
import torch

from .common import HydraAudioService
from .protos import voice_personification_pb2 as pb2
from .protos import voice_personification_pb2_grpc as pb2_grpc


class BrouhahaVADService(HydraAudioService):
    """Serve Brouhaha VAD predictions for single audio files."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path)
        self.chunk_samples = int(self.model.chunk_size * self.model.sample_rate)

    def predict(self, payload: bytes) -> Dict[str, List[float]]:
        waveform = self._load_waveform(payload)

        outputs: Dict[str, List[torch.Tensor]] = {"speech": [], "snr": [], "c50": []}

        with torch.no_grad():
            for start in range(0, waveform.shape[-1], self.chunk_samples):
                chunk = waveform[start : start + self.chunk_samples]
                if chunk.numel() == 0:
                    continue
                chunk = chunk.to(self.device)
                result = self.model(chunk)
                for key in outputs:
                    outputs[key].append(result[key].cpu())

        merged = {key: torch.cat(value).tolist() if value else [] for key, value in outputs.items()}

        return {
            "speech": merged["speech"],
            "snr": merged["snr"],
            "c50": merged["c50"],
            "sample_rate": self.sample_rate,
            "chunk_size_seconds": self.model.chunk_size,
        }


class BrouhahaVADGrpcServicer(pb2_grpc.BrouhahaVADGrpcServiceServicer):
    def __init__(self, service: BrouhahaVADService) -> None:
        self.service = service

    def Predict(self, request: pb2.AudioRequest, context: grpc.ServicerContext) -> pb2.BrouhahaResponse:
        result = self.service.predict(request.audio)
        return pb2.BrouhahaResponse(
            filename=request.filename,
            prediction=pb2.BrouhahaPrediction(
                speech=result["speech"],
                snr=result["snr"],
                c50=result["c50"],
                sample_rate=result["sample_rate"],
                chunk_size_seconds=result["chunk_size_seconds"],
            ),
        )


def serve(port: int = 50051) -> None:
    config_path = Path("experiments/brouhaha_vad/predict.yaml")
    service = BrouhahaVADService(config_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_BrouhahaVADGrpcServiceServicer_to_server(
        BrouhahaVADGrpcServicer(service),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Brouhaha VAD gRPC service.")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind the gRPC server.")
    args = parser.parse_args()
    serve(args.port)
