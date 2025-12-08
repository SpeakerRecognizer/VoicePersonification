import argparse
from concurrent import futures
from pathlib import Path
from typing import Dict, List

import grpc
import torch

from .common import HydraAudioService
from .protos import voice_personification_pb2 as pb2
from .protos import voice_personification_pb2_grpc as pb2_grpc


class ItmoPersonificationSegmentationService(HydraAudioService):
    """Serve speaker segmentation embeddings produced by the ITMO model."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path)

    def predict(self, payload: bytes) -> Dict[str, List[float]]:
        waveform = self._load_waveform(payload)
        batch = waveform.unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(batch).squeeze().cpu()

        with torch.no_grad():
            spk_embs, timestamps = self.model.model.segemntate(
                waveform.cpu(),
                sample_rate=self.sample_rate,
            )

        segments = []
        if spk_embs.numel() > 0:
            for emb, ts in zip(spk_embs, timestamps):
                start = ts[0].item() if isinstance(ts, torch.Tensor) else ts[0]
                end = ts[1].item() if isinstance(ts, torch.Tensor) else ts[1]
                segments.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "embedding": emb.cpu().tolist(),
                    }
                )

        return {
            "mean_embedding": embedding.tolist(),
            "dimension": embedding.numel(),
            "segments": segments,
        }


class ItmoPersonificationSegmentationGrpcServicer(pb2_grpc.ItmoPersonificationSegmentationGrpcServiceServicer):
    def __init__(self, service: ItmoPersonificationSegmentationService) -> None:
        self.service = service

    def Segment(self, request: pb2.AudioRequest, context: grpc.ServicerContext) -> pb2.SegmentationResponse:
        result = self.service.predict(request.audio)

        return pb2.SegmentationResponse(
            filename=request.filename,
            result=pb2.SegmentationPrediction(
                mean_embedding=pb2.EmbeddingPrediction(
                    values=result["mean_embedding"],
                    dimension=result["dimension"],
                ),
                segments=[
                    pb2.Segment(
                        start=segment["start"],
                        end=segment["end"],
                        embedding=segment["embedding"],
                    )
                    for segment in result["segments"]
                ],
            ),
        )


def serve(port: int = 50054) -> None:
    config_path = Path("experiments/itmo-personification-model-segmentation/test.yaml")
    service = ItmoPersonificationSegmentationService(config_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_ItmoPersonificationSegmentationGrpcServiceServicer_to_server(
        ItmoPersonificationSegmentationGrpcServicer(service),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ITMO personification segmentation gRPC service.")
    parser.add_argument("--port", type=int, default=50054, help="Port to bind the gRPC server.")
    args = parser.parse_args()
    serve(args.port)
