import argparse
from concurrent import futures
from pathlib import Path
from typing import Dict, List

import grpc
import torch

from .common import HydraAudioService
from .protos import voice_personification_pb2 as pb2
from .protos import voice_personification_pb2_grpc as pb2_grpc


class ItmoPersonificationLargeService(HydraAudioService):
    """Serve embeddings from the ITMO personification large model."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path)

    def predict(self, payload: bytes) -> Dict[str, List[float]]:
        inputs = self.preprocess(payload)

        with torch.no_grad():
            embedding = self.model(inputs).squeeze()

        embedding = embedding.cpu()
        return {
            "embedding": embedding.tolist(),
            "dimension": embedding.numel(),
        }


class ItmoPersonificationLargeGrpcServicer(pb2_grpc.ItmoPersonificationLargeGrpcServiceServicer):
    def __init__(self, service: ItmoPersonificationLargeService) -> None:
        self.service = service

    def Embed(self, request: pb2.AudioRequest, context: grpc.ServicerContext) -> pb2.EmbeddingResponse:
        result = self.service.predict(request.audio)
        return pb2.EmbeddingResponse(
            filename=request.filename,
            embedding=pb2.EmbeddingPrediction(
                values=result["embedding"],
                dimension=result["dimension"],
            ),
        )


def serve(port: int = 50053) -> None:
    config_path = Path("experiments/itmo-personification-model-large/test.yaml")
    service = ItmoPersonificationLargeService(config_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_ItmoPersonificationLargeGrpcServiceServicer_to_server(
        ItmoPersonificationLargeGrpcServicer(service),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ITMO personification large gRPC service.")
    parser.add_argument("--port", type=int, default=50053, help="Port to bind the gRPC server.")
    args = parser.parse_args()
    serve(args.port)
