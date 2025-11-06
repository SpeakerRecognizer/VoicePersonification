import argparse
import json
from pathlib import Path

import grpc

from services.protos import voice_personification_pb2 as pb2
from services.protos import voice_personification_pb2_grpc as pb2_grpc


def read_audio_bytes(path: Path) -> bytes:
    with path.open("rb") as file:
        return file.read()

def run_embedding(channel: grpc.Channel, payload: pb2.AudioRequest, stub_cls):
    stub = stub_cls(channel)
    response = stub.Embed(payload)
    values = list(response.embedding.values)
    return {
        "filename": response.filename,
        "dimension": response.embedding.dimension,
        "embedding": values 
    }


def main():
    parser = argparse.ArgumentParser(description="Test gRPC services with an audio file.")
    parser.add_argument("audio", type=Path, help="Path to audio file (wav/flac).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052, help="Service port override.")
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    port = args.port

    payload = pb2.AudioRequest(
        filename=str(args.audio.name),
        audio=read_audio_bytes(args.audio),
    )

    address = f"{args.host}:{port}"
    with grpc.insecure_channel(address) as channel:
        result = run_embedding(channel, payload, pb2_grpc.ECAPATDNNGrpcServiceStub)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
