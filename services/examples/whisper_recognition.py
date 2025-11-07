import argparse
import json
from pathlib import Path

import grpc

from services.protos import voice_personification_pb2 as pb2
from services.protos import voice_personification_pb2_grpc as pb2_grpc


def read_audio_bytes(path: Path) -> bytes:
    with path.open("rb") as file:
        return file.read()


def run_transcription(channel: grpc.Channel, payload: pb2.AudioRequest) -> dict:
    stub = pb2_grpc.WhisperRecognitionGrpcServiceStub(channel)
    response = stub.Transcribe(payload)
    segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in response.transcription.segments
    ]
    print(segments)
    return {
        "filename": response.filename,
        "text": response.transcription.text,
        "language": response.transcription.language,
        "segment_count": len(response.transcription.segments),
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Whisper gRPC service with an audio file.")
    parser.add_argument("audio", type=Path, help="Path to audio file (wav/flac).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50055, help="Service port override.")
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    payload = pb2.AudioRequest(
        filename=args.audio.name,
        audio=read_audio_bytes(args.audio),
    )

    address = f"{args.host}:{args.port}"

    with grpc.insecure_channel(address) as channel:
        result = run_transcription(channel, payload)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
