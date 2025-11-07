import argparse
import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import grpc
import numpy as np
import torch
import torchaudio
from sqlalchemy import BLOB, Column, DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from services.protos import voice_personification_pb2 as pb2
from services.protos import voice_personification_pb2_grpc as pb2_grpc


VAD_THR = 0.5
REQ_SR = 16_000

Base = declarative_base()


class MetadataEntry(Base):
    """ORM model representing metadata captured for each audio request."""

    __tablename__ = "metadata_entries"

    id = Column(Integer, primary_key=True)
    file_path = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    embedding = Column(BLOB, nullable=False)
    text = Column(String, nullable=False)
    speech_time = Column(Float, nullable=False)
    snr = Column(Float, nullable=False)
    c50 = Column(Float, nullable=False)
    language = Column(String, nullable=False)


@dataclass(frozen=True)
class ServiceEndpoint:
    """Connection details for a single gRPC service."""

    host: str
    port: int

    def address(self) -> str:
        return f"{self.host}:{self.port}"


def make_audio_request(audio_path: Path, audio:bytes=None) -> pb2.AudioRequest:
    return pb2.AudioRequest(
        filename=audio_path.name,
        audio=audio,
    )

def call_brouhaha(channel: grpc.Channel, payload: pb2.AudioRequest) -> Dict[str, object]:
    stub = pb2_grpc.BrouhahaVADGrpcServiceStub(channel)
    response = stub.Predict(payload)
    prediction = response.prediction
    return {
        "filename": response.filename,
        "speech": list(prediction.speech),
        "snr": list(prediction.snr),
        "c50": list(prediction.c50),
    }


def call_itmo_large(channel: grpc.Channel, payload: pb2.AudioRequest) -> Dict[str, object]:
    stub = pb2_grpc.ItmoPersonificationLargeGrpcServiceStub(channel)
    response = stub.Embed(payload)
    embedding = response.embedding
    return {
        "filename": response.filename,
        "dimension": embedding.dimension,
        "values": list(embedding.values),
    }


def call_whisper(channel: grpc.Channel, payload: pb2.AudioRequest) -> Dict[str, object]:
    stub = pb2_grpc.WhisperRecognitionGrpcServiceStub(channel)
    response = stub.Transcribe(payload)
    transcription = response.transcription
    segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in transcription.segments
    ]
    return {
        "filename": response.filename,
        "text": transcription.text,
        "language": transcription.language,
        "segment_count": len(segments),
        "segments": segments,
    }

def create_session_factory(db_path: Path):
    db_uri = f"sqlite:///{db_path.resolve()}"
    engine = create_engine(db_uri, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torchaudio.save(
        buffer,
        tensor.detach().cpu(),
        REQ_SR,
        format="wav",
        encoding="PCM_S", 
        bits_per_sample=16,
    )
    return buffer.getvalue()

def resample(audio_path:Path) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path, normalize=False)
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=REQ_SR)(
            wav
        )
    return wav

def apply_vad(audio_path:Path, vad_markup: List[float]) -> bytes:
    wav = resample(audio_path)
    
    vad_markup = torch.nn.functional.interpolate(
                    torch.from_numpy(vad_markup).float().unsqueeze(0).unsqueeze(0),
                    size=wav.shape[-1],
                ).squeeze().int()
    wav = wav[:, vad_markup > VAD_THR]
    return tensor_to_bytes(wav)

def enroll(session, result):
    try:
        entry = MetadataEntry(
            file_path=result.get("filename", ""), 
            user_id=result.get("user_id", ""), 
            user_name=result.get("user_name", ""), 
            created_at=result.get("created_at", ""), 
            embedding=result.get("embedding", ""), 
            text=result.get("text", ""), 
            speech_time=result.get("speech_time", ""), 
            snr=result.get("snr", ""), 
            c50=result.get("c50", ""), 
            language=result.get("language", ""), 
        )
        session.add(entry)
        session.commit()
    except:
        ...
        

def find(session, result) -> List[Dict[str, object]]:
    entries = session.query(MetadataEntry).order_by(MetadataEntry.created_at.desc()).all()
    enroll_embeddings, enroll_names = [], []
    for entry in entries:
        if entry.embedding:
            
            embedding_b64 = np.frombuffer(entry.embedding, dtype=np.float32)
            enroll_embeddings.append(embedding_b64)
            enroll_names.append(entry.user_name)

    enroll_embeddings = np.stack(enroll_embeddings)
    test_embedding = np.frombuffer(result["embedding"], dtype=np.float32)[None, ...]
    
    en_norm = np.sqrt((enroll_embeddings**2).sum(1))
    te_norm = np.sqrt((test_embedding**2).sum(1))

    enroll_w = (enroll_embeddings.T / np.clip(en_norm, 1e-8, None)).T
    test_w = (test_embedding.T / np.clip(te_norm, 1e-8, None)).T
    scores = np.dot(enroll_w, test_w.T)
    
    id = np.argmax(scores)
    spk = enroll_names[id]
    return spk,  scores.max()
        
        
def process(audio_path, 
            brouhaha_vad_url, 
            itmo_personification_large_url, 
            whisper_recognition_url):
    payload = make_audio_request(audio_path, tensor_to_bytes(resample(audio_path)))
    
    brouhaha_vad_url = grpc.insecure_channel(brouhaha_vad_url.address())
    itmo_personification_large_url = grpc.insecure_channel(itmo_personification_large_url.address())
    whisper_recognition_url = grpc.insecure_channel(whisper_recognition_url.address())
    
    vad_result = call_brouhaha(brouhaha_vad_url, payload)
    speech_time = sum(np.array(vad_result["speech"]) > VAD_THR)
    snr = np.array(vad_result["snr"])[np.array(vad_result["speech"]) > VAD_THR].mean()
    c50 = np.array(vad_result["c50"])[np.array(vad_result["speech"]) > VAD_THR].mean()
    
    payload = make_audio_request(audio_path, apply_vad(audio_path, 
                                                       np.array(vad_result["speech"])))
    
    sr_result = call_itmo_large(itmo_personification_large_url, payload)
    sr_embedding = np.array(sr_result["values"], dtype=np.float32).tobytes()
    
    asr_result = call_whisper(whisper_recognition_url, payload)
    
    text = asr_result["text"]
    language = asr_result["language"]
    
    return {
        "file_path" : audio_path,
        "created_at" : datetime.now(timezone.utc),
        "embedding" : sr_embedding,
        "text" : text,
        "speech_time" : speech_time,
        "snr" : snr,
        "c50" : c50,
        "language" : language
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Client that queries multiple gRPC audio services and aggregates metadata."
    )
    parser.add_argument("audio", type=Path, nargs="?", help="Path to the audio file to process.")
    parser.add_argument("--host", default="127.0.0.1", help="Common host for all services.")
    parser.add_argument("--vad-port", type=int, default=50051, help="Brouhaha VAD service port.")
    parser.add_argument("--sr-port", type=int, default=50053, help="ITMO personification large service port.")
    parser.add_argument("--asr-port", type=int, default=50055, help="Whisper recognition service port.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("multi_service_metadata.db"),
        help="Path to SQLite database file where metadata will be stored.",
    )
    args = parser.parse_args()
    
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    result = process(args.audio,
                     ServiceEndpoint(args.host, args.vad_port),
                     ServiceEndpoint(args.host, args.sr_port),
                     ServiceEndpoint(args.host, args.asr_port),)
    
    result["user_id"] = "10289"
    result["user_name"] = "id10289"
    
    session_maker = create_session_factory(args.database)
    with session_maker() as session:
        enroll(session, result)
        user_id, score = find(session, result)

    print("Congratulations")
    print(f"I will find spk {user_id} with score {score}")
    print(f"She or he say: {result['text']}")


if __name__ == "__main__":
    main()
