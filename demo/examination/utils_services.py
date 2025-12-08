import argparse
import base64
import io
import json
import os
import sys
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
from dotenv import load_dotenv

load_dotenv()

# Настраиваем пути для импортов
SCRIPT_DIR = Path(__file__).parent.resolve()
VOICEPERSONIFICATION_ROOT = SCRIPT_DIR.parent.parent

# Добавляем корень проекта в sys.path для импорта services.protos
sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT))

# Добавляем пути для protobuf модулей (если указана переменная окружения)
way_to_protos = os.getenv('PROTOS_WAY')
if way_to_protos:
    sys.path.insert(0, way_to_protos)

# Добавляем путь для multi_service_client
sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT / "services" / "examples"))

from multi_service_client import (
    ServiceEndpoint,
    process,
    create_session_factory,
    enroll,
    find,
    call_brouhaha,
    call_whisper,
    call_itmo_large,
    make_audio_request,
    resample,
    apply_vad,
    tensor_to_bytes,
    VAD_THR,
    MetadataEntry,
)


def recognize_audio(audio_path: Path) -> Dict[str, object]:
    payload = make_audio_request(audio_path, tensor_to_bytes(resample(audio_path)))
    vad_result = call_brouhaha(brouhaha_vad_url, payload)
    speech_time = sum(np.array(vad_result["speech"]) > VAD_THR)
    snr = np.array(vad_result["snr"])[np.array(vad_result["speech"]) > VAD_THR].mean()
    c50 = np.array(vad_result["c50"])[np.array(vad_result["speech"]) > VAD_THR].mean()
    payload = make_audio_request(audio_path, apply_vad(audio_path, np.array(vad_result["speech"])))
    sr_result = call_itmo_large(itmo_personification_large_url, payload)
    sr_embedding = np.array(sr_result["values"], dtype=np.float32).tobytes()
    asr_result = call_whisper(whisper_recognition_url, payload)
    text = asr_result["text"]
    language = asr_result["language"]
    return {
        "text": text,
        "language": language,
        "speech_time": speech_time,
        "snr": snr,
        "c50": c50,
        "embedding": sr_embedding,
    }


def verify_speaker(
    audio_path: Path,
    user_name: str,
    database_path: Path,
    host: str = None,
    vad_port: int = None,
    sr_port: int = None,
    verification_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Верификация диктора - проверка, что аудио принадлежит указанному пользователю.
    
    Args:
        audio_path: Путь к аудио файлу для верификации
        user_name: Имя пользователя для проверки (например, "id10289")
        database_path: Путь к базе данных с метаданными
        host: Хост сервера (по умолчанию из GRPC_HOST или nid-vg-01)
        vad_port: Порт VAD сервиса (по умолчанию из VAD_SERVICE_PORT или 50052)
        sr_port: Порт спикерной модели (по умолчанию из PERSONIFICATION_SERVICE_PORT или 50053)
        verification_threshold: Порог верификации (по умолчанию 0.5)
    
    Returns:
        dict: Словарь с результатами верификации:
            - verified: True если верификация успешна, False иначе
            - score: максимальный score совпадения с записями пользователя
            - user_name: имя проверяемого пользователя
            - threshold: использованный порог верификации
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
    
    # Используем значения из .env, если не указаны явно
    if host is None:
        host = os.getenv("GRPC_HOST", "nid-vg-01")
    if vad_port is None:
        vad_port = int(os.getenv("VAD_SERVICE_PORT", os.getenv("VAD_PORT", "50052")))
    if sr_port is None:
        sr_port = int(os.getenv("PERSONIFICATION_SERVICE_PORT", os.getenv("SR_PORT", "50053")))
    
    # Создаем endpoints для сервисов
    vad_endpoint = ServiceEndpoint(host, vad_port)
    sr_endpoint = ServiceEndpoint(host, sr_port)
    
    # Подготавливаем аудио для VAD
    payload = make_audio_request(audio_path, tensor_to_bytes(resample(audio_path)))
    
    # Подключаемся к VAD сервису
    vad_channel = grpc.insecure_channel(vad_endpoint.address())
    vad_result = call_brouhaha(vad_channel, payload)
    
    # Применяем VAD к аудио
    vad_audio = apply_vad(audio_path, np.array(vad_result["speech"]))
    payload = make_audio_request(audio_path, vad_audio)
    
    # Подключаемся к спикерной модели и получаем embedding
    sr_channel = grpc.insecure_channel(sr_endpoint.address())
    sr_result = call_itmo_large(sr_channel, payload)
    sr_embedding = np.array(sr_result["values"], dtype=np.float32).tobytes()
    
    # Закрываем каналы
    vad_channel.close()
    sr_channel.close()
    
    # Подготавливаем результат для поиска в базе данных
    test_result = {
        "embedding": sr_embedding
    }
    
    # Ищем все записи указанного пользователя в базе данных
    session_maker = create_session_factory(database_path)
    with session_maker() as session:
        # Получаем все записи указанного пользователя
        user_entries = session.query(MetadataEntry).filter(
            MetadataEntry.user_name == user_name
        ).order_by(MetadataEntry.created_at.desc()).all()
        
        if not user_entries:
            return {
                "verified": False,
                "score": 0.0,
                "user_name": user_name,
                "threshold": verification_threshold,
                "message": f"Пользователь {user_name} не найден в базе данных"
            }
        
        # Извлекаем embedding'и пользователя
        enroll_embeddings = []
        for entry in user_entries:
            if entry.embedding:
                embedding_array = np.frombuffer(entry.embedding, dtype=np.float32)
                enroll_embeddings.append(embedding_array)
        
        if not enroll_embeddings:
            return {
                "verified": False,
                "score": 0.0,
                "user_name": user_name,
                "threshold": verification_threshold,
                "message": f"У пользователя {user_name} нет валидных embedding'ов в базе данных"
            }
        
        # Сравниваем test embedding с embedding'ами пользователя
        enroll_embeddings = np.stack(enroll_embeddings)
        test_embedding = np.frombuffer(test_result["embedding"], dtype=np.float32)[None, ...]
        
        # Нормализуем embedding'и для косинусного сходства
        en_norm = np.sqrt((enroll_embeddings**2).sum(1))
        te_norm = np.sqrt((test_embedding**2).sum(1))
        
        enroll_w = (enroll_embeddings.T / np.clip(en_norm, 1e-8, None)).T
        test_w = (test_embedding.T / np.clip(te_norm, 1e-8, None)).T
        
        # Вычисляем косинусное сходство
        scores = np.dot(enroll_w, test_w.T)
        max_score = float(scores.max())
        
        # Проверяем верификацию
        verified = max_score >= verification_threshold
    
    return {
        "verified": verified,
        "score": max_score,
        "user_name": user_name,
        "threshold": verification_threshold,
        "message": f"Верификация {'успешна' if verified else 'не пройдена'}: score={max_score:.4f}, threshold={verification_threshold}"
    }


def run_asr(
    audio_path: Path,
    host: str = None,
    vad_port: int = None,
    asr_port: int = None,
) -> Dict[str, object]:
    """
    Запуск ASR (распознавание речи) и получение транскрипции текста.
    
    Args:
        audio_path: Путь к аудио файлу
        host: Хост сервера (по умолчанию из GRPC_HOST или nid-vg-01)
        vad_port: Порт VAD сервиса (по умолчанию из VAD_SERVICE_PORT или 50052)
        asr_port: Порт ASR сервиса (по умолчанию из RECOGNITION_SERVICE_PORT или 50055)
    
    Returns:
        dict: Словарь с результатами распознавания:
            - text: распознанный текст
            - language: язык распознавания
            - segments: список сегментов с временными метками
            - segment_count: количество сегментов
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
    
    # Используем значения из .env, если не указаны явно
    if host is None:
        host = os.getenv("GRPC_HOST", "nid-vg-01")
    if vad_port is None:
        vad_port = int(os.getenv("VAD_SERVICE_PORT", os.getenv("VAD_PORT", "50052")))
    if asr_port is None:
        asr_port = int(os.getenv("RECOGNITION_SERVICE_PORT", os.getenv("ASR_PORT", "50055")))
    
    # Создаем endpoints для сервисов
    vad_endpoint = ServiceEndpoint(host, vad_port)
    asr_endpoint = ServiceEndpoint(host, asr_port)
    
    # Подготавливаем аудио для VAD
    payload = make_audio_request(audio_path, tensor_to_bytes(resample(audio_path)))
    
    # Подключаемся к VAD сервису
    vad_channel = grpc.insecure_channel(vad_endpoint.address())
    vad_result = call_brouhaha(vad_channel, payload)
    
    # Применяем VAD к аудио
    vad_audio = apply_vad(audio_path, np.array(vad_result["speech"]))
    payload = make_audio_request(audio_path, vad_audio)
    
    # Подключаемся к ASR сервису и получаем транскрипцию
    asr_channel = grpc.insecure_channel(asr_endpoint.address())
    asr_result = call_whisper(asr_channel, payload)
    
    # Закрываем каналы
    vad_channel.close()
    asr_channel.close()
    
    return {
        "text": asr_result["text"],
        "language": asr_result["language"],
        "segments": asr_result["segments"],
        "segment_count": asr_result["segment_count"]
    }


def identify_speaker(
    audio_path: Path,
    database_path: Path,
    host: str = None,
    vad_port: int = None,
    sr_port: int = None,
    asr_port: int = None,
) -> Dict[str, object]:
    """
    Идентификация диктора - поиск пользователя по аудио в базе данных.
    
    Args:
        audio_path: Путь к аудио файлу для идентификации
        database_path: Путь к базе данных с метаданными
        host: Хост сервера (по умолчанию из GRPC_HOST или nid-vg-01)
        vad_port: Порт VAD сервиса (по умолчанию из VAD_SERVICE_PORT или 50052)
        sr_port: Порт спикерной модели (по умолчанию из PERSONIFICATION_SERVICE_PORT или 50053)
        asr_port: Порт ASR сервиса (по умолчанию из RECOGNITION_SERVICE_PORT или 50055)
    
    Returns:
        dict: Словарь с результатами идентификации:
            - user_id: найденный user_id
            - user_name: имя пользователя
            - score: оценка совпадения (0-1)
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
    
    # Используем значения из .env, если не указаны явно
    if host is None:
        host = os.getenv("GRPC_HOST", "nid-vg-01")
    if vad_port is None:
        vad_port = int(os.getenv("VAD_SERVICE_PORT", os.getenv("VAD_PORT", "50052")))
    if sr_port is None:
        sr_port = int(os.getenv("PERSONIFICATION_SERVICE_PORT", os.getenv("SR_PORT", "50053")))
    if asr_port is None:
        asr_port = int(os.getenv("RECOGNITION_SERVICE_PORT", os.getenv("ASR_PORT", "50055")))
    
    # Создаем endpoints для сервисов
    vad_endpoint = ServiceEndpoint(host, vad_port)
    sr_endpoint = ServiceEndpoint(host, sr_port)
    
    # Подготавливаем аудио для VAD
    payload = make_audio_request(audio_path, tensor_to_bytes(resample(audio_path)))
    
    # Подключаемся к VAD сервису
    vad_channel = grpc.insecure_channel(vad_endpoint.address())
    vad_result = call_brouhaha(vad_channel, payload)
    
    # Применяем VAD к аудио
    vad_audio = apply_vad(audio_path, np.array(vad_result["speech"]))
    payload = make_audio_request(audio_path, vad_audio)
    
    # Подключаемся к спикерной модели и получаем embedding
    sr_channel = grpc.insecure_channel(sr_endpoint.address())
    sr_result = call_itmo_large(sr_channel, payload)
    sr_embedding = np.array(sr_result["values"], dtype=np.float32).tobytes()
    
    # Закрываем каналы
    vad_channel.close()
    sr_channel.close()
    
    # Подготавливаем результат для поиска в базе данных
    result = {
        "embedding": sr_embedding
    }
    
    # Ищем подходящего пользователя в базе данных
    session_maker = create_session_factory(database_path)
    with session_maker() as session:
        user_name, score = find(session, result)
    
    # Извлекаем user_id из user_name (предполагаем формат "id10289")
    user_id = user_name.replace("id", "") if user_name.startswith("id") else user_name
    
    return {
        "user_id": user_id,
        "user_name": user_name,
        "score": float(score)
    }


def test_speaker_verification():
    """
    Тест верификации диктора:
    1. Регистрирует пользователя используя первый аудио файл
    2. Проверяет ASR (распознавание речи) для обоих аудио файлов
    3. Идентифицирует пользователя используя второй аудио файл
    4. Верифицирует пользователя
    
    Требования:
    - Сервисы должны быть запущены на указанном хосте
    - Параметры подключения берутся из .env файла:
      * GRPC_HOST (по умолчанию nid-vg-01)
      * VAD_SERVICE_PORT (по умолчанию 50052)
      * PERSONIFICATION_SERVICE_PORT (по умолчанию 50053)
      * RECOGNITION_SERVICE_PORT (по умолчанию 50055)
    - Можно переопределить через переменные окружения или параметры функции
    """
    # Пути к аудио файлам
    enroll_audio_path = Path("/mnt/asr_hot/dutov/study/nirsi/temp_audio/015014c5-25bd-4516-9724-57643e860b4d.wav")
    test_audio_path = Path("/mnt/asr_hot/dutov/study/nirsi/temp_audio/0a46094a-90da-4e8e-9ff6-314aa39b7fba.wav")
    
    # Параметры подключения (используем переменные из .env)
    host = os.getenv("GRPC_HOST", "nid-vg-01")
    vad_port = int(os.getenv("VAD_SERVICE_PORT", os.getenv("VAD_PORT", "50052")))
    sr_port = int(os.getenv("PERSONIFICATION_SERVICE_PORT", os.getenv("SR_PORT", "50053")))
    asr_port = int(os.getenv("RECOGNITION_SERVICE_PORT", os.getenv("ASR_PORT", "50055")))
    
    # Путь к базе данных (можно использовать временную для теста)
    database_path = Path("test_verification.db")
    
    # Имя пользователя для регистрации
    user_name = "id10001"
    user_id = "10001"
    
    print("=" * 70)
    print("ТЕСТ ВЕРИФИКАЦИИ ДИКТОРА")
    print("=" * 70)
    
    # Проверяем существование файлов
    if not enroll_audio_path.exists():
        print(f"ОШИБКА: Файл для регистрации не найден: {enroll_audio_path}")
        return
    
    if not test_audio_path.exists():
        print(f"ОШИБКА: Файл для теста не найден: {test_audio_path}")
        return
    
    # Шаг 1: Регистрация пользователя
    print("\n[1] Регистрация пользователя...")
    print(f"    Файл: {enroll_audio_path.name}")
    print(f"    Пользователь: {user_name}")
    print(f"    Хост: {host}")
    print(f"    Порты: VAD={vad_port}, SR={sr_port}, ASR={asr_port}")
    
    try:
        # Создаем endpoints для сервисов
        vad_endpoint = ServiceEndpoint(host, vad_port)
        sr_endpoint = ServiceEndpoint(host, sr_port)
        asr_endpoint = ServiceEndpoint(host, asr_port)
        
        # Проверяем подключение к сервисам
        print("    Проверка подключения к сервисам...")
        try:
            test_channel = grpc.insecure_channel(vad_endpoint.address())
            grpc.channel_ready_future(test_channel).result(timeout=5)
            test_channel.close()
            print("    ✓ Подключение к VAD сервису успешно")
        except Exception as e:
            print(f"    ✗ Ошибка подключения к VAD сервису: {e}")
            print(f"    Попробуйте проверить, запущены ли сервисы на {host}")
            return
        
        # Обрабатываем аудио для регистрации
        print("    Обработка аудио...")
        try:
            result = process(enroll_audio_path, vad_endpoint, sr_endpoint, asr_endpoint)
        except grpc.RpcError as e:
            print(f"    ✗ gRPC ошибка: {e.code()} - {e.details()}")
            print(f"\n    Диагностика:")
            print(f"    1. Проверьте, запущены ли сервисы на {host}")
            print(f"    2. Проверьте порты: VAD={vad_port}, SR={sr_port}, ASR={asr_port}")
            print(f"    3. Возможно, сервисы используют другой интерфейс или версию protobuf")
            print(f"    4. Попробуйте использовать другой хост через переменную окружения:")
            print(f"       GRPC_HOST=localhost python demo/examination/utils2.py")
            print(f"\n    Примечание: Тест требует запущенных gRPC сервисов.")
            print(f"    Если сервисы не запущены, тест не может быть выполнен.")
            raise
        
        # Добавляем информацию о пользователе
        result["user_id"] = user_id
        result["user_name"] = user_name
        result["filename"] = enroll_audio_path.name
        
        # Сохраняем в базу данных
        session_maker = create_session_factory(database_path)
        with session_maker() as session:
            enroll(session, result)
        
        print("    ✓ Пользователь успешно зарегистрирован")
        print(f"    Текст: {result.get('text', '')[:50]}...")
        print(f"    Язык: {result.get('language', '')}")
        
    except Exception as e:
        print(f"    ✗ Ошибка при регистрации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Шаг 2: Проверка ASR для обоих аудио файлов
    print("\n[2] Проверка ASR (распознавание речи)...")
    
    # ASR для первого файла (регистрация)
    print(f"\n    [2.1] ASR для файла регистрации: {enroll_audio_path.name}")
    try:
        asr_result_enroll = run_asr(
            enroll_audio_path,
            host=host,
            vad_port=vad_port,
            asr_port=asr_port
        )
        print(f"    ✓ ASR выполнен успешно")
        print(f"    Текст: {asr_result_enroll.get('text', '')[:100]}...")
        print(f"    Язык: {asr_result_enroll.get('language', '')}")
        print(f"    Сегментов: {asr_result_enroll.get('segment_count', 0)}")
    except Exception as e:
        print(f"    ✗ Ошибка при выполнении ASR: {e}")
        import traceback
        traceback.print_exc()
    
    # ASR для второго файла (тест)
    print(f"\n    [2.2] ASR для файла теста: {test_audio_path.name}")
    try:
        asr_result_test = run_asr(
            test_audio_path,
            host=host,
            vad_port=vad_port,
            asr_port=asr_port
        )
        print(f"    ✓ ASR выполнен успешно")
        print(f"    Текст: {asr_result_test.get('text', '')[:100]}...")
        print(f"    Язык: {asr_result_test.get('language', '')}")
        print(f"    Сегментов: {asr_result_test.get('segment_count', 0)}")
    except Exception as e:
        print(f"    ✗ Ошибка при выполнении ASR: {e}")
        import traceback
        traceback.print_exc()
    
    # Шаг 3: Идентификация пользователя
    print("\n[3] Идентификация пользователя...")
    print(f"    Файл: {test_audio_path.name}")
    
    try:
        ident_result = identify_speaker(
            test_audio_path,
            database_path,
            host=host,
            vad_port=vad_port,
            sr_port=sr_port,
            asr_port=asr_port
        )
        
        found_user_name = ident_result.get("user_name", "")
        score = ident_result.get("score", 0.0)
        
        print(f"    ✓ Пользователь найден: {found_user_name}")
        print(f"    Score: {score:.4f}")
        
        # Проверяем, что найден правильный пользователь
        if found_user_name == user_name:
            print(f"    ✓ Найден правильный пользователь!")
        else:
            print(f"    ✗ Найден другой пользователь! Ожидался {user_name}, найден {found_user_name}")
        
        # Шаг 4: Верификация пользователя
        print("\n[4] Верификация пользователя...")
        print(f"    Проверка: принадлежит ли аудио пользователю {user_name}?")
        
        verify_result = verify_speaker(
            test_audio_path,
            user_name,
            database_path,
            host=host,
            vad_port=vad_port,
            sr_port=sr_port,
            verification_threshold=0.5
        )
        
        verified = verify_result.get("verified", False)
        verify_score = verify_result.get("score", 0.0)
        message = verify_result.get("message", "")
        
        print(f"    {message}")
        
        if verified:
            print(f"    ✓ Верификация успешна!")
        else:
            print(f"    ✗ Верификация не пройдена")
        
    except Exception as e:
        print(f"    ✗ Ошибка при идентификации/верификации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    test_speaker_verification()
