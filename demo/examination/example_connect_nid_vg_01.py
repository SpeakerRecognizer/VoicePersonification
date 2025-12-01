#!/usr/bin/env python3
"""
Пример подключения к сервисам на машине nid-vg-01
"""

import sys
from pathlib import Path

# Настраиваем пути для импортов
SCRIPT_DIR = Path(__file__).parent.resolve()
VOICEPERSONIFICATION_ROOT = SCRIPT_DIR / "VoicePersonification"
sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT))
sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT / "services" / "protos"))
sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT / "services" / "examples"))

import grpc
import numpy as np

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
    VAD_THR
)

# Настройки подключения к сервисам на nid-vg-01
HOST = "nid-vg-01"
VAD_PORT = 50051      # VAD сервис
SR_PORT = 50053       # Спикерная модель
ASR_PORT = 50055      # ASR сервис

# Путь к аудио файлу (измените на свой)
AUDIO_FILE = Path("/mnt/asr_hot/dutov/study/nirsi/temp_audio/2e7a1afa-ae79-4b03-af7b-e61cd62bbc86.wav")

# Путь к базе данных
DATABASE_PATH = Path("multi_service_metadata.db")


def run_asr(audio_path: Path, host: str = HOST, vad_port: int = VAD_PORT, asr_port: int = ASR_PORT) -> dict:
    """
    Запуск ASR (распознавание речи) и получение транскрипции текста.
    
    Args:
        audio_path: Путь к аудио файлу
        host: Хост сервера (по умолчанию nid-vg-01)
        vad_port: Порт VAD сервиса
        asr_port: Порт ASR сервиса
    
    Returns:
        dict: Словарь с результатами распознавания:
            - text: распознанный текст
            - language: язык распознавания
            - segments: список сегментов с временными метками
    """
    # Проверяем существование аудио файла
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
    
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


def run_identification(audio_path: Path, database_path: Path = DATABASE_PATH, 
                      host: str = HOST, vad_port: int = VAD_PORT, sr_port: int = SR_PORT) -> dict:
    """
    Запуск идентификации спикера и получение подходящего user_id.
    
    Args:
        audio_path: Путь к аудио файлу
        database_path: Путь к базе данных с метаданными
        host: Хост сервера (по умолчанию nid-vg-01)
        vad_port: Порт VAD сервиса
        sr_port: Порт спикерной модели
    
    Returns:
        dict: Словарь с результатами идентификации:
            - user_id: найденный user_id
            - user_name: имя пользователя
            - score: оценка совпадения (0-1)
    """
    # Проверяем существование аудио файла
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
    
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


def main():
    """Основная функция для обработки аудио через сервисы на nid-vg-01"""
    
    # Проверяем существование аудио файла
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {AUDIO_FILE}")
    
    print(f"Подключение к сервисам на {HOST}...")
    print(f"  VAD: {HOST}:{VAD_PORT}")
    print(f"  Спикерная модель: {HOST}:{SR_PORT}")
    print(f"  ASR: {HOST}:{ASR_PORT}")
    print()
    
    # Создаем endpoints для сервисов
    vad_endpoint = ServiceEndpoint(HOST, VAD_PORT)
    sr_endpoint = ServiceEndpoint(HOST, SR_PORT)
    asr_endpoint = ServiceEndpoint(HOST, ASR_PORT)
    
    # Обрабатываем аудио
    print(f"Обработка аудио файла: {AUDIO_FILE}")
    result = process(
        AUDIO_FILE,
        vad_endpoint,
        sr_endpoint,
        asr_endpoint
    )
    
    # Добавляем информацию о пользователе
    result["user_id"] = "10289"
    result["user_name"] = "id10289"
    
    # Сохраняем в базу данных
    print("Сохранение результатов в базу данных...")
    session_maker = create_session_factory(DATABASE_PATH)
    with session_maker() as session:
        enroll(session, result)
        user_id, score = find(session, result)
    
    # Выводим результаты
    print("\n" + "="*50)
    print("Результаты обработки:")
    print("="*50)
    print(f"Найден спикер: {user_id}")
    print(f"Score: {score:.4f}")
    print(f"Текст: {result['text']}")
    print(f"Язык: {result['language']}")
    print(f"Время речи: {result['speech_time']}")
    print(f"SNR: {result['snr']:.2f}")
    print(f"C50: {result['c50']:.2f}")


def test_functions():
    """Тестирование функций run_asr и run_identification на двух аудио файлах"""
    
    audio_files = [
        Path("/mnt/asr_hot/dutov/study/nirsi/temp_audio/2e7a1afa-ae79-4b03-af7b-e61cd62bbc86.wav"),
        Path("/mnt/asr_hot/dutov/study/nirsi/temp_audio/d9f7acc5-85a9-4d6b-b8b3-d3b7ac682c24.wav")
    ]
    
    for audio_file in audio_files:
        print("\n" + "="*70)
        print(f"Тестирование файла: {audio_file.name}")
        print("="*70)
        
        if not audio_file.exists():
            print(f"⚠️  Файл не найден: {audio_file}")
            continue
        
        # Тест ASR
        print("\n[1] Запуск ASR (распознавание речи)...")
        try:
            asr_result = run_asr(audio_file)
            print("✅ ASR выполнен успешно")
            print(f"   Текст: {asr_result['text']}")
            print(f"   Язык: {asr_result['language']}")
            print(f"   Сегментов: {asr_result['segment_count']}")
        except Exception as e:
            print(f"❌ Ошибка при выполнении ASR: {e}")
            continue
        
        # Тест идентификации
        print("\n[2] Запуск идентификации спикера...")
        try:
            ident_result = run_identification(audio_file)
            print("✅ Идентификация выполнена успешно")
            print(f"   User ID: {ident_result['user_id']}")
            print(f"   User Name: {ident_result['user_name']}")
            print(f"   Score: {ident_result['score']:.4f}")
        except Exception as e:
            print(f"❌ Ошибка при выполнении идентификации: {e}")
        
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_functions()
    else:
        main()

