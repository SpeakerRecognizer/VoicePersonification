import copy
import json
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from utils_services import (
    identify_speaker,
    verify_speaker,
    run_asr,
    process,
    create_session_factory,
    enroll,
    ServiceEndpoint,
)
from utils_examinator import (
    create_question_bank,
    check_answer,
    analyze_exam_results,
    get_recommendations,
    get_topics,
)

load_dotenv()

# Константы
SCRIPT_DIR = Path(__file__).parent.resolve()
QUESTION_BANK_FILE = SCRIPT_DIR / "question_bank.json"
EXAMS_DATABASE_FILE = SCRIPT_DIR / "exams_database.json"
USERS_DATABASE_FILE = SCRIPT_DIR / "users_database.json"
DATABASE_PATH = SCRIPT_DIR / "multi_service_metadata.db"

IDENTIFICATION_THRESHOLD = 0.7
VERIFICATION_THRESHOLD = 0.5
EXAM_QUESTIONS_COUNT = 3


# Вспомогательные функции для работы с JSON
def save_question_bank(question_bank: List[Tuple[str, str]]) -> None:
    """Сохраняет банк вопросов в JSON файл."""
    topics = get_topics()
    questions_data = []
    for i, (question, answer) in enumerate(question_bank):
        topic = topics[i] if i < len(topics) else "неизвестно"
        questions_data.append({
            "question": question,
            "answer": answer,
            "topic": topic
        })
    
    data = {
        "questions": questions_data,
        "created_at": datetime.now().isoformat()
    }
    
    with open(QUESTION_BANK_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_question_bank() -> Optional[List[Tuple[str, str]]]:
    """Загружает банк вопросов из JSON файла."""
    if not QUESTION_BANK_FILE.exists():
        return None
    
    try:
        with open(QUESTION_BANK_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = []
            for item in data.get("questions", []):
                questions.append((item["question"], item["answer"]))
            return questions
    except Exception:
        return None


def generate_exam_id() -> str:
    """Генерирует уникальный ID для экзамена."""
    return str(uuid.uuid4())


def save_exam_result(user_name: str, exam_data: dict) -> None:
    """Сохраняет результат экзамена в JSON файл."""
    if not EXAMS_DATABASE_FILE.exists():
        exams_data = {}
    else:
        try:
            with open(EXAMS_DATABASE_FILE, 'r', encoding='utf-8') as f:
                exams_data = json.load(f)
        except Exception:
            exams_data = {}
    
    if user_name not in exams_data:
        exams_data[user_name] = {"exams": []}
    
    exams_data[user_name]["exams"].append(exam_data)
    
    with open(EXAMS_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(exams_data, f, ensure_ascii=False, indent=2)


def load_user_exams(user_name: str) -> dict:
    """Загружает экзамены пользователя из JSON файла."""
    if not EXAMS_DATABASE_FILE.exists():
        return {}
    
    try:
        with open(EXAMS_DATABASE_FILE, 'r', encoding='utf-8') as f:
            exams_data = json.load(f)
            return exams_data
    except Exception:
        return {}


def calculate_grade(passed_questions: int) -> int:
    """Вычисляет оценку на основе количества принятых вопросов."""
    if passed_questions == 3:
        return 5
    elif passed_questions == 2:
        return 4
    elif passed_questions == 1:
        return 3
    else:
        return 2


def convert_audio_to_float32(audio_array: np.ndarray) -> np.ndarray:
    """Конвертирует аудио массив в float32 формат."""
    if audio_array.dtype == np.float32:
        return audio_array
    elif audio_array.dtype == np.int16:
        return audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.int32:
        return audio_array.astype(np.float32) / 2147483648.0
    else:
        return audio_array.astype(np.float32)


# Функции для страницы регистрации
def process_audio_for_registration(audio_data):
    """Обрабатывает аудио для регистрации/идентификации."""
    if audio_data is None:
        return (
            None,
            gr.update(visible=False),  # confirm_block
            "",  # confirm_user_name
            gr.update(visible=False),  # register_block
            "Пожалуйста, запишите аудио перед обработкой."
        )
    
    try:
        # Сохраняем аудио во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = Path(tmp_file.name)
            # Gradio возвращает кортеж (sample_rate, audio_data)
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                audio_array = convert_audio_to_float32(audio_array)
                sf.write(tmp_path, audio_array, sample_rate)
            else:
                # Если это путь к файлу
                shutil.copy(audio_data, tmp_path)
        
        # Идентифицируем пользователя
        ident_result = identify_speaker(
            tmp_path,
            DATABASE_PATH
        )
        
        score = ident_result.get("score", 0.0)
        found_user_name = ident_result.get("user_name", "")
        
        # Очищаем временный файл
        try:
            tmp_path.unlink()
        except:
            pass
        
        # Проверяем вероятность идентификации (должна быть >= 50%)
        if found_user_name and found_user_name != "unknown" and score >= 0.5:
            # Пользователь найден с достаточной вероятностью - показываем блок подтверждения
            return (
                found_user_name,
                gr.update(visible=True),  # confirm_block
                found_user_name,  # confirm_user_name
                gr.update(visible=False),  # register_block
                f"Найден пользователь: {found_user_name} (вероятность: {score:.2%}). Подтвердите, что это вы."
            )
        else:
            # Пользователь не найден или вероятность слишком низкая - показываем блок регистрации
            if found_user_name == "unknown" or score == 0.0:
                message = "Пользователь не найден в базе данных. Пожалуйста, введите имя и фамилию для регистрации."
            else:
                message = f"Вероятность идентификации слишком низкая ({score:.2%} < 50%). Пожалуйста, введите имя и фамилию для регистрации."
            
            return (
                None,
                gr.update(visible=False),  # confirm_block
                "",  # confirm_user_name
                gr.update(visible=True),  # register_block
                message
            )
    except Exception as e:
        return (
            None,
            gr.update(visible=False),  # confirm_block
            "",  # confirm_user_name
            gr.update(visible=False),  # register_block
            f"Ошибка при обработке аудио: {str(e)}"
        )


def decline_confirmation():
    """Отказывается от подтверждения и показывает форму регистрации."""
    return (
        gr.update(visible=False),  # confirm_block
        "",  # confirm_user_name
        gr.update(visible=True),  # register_block
        "Вы отказались от подтверждения. Пожалуйста, введите имя и фамилию для регистрации."
    )


def confirm_user(audio_data, user_name):
    """Подтверждает пользователя и сохраняет его данные."""
    if audio_data is None or not user_name:
        return "Пожалуйста, запишите аудио и укажите имя пользователя."
    
    try:
        # Сохраняем аудио во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = Path(tmp_file.name)
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                audio_array = convert_audio_to_float32(audio_array)
                sf.write(tmp_path, audio_array, sample_rate)
            else:
                shutil.copy(audio_data, tmp_path)
        
        # Верифицируем пользователя
        verify_result = verify_speaker(
            tmp_path,
            user_name,
            DATABASE_PATH,
            verification_threshold=VERIFICATION_THRESHOLD
        )
        
        verified = verify_result.get("verified", False)
        score = verify_result.get("score", 0.0)
        
        # Очищаем временный файл
        try:
            tmp_path.unlink()
        except:
            pass
        
        if verified:
            return f"Пользователь {user_name} успешно подтвержден! Вы можете перейти к экзамену.", user_name
        else:
            return f"Верификация не пройдена (score: {score:.2%}). Попробуйте еще раз.", None
    except Exception as e:
        return f"Ошибка при подтверждении: {str(e)}", None


def register_user(audio_data, first_name, last_name):
    """Регистрирует нового пользователя."""
    if audio_data is None:
        return "Пожалуйста, запишите аудио перед регистрацией.", None
    
    if not first_name or not first_name.strip():
        return "Пожалуйста, введите имя.", None
    
    if not last_name or not last_name.strip():
        return "Пожалуйста, введите фамилию.", None
    
    # Формируем user_name из имени и фамилии
    user_name = f"id{first_name.strip()}_{last_name.strip()}".lower().replace(" ", "_")
    
    try:
        # Сохраняем аудио во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = Path(tmp_file.name)
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                audio_array = convert_audio_to_float32(audio_array)
                sf.write(tmp_path, audio_array, sample_rate)
            else:
                shutil.copy(audio_data, tmp_path)
        
        # Получаем параметры подключения
        host = os.getenv("GRPC_HOST", "nid-vg-01")
        vad_port = int(os.getenv("VAD_SERVICE_PORT", os.getenv("VAD_PORT", "50052")))
        sr_port = int(os.getenv("PERSONIFICATION_SERVICE_PORT", os.getenv("SR_PORT", "50053")))
        asr_port = int(os.getenv("RECOGNITION_SERVICE_PORT", os.getenv("ASR_PORT", "50055")))
        
        # Создаем endpoints
        vad_endpoint = ServiceEndpoint(host, vad_port)
        sr_endpoint = ServiceEndpoint(host, sr_port)
        asr_endpoint = ServiceEndpoint(host, asr_port)
        
        # Обрабатываем аудио
        result = process(tmp_path, vad_endpoint, sr_endpoint, asr_endpoint)
        
        # Извлекаем user_id из user_name
        user_id = user_name.replace("id", "") if user_name.startswith("id") else user_name
        
        # Добавляем информацию о пользователе
        result["user_id"] = user_id
        result["user_name"] = user_name
        result["filename"] = tmp_path.name
        
        # Сохраняем в базу данных
        session_maker = create_session_factory(DATABASE_PATH)
        with session_maker() as session:
            enroll(session, result)
        
        # Очищаем временный файл
        try:
            tmp_path.unlink()
        except:
            pass
        
        return f"Пользователь {first_name.strip()} {last_name.strip()} ({user_name}) успешно зарегистрирован! Вы можете перейти к экзамену.", user_name
    except Exception as e:
        return f"Ошибка при регистрации: {str(e)}", None


# Функции для страницы экзамена
def get_questions_progress(current_exam):
    """Возвращает красивую строку с прогрессом вопросов."""
    if current_exam is None:
        return ""
    
    questions = current_exam.get("questions", [])
    if not questions:
        return ""
    
    progress_parts = []
    for i, q in enumerate(questions):
        if q.get("best_result", False):
            progress_parts.append("✓")
        elif q.get("explanation_viewed", False):
            progress_parts.append("✗")
        elif len(q.get("attempts", [])) > 0:
            # Есть попытки, но нет принятого ответа
            progress_parts.append("○")
        else:
            # Нет попыток
            progress_parts.append("○")
    
    progress_text = " ".join(progress_parts)
    return f"**Прогресс:** {progress_text}"


def generate_exam(current_user, question_bank_state):
    """Генерирует новый экзамен для пользователя."""
    if not current_user:
        return None, None, None, None, None, None, None, None, "Пожалуйста, сначала зарегистрируйтесь или войдите в систему."
    
    try:
        # Загружаем или создаем банк вопросов
        bank = load_question_bank()
        if bank is None:
            bank = create_question_bank()
            save_question_bank(bank)
        
        if len(bank) < EXAM_QUESTIONS_COUNT:
            return None, None, None, None, None, None, None, None, f"Недостаточно вопросов в банке. Требуется минимум {EXAM_QUESTIONS_COUNT}, доступно {len(bank)}."
        
        # Выбираем случайные вопросы
        import random
        selected_questions = random.sample(bank, EXAM_QUESTIONS_COUNT)
        
        # Создаем структуру экзамена
        exam_id = generate_exam_id()
        exam_data = {
            "exam_id": exam_id,
            "date": datetime.now().isoformat(),
            "questions": [
                {
                    "question": q,
                    "reference_answer": a,
                    "attempts": [],
                    "best_result": False,
                    "explanation_viewed": False
                }
                for q, a in selected_questions
            ],
            "grade": 0,
            "passed_questions": 0
        }
        
        # Обновляем состояние
        question_bank_state = bank
        
        # Возвращаем первый вопрос
        first_question = exam_data["questions"][0]
        question_text = f"Вопрос 1 из {EXAM_QUESTIONS_COUNT}:\n\n{first_question['question']}"
        
        progress_text = get_questions_progress(exam_data)
        
        return (
            exam_data,
            0,
            question_text,
            gr.update(visible=True),  # answer_audio
            gr.update(visible=True),  # recognize_btn
            gr.update(visible=True),  # recognized_text
            gr.update(visible=True),  # submit_btn
            gr.update(visible=True),  # view_explanation_btn
            progress_text  # questions_progress
        )
    except Exception as e:
        return None, None, None, None, None, None, None, f"Ошибка при генерации экзамена: {str(e)}"


def recognize_answer_audio(audio_data, current_exam, current_question_index):
    """Распознает текст из аудио ответа."""
    if audio_data is None:
        return "Пожалуйста, запишите аудио перед распознаванием."
    
    if current_exam is None or current_question_index is None:
        return "Сначала сгенерируйте экзамен."
    
    try:
        # Сохраняем аудио во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = Path(tmp_file.name)
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                audio_array = convert_audio_to_float32(audio_array)
                sf.write(tmp_path, audio_array, sample_rate)
            else:
                shutil.copy(audio_data, tmp_path)
        
        # Распознаем речь
        asr_result = run_asr(tmp_path)
        recognized_text = asr_result.get("text", "")
        
        # Очищаем временный файл
        try:
            tmp_path.unlink()
        except:
            pass
        
        return recognized_text
    except Exception as e:
        return f"Ошибка при распознавании: {str(e)}"


def check_answer_submit(recognized_text, current_exam, current_question_index):
    """Проверяет ответ пользователя."""
    if not recognized_text or not recognized_text.strip():
        return "Пожалуйста, сначала распознайте ваш ответ.", None, None, ""
    
    if current_exam is None or current_question_index is None:
        return "Сначала сгенерируйте экзамен.", None, None, ""
    
    try:
        # Создаем глубокую копию экзамена для сохранения состояния
        import copy
        exam_copy = copy.deepcopy(current_exam)
        
        question_data = exam_copy["questions"][current_question_index]
        
        # Проверяем, не просмотрели ли уже объяснение
        if question_data["explanation_viewed"]:
            return "Вы уже просмотрели объяснение. Этот вопрос больше нельзя сдавать.", exam_copy, None, get_questions_progress(exam_copy)
        
        # Проверяем ответ
        is_correct, explanation = check_answer(
            question_data["question"],
            question_data["reference_answer"],
            recognized_text
        )
        
        # Сохраняем попытку
        attempt = {
            "answer": recognized_text,
            "is_correct": is_correct,
            "explanation": explanation
        }
        question_data["attempts"].append(attempt)
        
        # Обновляем лучший результат
        if is_correct and not question_data["best_result"]:
            question_data["best_result"] = True
        
        # Формируем сообщение только с результатом (без объяснения)
        result_text = "Принято" if is_correct else "Не принято"
        message = f"Результат проверки: {result_text}"
        
        # Обновляем прогресс
        progress_text = get_questions_progress(exam_copy)
        
        # Возвращаем обновленное состояние экзамена
        return message, exam_copy, None, progress_text
    except Exception as e:
        return f"Ошибка при проверке ответа: {str(e)}", None, None, ""


def view_explanation(current_exam, current_question_index):
    """Показывает объяснение и блокирует дальнейшие попытки."""
    if current_exam is None or current_question_index is None:
        return "Сначала сгенерируйте экзамен.", None
    
    try:
        # Создаем глубокую копию экзамена для сохранения состояния
        import copy
        exam_copy = copy.deepcopy(current_exam)
        
        question_data = exam_copy["questions"][current_question_index]
        question_data["explanation_viewed"] = True
        
        # Находим лучшую попытку или последнюю
        best_attempt = None
        for attempt in question_data["attempts"]:
            if attempt["is_correct"]:
                best_attempt = attempt
                break
        
        if not best_attempt and question_data["attempts"]:
            best_attempt = question_data["attempts"][-1]
        
        if best_attempt:
            explanation_text = f"Вопрос: {question_data['question']}\n\n"
            explanation_text += f"Правильный ответ: {question_data['reference_answer']}\n\n"
            explanation_text += f"Ваш ответ: {best_attempt['answer']}\n\n"
            explanation_text += f"Объяснение:\n{best_attempt['explanation']}"
        else:
            explanation_text = f"Вопрос: {question_data['question']}\n\n"
            explanation_text += f"Правильный ответ: {question_data['reference_answer']}\n\n"
            if question_data["attempts"]:
                explanation_text += f"Ваш ответ: {question_data['attempts'][-1]['answer']}\n\n"
            explanation_text += "Объяснение:\n"
            if question_data["attempts"]:
                explanation_text += question_data["attempts"][-1]["explanation"]
            else:
                explanation_text += "Вы не предоставили ответов на этот вопрос."
        
        # Возвращаем обновленное состояние экзамена
        return explanation_text, exam_copy
    except Exception as e:
        return f"Ошибка при получении объяснения: {str(e)}", None


def next_question(current_exam, current_question_index, current_user):
    """Переходит к следующему вопросу или завершает экзамен."""
    if current_exam is None or current_question_index is None:
        return None, None, None, None, "", "", "", gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Создаем глубокую копию экзамена для сохранения состояния
        exam_copy = copy.deepcopy(current_exam)
        
        next_index = current_question_index + 1
        
        if next_index >= len(exam_copy["questions"]):
            # Экзамен завершен, вычисляем оценку
            passed_count = sum(1 for q in exam_copy["questions"] if q["best_result"])
            grade = calculate_grade(passed_count)
            exam_copy["grade"] = grade
            exam_copy["passed_questions"] = passed_count
            
            # Сохраняем результат
            save_exam_result(current_user, exam_copy)
            
            grade_text = f"## Экзамен завершен!\n\n"
            grade_text += f"**Принято вопросов:** {passed_count} из {len(exam_copy['questions'])}\n\n"
            grade_text += f"**Оценка:** {grade}"
            
            progress_text = get_questions_progress(exam_copy)
            
            return None, None, None, grade_text, progress_text, "", "", gr.update(visible=False), gr.update(visible=False)
        else:
            # Переходим к следующему вопросу
            question_data = exam_copy["questions"][next_index]
            question_text = f"Вопрос {next_index + 1} из {len(exam_copy['questions'])}:\n\n{question_data['question']}"
            
            progress_text = get_questions_progress(exam_copy)
            
            # Управляем видимостью кнопок
            # Кнопка "Предыдущий вопрос" всегда видна (кроме первого вопроса)
            prev_visible = next_index > 0
            # Если это последний вопрос, скрываем кнопку "Следующий вопрос"
            next_visible = next_index < len(exam_copy["questions"]) - 1
            
            # Очищаем поля результата и объяснения
            return (
                exam_copy, 
                next_index, 
                question_text, 
                None,  # final_grade
                progress_text, 
                "",  # check_result
                "",  # explanation_display
                gr.update(visible=prev_visible),  # prev_question_btn
                gr.update(visible=next_visible)  # next_question_btn
            )
    except Exception as e:
        return None, None, None, None, f"Ошибка при переходе к следующему вопросу: {str(e)}", "", "", gr.update(visible=False), gr.update(visible=False)


def prev_question(current_exam, current_question_index):
    """Переходит к предыдущему вопросу."""
    if current_exam is None or current_question_index is None:
        return None, None, None, "", "", "", gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Создаем глубокую копию экзамена для сохранения состояния
        exam_copy = copy.deepcopy(current_exam)
        
        prev_index = current_question_index - 1
        
        if prev_index < 0:
            return exam_copy, current_question_index, None, "", "", "", gr.update(visible=False), gr.update(visible=True)
        
        # Переходим к предыдущему вопросу
        question_data = exam_copy["questions"][prev_index]
        question_text = f"Вопрос {prev_index + 1} из {len(exam_copy['questions'])}:\n\n{question_data['question']}"
        
        progress_text = get_questions_progress(exam_copy)
        
        # Управляем видимостью кнопок
        # Если это первый вопрос, скрываем кнопку "Предыдущий вопрос"
        prev_visible = prev_index > 0
        # Кнопка "Следующий вопрос" всегда видна (кроме последнего вопроса)
        next_visible = prev_index < len(exam_copy["questions"]) - 1
        
        # Очищаем поля результата и объяснения
        return (
            exam_copy, 
            prev_index, 
            question_text, 
            progress_text, 
            "",  # check_result
            "",  # explanation_display
            gr.update(visible=prev_visible),  # prev_question_btn
            gr.update(visible=next_visible)  # next_question_btn
        )
    except Exception as e:
        return None, None, None, f"Ошибка при переходе к предыдущему вопросу: {str(e)}", "", "", gr.update(visible=False), gr.update(visible=False)


# Функции для страницы профиля
def analyze_topics_progress(exams_data: dict, user_name: str) -> dict:
    """Анализирует прогресс по темам."""
    if not exams_data or user_name not in exams_data:
        return {}
    
    user_exams = exams_data[user_name].get("exams", [])
    
    # Загружаем банк вопросов из JSON файла
    question_to_topic = {}
    if QUESTION_BANK_FILE.exists():
        try:
            with open(QUESTION_BANK_FILE, 'r', encoding='utf-8') as f:
                bank_data = json.load(f)
                for item in bank_data.get("questions", []):
                    question_to_topic[item.get("question", "")] = item.get("topic", "неизвестно")
        except Exception:
            pass
    
    # Собираем статистику по темам
    topics_stats = defaultdict(lambda: {"total": 0, "passed": 0, "attempts": []})
    
    for exam in user_exams:
        for question_data in exam.get("questions", []):
            question_text = question_data.get("question", "")
            topic = question_to_topic.get(question_text, "неизвестно")
            
            topics_stats[topic]["total"] += 1
            if question_data.get("best_result", False):
                topics_stats[topic]["passed"] += 1
            
            # Собираем попытки
            for attempt in question_data.get("attempts", []):
                topics_stats[topic]["attempts"].append(attempt.get("is_correct", False))
    
    # Вычисляем процент успешности для каждой темы
    result = {}
    for topic, stats in topics_stats.items():
        success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        result[topic] = {
            "total_questions": stats["total"],
            "passed_questions": stats["passed"],
            "success_rate": round(success_rate, 1),
            "total_attempts": len(stats["attempts"]),
            "successful_attempts": sum(1 for a in stats["attempts"] if a)
        }
    
    return result


def create_exams_visualization(exams_data: dict, user_name: str) -> str:
    """Создает визуализацию экзаменов и возвращает путь к изображению."""
    if not exams_data or user_name not in exams_data:
        return None
    
    user_exams = exams_data[user_name].get("exams", [])
    if not user_exams:
        return None
    
    # Подготовка данных
    dates = []
    grades = []
    passed_counts = []
    
    for exam in sorted(user_exams, key=lambda x: x.get("date", "")):
        date_str = exam.get("date", "")
        try:
            date = datetime.fromisoformat(date_str)
            dates.append(date)
            grades.append(exam.get("grade", 0))
            passed_counts.append(exam.get("passed_questions", 0))
        except:
            continue
    
    if not dates:
        return None
    
    # Создаем фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'Статистика экзаменов пользователя {user_name}', fontsize=14, fontweight='bold')
    
    # График 1: Оценки по времени
    ax1.plot(dates, grades, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_ylabel('Оценка', fontsize=11)
    ax1.set_title('Динамика оценок', fontsize=12, fontweight='bold')
    ax1.set_ylim(1.5, 5.5)
    ax1.set_yticks([2, 3, 4, 5])
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # График 2: Количество принятых вопросов
    ax2.bar(range(len(dates)), passed_counts, color='#A23B72', alpha=0.7)
    ax2.set_xlabel('Номер экзамена', fontsize=11)
    ax2.set_ylabel('Принято вопросов', fontsize=11)
    ax2.set_title('Количество принятых вопросов по экзаменам', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 3.5)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Сохраняем в временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
    plt.close()
    
    return temp_file.name


def create_topics_visualization(topics_stats: dict) -> str:
    """Создает визуализацию прогресса по темам."""
    if not topics_stats:
        return None
    
    topics = list(topics_stats.keys())
    success_rates = [topics_stats[topic]["success_rate"] for topic in topics]
    total_questions = [topics_stats[topic]["total_questions"] for topic in topics]
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Прогресс по темам', fontsize=14, fontweight='bold')
    
    # График 1: Процент успешности по темам
    colors = ['#06A77D' if rate >= 70 else '#F18F01' if rate >= 50 else '#C73E1D' for rate in success_rates]
    bars1 = ax1.barh(topics, success_rates, color=colors, alpha=0.7)
    ax1.set_xlabel('Процент успешности (%)', fontsize=11)
    ax1.set_title('Процент успешности по темам', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцы
    for i, (bar, rate) in enumerate(zip(bars1, success_rates)):
        ax1.text(rate + 2, i, f'{rate}%', va='center', fontsize=10)
    
    # График 2: Количество вопросов по темам
    bars2 = ax2.barh(topics, total_questions, color='#2E86AB', alpha=0.7)
    ax2.set_xlabel('Количество вопросов', fontsize=11)
    ax2.set_title('Количество вопросов по темам', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars2, total_questions):
        ax2.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{count}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Сохраняем в временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
    plt.close()
    
    return temp_file.name


def load_user_profile(current_user):
    """Загружает профиль пользователя с визуализациями."""
    if not current_user:
        return (
            "Пожалуйста, сначала зарегистрируйтесь или войдите в систему.",
            None,  # exams_data_state
            f"Текущий пользователь: не авторизован",  # profile_user_info
            None,  # exams_chart
            None   # topics_chart
        )
    
    try:
        exams_data = load_user_exams(current_user)
        
        if not exams_data or current_user not in exams_data:
            return (
                f"Пользователь: {current_user}\n\nЭкзаменов не найдено.",
                exams_data if exams_data else {},
                f"Текущий пользователь: {current_user}",
                None,
                None
            )
        
        user_exams = exams_data[current_user].get("exams", [])
        
        if not user_exams:
            return (
                f"Пользователь: {current_user}\n\nЭкзаменов не найдено.",
                exams_data,
                f"Текущий пользователь: {current_user}",
                None,
                None
            )
        
        # Формируем красивый список экзаменов
        exams_list = f"## 📊 История экзаменов\n\n"
        exams_list += f"**Всего экзаменов:** {len(user_exams)}\n\n"
        exams_list += "| № | Дата | Оценка | Принято вопросов |\n"
        exams_list += "|---|---|---|---|\n"
        
        for i, exam in enumerate(user_exams, 1):
            exam_date = exam.get("date", "неизвестно")
            try:
                date_obj = datetime.fromisoformat(exam_date)
                date_str = date_obj.strftime("%d.%m.%Y %H:%M")
            except:
                date_str = exam_date[:10] if len(exam_date) >= 10 else exam_date
            
            grade = exam.get("grade", 0)
            passed = exam.get("passed_questions", 0)
            grade_emoji = "⭐" * grade if grade > 0 else "❌"
            exams_list += f"| {i} | {date_str} | {grade} {grade_emoji} | {passed}/3 |\n"
        
        # Создаем визуализации
        exams_chart = create_exams_visualization(exams_data, current_user)
        topics_stats = analyze_topics_progress(exams_data, current_user)
        topics_chart = create_topics_visualization(topics_stats) if topics_stats else None
        
        # Добавляем статистику по темам
        if topics_stats:
            exams_list += "\n## 📚 Прогресс по темам\n\n"
            exams_list += "| Тема | Вопросов | Принято | Успешность |\n"
            exams_list += "|---|---|---|---|\n"
            
            for topic, stats in sorted(topics_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True):
                success_emoji = "✅" if stats["success_rate"] >= 70 else "⚠️" if stats["success_rate"] >= 50 else "❌"
                exams_list += f"| {topic} | {stats['total_questions']} | {stats['passed_questions']} | {stats['success_rate']}% {success_emoji} |\n"
        
        return (
            exams_list,
            exams_data,
            f"Текущий пользователь: {current_user}",
            exams_chart,
            topics_chart
        )
    except Exception as e:
        return (
            f"Ошибка при загрузке профиля: {str(e)}",
            None,
            f"Текущий пользователь: {current_user if current_user else 'не авторизован'}",
            None,
            None
        )


def run_analysis(current_user, exams_data_state):
    """Запускает анализ экзаменов пользователя."""
    if not current_user:
        return "Пожалуйста, сначала зарегистрируйтесь или войдите в систему."
    
    if exams_data_state is None:
        exams_data_state = load_user_exams(current_user)
    
    if not exams_data_state or current_user not in exams_data_state:
        return "У вас пока нет сданных экзаменов для анализа."
    
    try:
        # Анализируем результаты
        analysis = analyze_exam_results(current_user, exams_data_state)
        
        # Формируем отчет
        report = f"## Анализ результатов экзаменов\n\n"
        report += f"**Всего экзаменов:** {analysis['total_exams']}\n\n"
        report += f"**Средняя оценка:** {analysis['average_grade']}\n\n"
        report += f"**Процент успешных ответов:** {analysis['success_rate']}%\n\n"
        report += f"**Принято вопросов:** {analysis['passed_questions_count']} из {analysis['total_questions_count']}\n\n"
        report += f"**Распределение оценок:**\n"
        report += f"- Оценка 5: {analysis['grade_distribution'].get(5, 0)}\n"
        report += f"- Оценка 4: {analysis['grade_distribution'].get(4, 0)}\n"
        report += f"- Оценка 3: {analysis['grade_distribution'].get(3, 0)}\n"
        report += f"- Оценка 2: {analysis['grade_distribution'].get(2, 0)}\n\n"
        
        # Получаем рекомендации
        recommendations = get_recommendations(current_user, exams_data_state)
        report += f"## Рекомендации\n\n{recommendations}"
        
        return report
    except Exception as e:
        return f"Ошибка при анализе: {str(e)}"


# Создание интерфейса Gradio
def create_interface():
    """Создает интерфейс Gradio с тремя вкладками."""
    
    with gr.Blocks(title="Система экзаменов") as app:
        # Состояние приложения
        current_user_state = gr.State(value=None)
        current_exam_state = gr.State(value=None)
        current_question_index_state = gr.State(value=None)
        question_bank_state = gr.State(value=None)
        exams_data_state = gr.State(value=None)
        
        gr.Markdown("# Система экзаменов с голосовой идентификацией")
        
        with gr.Tabs():
            # Вкладка регистрации
            with gr.Tab("Регистрация"):
                gr.Markdown("## Регистрация или вход в систему")
                gr.Markdown("Запишите аудио для идентификации или регистрации.")
                
                registration_audio = gr.Audio(label="Запись аудио", type="numpy", sources=["microphone"])
                process_btn = gr.Button("Обработать аудио", variant="primary")
                registration_status = gr.Textbox(label="Статус", interactive=False)
                
                # Блок подтверждения (показывается если пользователь найден)
                with gr.Row(visible=False) as confirm_block:
                    confirm_user_name = gr.Textbox(label="Имя пользователя", interactive=False)
                    with gr.Row():
                        confirm_btn = gr.Button("Подтвердить", variant="primary")
                        decline_btn = gr.Button("Отказаться", variant="secondary")
                
                # Блок регистрации (показывается если пользователь не найден)
                with gr.Row(visible=False) as register_block:
                    with gr.Column():
                        register_first_name = gr.Textbox(label="Имя", placeholder="Введите ваше имя")
                        register_last_name = gr.Textbox(label="Фамилия", placeholder="Введите вашу фамилию")
                        register_btn = gr.Button("Зарегистрироваться", variant="primary")
                
                registration_result = gr.Textbox(label="Результат", interactive=False)
                
                process_btn.click(
                    fn=process_audio_for_registration,
                    inputs=[registration_audio],
                    outputs=[current_user_state, confirm_block, confirm_user_name, register_block, registration_status]
                )
                
                confirm_btn.click(
                    fn=confirm_user,
                    inputs=[registration_audio, confirm_user_name],
                    outputs=[registration_result, current_user_state]
                )
                
                decline_btn.click(
                    fn=decline_confirmation,
                    inputs=[],
                    outputs=[confirm_block, confirm_user_name, register_block, registration_status]
                )
                
                register_btn.click(
                    fn=register_user,
                    inputs=[registration_audio, register_first_name, register_last_name],
                    outputs=[registration_result, current_user_state]
                )
            
            # Вкладка экзамена
            with gr.Tab("Экзамен"):
                gr.Markdown("## Прохождение экзамена")
                
                exam_user_info = gr.Markdown("Текущий пользователь: не авторизован")
                generate_exam_btn = gr.Button("Сгенерировать экзамен", variant="primary")
                questions_progress = gr.Markdown("")
                
                question_display = gr.Markdown("Сначала сгенерируйте экзамен.")
                
                with gr.Row():
                    answer_audio = gr.Audio(label="Запись ответа", type="numpy", sources=["microphone"], visible=False)
                    recognize_btn = gr.Button("Распознать", visible=False)
                
                recognized_text = gr.Textbox(label="Распознанный текст", visible=False, lines=5)
                
                with gr.Row():
                    submit_btn = gr.Button("Отправить на проверку", visible=False)
                    view_explanation_btn = gr.Button("Посмотреть объяснение", visible=False)
                
                with gr.Row():
                    prev_question_btn = gr.Button("Предыдущий вопрос", visible=False)
                    next_question_btn = gr.Button("Следующий вопрос", visible=False)
                
                check_result = gr.Textbox(label="Результат проверки", visible=False, lines=3)
                explanation_display = gr.Textbox(label="Объяснение", visible=False, lines=10)
                final_grade = gr.Markdown(visible=False)
                
                def update_user_info(current_user):
                    if current_user:
                        return f"Текущий пользователь: {current_user}"
                    return "Текущий пользователь: не авторизован"
                
                generate_exam_btn.click(
                    fn=generate_exam,
                    inputs=[current_user_state, question_bank_state],
                    outputs=[current_exam_state, current_question_index_state, question_display, 
                            answer_audio, recognize_btn, recognized_text, submit_btn, view_explanation_btn, questions_progress]
                ).then(
                    fn=update_user_info,
                    inputs=[current_user_state],
                    outputs=[exam_user_info]
                ).then(
                    fn=lambda: ("", ""),
                    outputs=[check_result, explanation_display]
                ).then(
                    fn=lambda: gr.update(visible=False),  # На первом вопросе скрываем "Предыдущий вопрос"
                    outputs=[prev_question_btn]
                ).then(
                    fn=lambda: gr.update(visible=True),  # Показываем "Следующий вопрос"
                    outputs=[next_question_btn]
                )
                
                recognize_btn.click(
                    fn=recognize_answer_audio,
                    inputs=[answer_audio, current_exam_state, current_question_index_state],
                    outputs=[recognized_text]
                )
                
                submit_btn.click(
                    fn=check_answer_submit,
                    inputs=[recognized_text, current_exam_state, current_question_index_state],
                    outputs=[check_result, current_exam_state, recognized_text, questions_progress]
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[check_result]
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[next_question_btn]
                )
                
                view_explanation_btn.click(
                    fn=view_explanation,
                    inputs=[current_exam_state, current_question_index_state],
                    outputs=[explanation_display, current_exam_state]
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[explanation_display]
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[next_question_btn]
                )
                
                prev_question_btn.click(
                    fn=prev_question,
                    inputs=[current_exam_state, current_question_index_state],
                    outputs=[current_exam_state, current_question_index_state, question_display, questions_progress, check_result, explanation_display, prev_question_btn, next_question_btn]
                )
                
                next_question_btn.click(
                    fn=next_question,
                    inputs=[current_exam_state, current_question_index_state, current_user_state],
                    outputs=[current_exam_state, current_question_index_state, question_display, final_grade, questions_progress, check_result, explanation_display, prev_question_btn, next_question_btn]
                )
            
            # Вкладка профиля
            with gr.Tab("Профиль"):
                gr.Markdown("## Профиль пользователя")
                
                profile_user_info = gr.Markdown("Текущий пользователь: не авторизован")
                load_profile_btn = gr.Button("Загрузить профиль", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        profile_content = gr.Markdown("Нажмите 'Загрузить профиль' для просмотра истории экзаменов.")
                    with gr.Column(scale=1):
                        exams_chart = gr.Image(label="График экзаменов", visible=False)
                        topics_chart = gr.Image(label="График по темам", visible=False)
                
                analyze_btn = gr.Button("Запустить анализ экзаменов")
                analysis_result = gr.Markdown()
                
                def update_charts_visibility(exams_img, topics_img):
                    """Обновляет видимость графиков."""
                    return (
                        gr.update(visible=exams_img is not None, value=exams_img) if exams_img else gr.update(visible=False),
                        gr.update(visible=topics_img is not None, value=topics_img) if topics_img else gr.update(visible=False)
                    )
                
                load_profile_btn.click(
                    fn=load_user_profile,
                    inputs=[current_user_state],
                    outputs=[profile_content, exams_data_state, profile_user_info, exams_chart, topics_chart]
                ).then(
                    fn=update_user_info,
                    inputs=[current_user_state],
                    outputs=[profile_user_info]
                ).then(
                    fn=update_charts_visibility,
                    inputs=[exams_chart, topics_chart],
                    outputs=[exams_chart, topics_chart]
                )
                
                analyze_btn.click(
                    fn=run_analysis,
                    inputs=[current_user_state, exams_data_state],
                    outputs=[analysis_result]
                )
        
        # Обновление информации о пользователе при изменении состояния
        def update_all_user_info(current_user):
            info = f"Текущий пользователь: {current_user}" if current_user else "Текущий пользователь: не авторизован"
            return info, info, info
        
        current_user_state.change(
            fn=update_all_user_info,
            inputs=[current_user_state],
            outputs=[exam_user_info, profile_user_info, profile_user_info]
        )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=6860, share=True, debug=True)

