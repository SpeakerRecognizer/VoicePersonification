import gradio as gr
import soundfile as sf
import numpy as np
import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from utils import get_questions, get_result

# Импорты для ASR и идентификации
import sys
import importlib.util
SCRIPT_DIR = Path(__file__).parent.resolve()
VOICEPERSONIFICATION_ROOT = SCRIPT_DIR.parent.parent

# Импортируем example_connect_nid_vg_01 напрямую, избегая импорта всего пакета
example_connect_path = SCRIPT_DIR / "example_connect_nid_vg_01.py"
spec = importlib.util.spec_from_file_location("example_connect", example_connect_path)
example_connect = importlib.util.module_from_spec(spec)

# Временно добавляем пути только для этого импорта
original_path = sys.path.copy()
try:
    sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT / "services" / "protos"))
    sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT / "services" / "examples"))
    spec.loader.exec_module(example_connect)
finally:
    sys.path[:] = original_path

run_asr = example_connect.run_asr
run_identification = example_connect.run_identification

# Настройки
DATABASE_PATH = Path("multi_service_metadata.db")
USERS_DB_PATH = Path("users_database.json")
EXAMS_DB_PATH = Path("exams_database.json")
TEMP_AUDIO_DIR = "/mnt/asr_hot/dutov/study/nirsi/temp_audio"
HOST = "nid-vg-01"
VAD_PORT = 50051
SR_PORT = 50053
ASR_PORT = 50055

# Глобальные переменные
current_user = None
pending_registration_audio = None  # Сохраняем аудио для регистрации
exam_state = {
    "current_question_index": 0,
    "answers": {},
    "questions": [],
    "exam_id": None
}

# Инициализация баз данных
def init_databases():
    """Инициализация JSON баз данных"""
    if not USERS_DB_PATH.exists():
        with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    
    if not EXAMS_DB_PATH.exists():
        with open(EXAMS_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

def load_users_db():
    """Загрузка базы данных пользователей"""
    if not USERS_DB_PATH.exists():
        return {}
    with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users_db(data):
    """Сохранение базы данных пользователей"""
    with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_exams_db():
    """Загрузка базы данных экзаменов"""
    if not EXAMS_DB_PATH.exists():
        return {}
    with open(EXAMS_DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_exams_db(data):
    """Сохранение базы данных экзаменов"""
    with open(EXAMS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def convert_to_wav_16khz(audio_input, target_sr: int = 16000):
    """Конвертирует входное аудио в wav формат 16 кГц PCM16"""
    if audio_input is None:
        return None

    sr = None
    audio_array = None

    if isinstance(audio_input, dict):
        sr = audio_input.get("sample_rate") or audio_input.get("sr")
        audio_array = (audio_input.get("array") or
                       audio_input.get("data") or
                       audio_input.get("samples"))
    elif isinstance(audio_input, tuple) and len(audio_input) == 2:
        sr, audio_array = audio_input
    else:
        audio_array = audio_input

    if audio_array is None:
        return None

    audio_array = np.asarray(audio_array)
    if audio_array.size == 0:
        return None

    if sr is None:
        sr = 44100

    # Конвертируем в моно
    if audio_array.ndim == 2:
        if audio_array.shape[0] <= 4 and audio_array.shape[0] < audio_array.shape[1]:
            audio_array = audio_array.mean(axis=0)
        else:
            audio_array = audio_array.mean(axis=-1)
    elif audio_array.ndim > 2:
        audio_array = audio_array.reshape(audio_array.shape[0], -1).mean(axis=-1)

    # Нормализация
    if np.issubdtype(audio_array.dtype, np.integer):
        max_val = np.iinfo(audio_array.dtype).max
        audio_array = audio_array.astype(np.float32) / max_val
    else:
        audio_array = audio_array.astype(np.float32)
        peak = float(np.max(np.abs(audio_array)))
        if peak > 1.0:
            audio_array /= peak

    # Ресемплинг
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            audio_array = resample_poly(audio_array, target_sr, sr)
        except ImportError:
            import warnings
            warnings.warn("scipy не установлен, используется линейный ресемплинг")
            xp = np.linspace(0, len(audio_array) - 1, num=len(audio_array), dtype=np.float32)
            fp = audio_array
            new_length = int(round(len(audio_array) * target_sr / sr))
            x_new = np.linspace(0, len(audio_array) - 1, num=new_length, dtype=np.float32)
            audio_array = np.interp(x_new, xp, fp)
        sr = target_sr

    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    # Сохраняем в указанную директорию
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(TEMP_AUDIO_DIR, filename)
    sf.write(filepath, audio_int16, target_sr, subtype="PCM_16")
    return os.path.abspath(filepath)

# ============ ФУНКЦИИ РЕГИСТРАЦИИ ============

def process_login_audio(audio):
    """Обработка аудио для входа/регистрации"""
    global current_user, pending_registration_audio
    
    if audio is None:
        return "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Конвертируем аудио
        wav_path = convert_to_wav_16khz(audio)
        if wav_path is None:
            return "Ошибка: не удалось обработать аудио", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Сохраняем аудио для возможной регистрации
        pending_registration_audio = wav_path
        
        # Ищем пользователя в базе
        audio_path = Path(wav_path)
        ident_result = run_identification(audio_path, DATABASE_PATH, HOST, VAD_PORT, SR_PORT)
        
        users_db = load_users_db()
        user_name_db = ident_result.get("user_name", "")
        score = ident_result.get("score", 0.0)
        
        # Проверяем, есть ли пользователь в нашей базе
        if user_name_db and user_name_db in users_db and score > 0.5:
            # Пользователь найден
            user_data = users_db[user_name_db]
            name = user_data.get("name", "")
            surname = user_data.get("surname", "")
            current_user = {
                "user_name": user_name_db,
                "name": name,
                "surname": surname
            }
            
            return (f"Найден пользователь: {name} {surname}\nScore: {score:.4f}",
                   f"{name} {surname}",
                   gr.update(visible=True),  # Кнопка "Принять"
                   gr.update(visible=False),  # Форма регистрации
                   gr.update(visible=False))
        else:
            # Пользователь не найден - используем уже записанное аудио
            current_user = None
            return ("Пользователь не найден в базе. Пожалуйста, введите имя и фамилию для регистрации.",
                   "",
                   gr.update(visible=False),
                   gr.update(visible=True),  # Форма регистрации
                   gr.update(value="Используется уже записанное аудио", visible=True))
    except Exception as e:
        return f"Ошибка при обработке: {str(e)}", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def accept_user():
    """Подтверждение найденного пользователя"""
    global current_user
    if current_user:
        return f"Вход выполнен! Добро пожаловать, {current_user['name']} {current_user['surname']}"
    return "Ошибка: пользователь не найден"

def register_new_user(name, surname, audio):
    """Регистрация нового пользователя"""
    global current_user, pending_registration_audio
    
    if not name or not surname:
        return "Пожалуйста, введите имя и фамилию", gr.update(visible=True)
    
    # Используем сохраненное аудио
    if pending_registration_audio and os.path.exists(pending_registration_audio):
        wav_path = pending_registration_audio
    elif audio is not None:
        # Конвертируем новое аудио, если предоставлено (резервный вариант)
        wav_path = convert_to_wav_16khz(audio)
        if wav_path is None:
            return "Ошибка: не удалось обработать аудио", gr.update(visible=True)
    else:
        return "Ошибка: аудио не найдено. Пожалуйста, сначала запишите аудио для идентификации", gr.update(visible=True)
    
    try:
        # Обрабатываем через сервисы для получения embedding
        audio_path = Path(wav_path)
        
        # Используем process для получения embedding и сохранения в базу
        # Импортируем только необходимые функции напрямую
        multi_service_path = VOICEPERSONIFICATION_ROOT / "services" / "examples" / "multi_service_client.py"
        spec = importlib.util.spec_from_file_location("multi_service_client", multi_service_path)
        multi_service = importlib.util.module_from_spec(spec)
        
        # Временно добавляем пути только для protos
        original_path = sys.path.copy()
        try:
            sys.path.insert(0, str(VOICEPERSONIFICATION_ROOT / "services" / "protos"))
            spec.loader.exec_module(multi_service)
        finally:
            sys.path[:] = original_path
        
        ServiceEndpoint = multi_service.ServiceEndpoint
        process = multi_service.process
        create_session_factory = multi_service.create_session_factory
        enroll = multi_service.enroll
        
        from datetime import timezone
        
        vad_endpoint = ServiceEndpoint(HOST, VAD_PORT)
        sr_endpoint = ServiceEndpoint(HOST, SR_PORT)
        asr_endpoint = ServiceEndpoint(HOST, ASR_PORT)
        
        result = process(audio_path, vad_endpoint, sr_endpoint, asr_endpoint)
        
        # Генерируем уникальный user_name
        users_db = load_users_db()
        user_name = f"id{len(users_db) + 10000}"
        
        result["user_id"] = user_name.replace("id", "")
        result["user_name"] = user_name
        result["filename"] = audio_path.name
        
        # Сохраняем в базу данных сервисов
        session_maker = create_session_factory(DATABASE_PATH)
        with session_maker() as session:
            enroll(session, result)
        
        # Сохраняем в нашу базу пользователей
        users_db[user_name] = {
            "name": name,
            "surname": surname,
            "registered_at": datetime.now().isoformat()
        }
        save_users_db(users_db)
        
        current_user = {
            "user_name": user_name,
            "name": name,
            "surname": surname
        }
        
        # Очищаем сохраненное аудио
        pending_registration_audio = None
        
        return f"Регистрация успешна! Вход выполнен. Добро пожаловать, {name} {surname}!\nТеперь вы можете перейти к экзамену или просмотреть профиль.", gr.update(visible=False)
    except Exception as e:
        return f"Ошибка при регистрации: {str(e)}", gr.update(visible=True)

# ============ ФУНКЦИИ ЭКЗАМЕНА ============

def start_exam():
    """Инициализация экзамена"""
    global current_user, exam_state
    
    if current_user is None:
        return ("Ошибка: необходимо войти в систему", "", "", "", 
                gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=False))
    
    exam_state["current_question_index"] = 0
    exam_state["answers"] = {}
    exam_state["questions"] = get_questions()
    exam_state["exam_id"] = str(uuid.uuid4())
    
    if not exam_state["questions"]:
        return ("Ошибка: вопросы не найдены", "", "", "", 
                gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=False))
    
    question, reference = exam_state["questions"][0]
    question_text = f"Вопрос 1 из {len(exam_state['questions'])}: {question}"
    
    user_info = f"Сдающий: {current_user['name']} {current_user['surname']}"
    
    return (question_text, user_info, "", "", 
            gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=False))

def process_audio_recognition(audio):
    """Обработка записи аудио и распознавание текста через ASR"""
    if audio is None:
        return ""
    
    try:
        # Конвертируем в wav 16 кГц
        wav_path = convert_to_wav_16khz(audio)
        if wav_path is None:
            return "Ошибка: не удалось обработать аудио"
        
        # Используем run_asr для распознавания
        audio_path = Path(wav_path)
        asr_result = run_asr(audio_path, HOST, VAD_PORT, ASR_PORT)
        
        text = asr_result.get("text", "")
        
        # Сохраняем в состояние
        current_idx = exam_state["current_question_index"]
        answer_entry = exam_state["answers"].get(current_idx, {})
        answer_entry.update({
            "audio_path": wav_path,
            "transcript": text
        })
        exam_state["answers"][current_idx] = answer_entry
        
        return text if text else ""
    except Exception as e:
        return f"Ошибка распознавания: {str(e)}"

def submit_answer_wrapper(edited_text):
    """Обертка для отправки ответа"""
    question_index = exam_state["current_question_index"]
    return submit_answer(question_index, edited_text)

def submit_answer(question_index, edited_text):
    """Отправка ответа на проверку"""
    if not edited_text or not edited_text.strip():
        return "Пожалуйста, введите или отредактируйте ответ", gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)
    
    question, reference = exam_state["questions"][question_index]
    
    try:
        # Проверяем ответ
        is_correct, explanation = get_result(question, reference, edited_text)
        
        # Сохраняем результат
        answer_entry = exam_state["answers"].get(question_index, {})
        answer_entry.update({
            "text": edited_text,
            "transcript": edited_text,
            "result": is_correct,
            "explanation": explanation,
            "can_retry": True,
            "question": question,
            "reference_answer": reference
        })
        exam_state["answers"][question_index] = answer_entry
        
        result_text = "✓ Правильно!" if is_correct else "✗ Неправильно"
        
        return result_text, gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        return f"Ошибка при проверке: {str(e)}", gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)

def get_explanation_wrapper():
    """Обертка для получения объяснения"""
    question_index = exam_state["current_question_index"]
    return get_explanation(question_index)

def get_explanation(question_index):
    """Получение объяснения (блокирует возможность перезаписи)"""
    answer_data = exam_state["answers"].get(question_index)
    if not answer_data or "explanation" not in answer_data:
        return gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False)
    
    answer_data["can_retry"] = False
    exam_state["answers"][question_index] = answer_data
    
    explanation_text = answer_data.get("explanation") or ""
    return gr.update(value=explanation_text, visible=True), gr.update(interactive=False), gr.update(interactive=False)

def next_question():
    """Переход к следующему вопросу"""
    current_idx = exam_state["current_question_index"]
    
    if current_idx < len(exam_state["questions"]) - 1:
        exam_state["current_question_index"] = current_idx + 1
        question, reference = exam_state["questions"][current_idx + 1]
        question_text = f"Вопрос {current_idx + 2} из {len(exam_state['questions'])}: {question}"
        
        if current_idx + 1 in exam_state["answers"]:
            saved_answer = exam_state["answers"][current_idx + 1]
            result_flag = saved_answer.get("result")
            result_text = "✓ Правильно!" if result_flag else ("✗ Неправильно" if result_flag is not None else "")
            explanation = saved_answer.get("explanation") or ""
            can_retry = saved_answer.get("can_retry", True)
            explanation_visible = bool(explanation)
            explanation_button_active = bool(saved_answer.get("transcript"))
        else:
            result_text = ""
            explanation = ""
            can_retry = True
            explanation_visible = False
            explanation_button_active = False
        
        user_info = f"Сдающий: {current_user['name']} {current_user['surname']}" if current_user else ""
        
        return (question_text, user_info, "", result_text, 
                gr.update(value=explanation, visible=explanation_visible),
                gr.update(interactive=can_retry),
                gr.update(interactive=explanation_button_active),
                gr.update(interactive=True))
    else:
        user_info = f"Сдающий: {current_user['name']} {current_user['surname']}" if current_user else ""
        return ("Все вопросы пройдены. Нажмите 'Завершить экзамен'", user_info, "", "", 
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False))

def finish_exam():
    """Завершение экзамена и подсчет оценки"""
    global current_user, exam_state
    
    if current_user is None:
        return "Ошибка: пользователь не авторизован"
    
    total_questions = len(exam_state["questions"])
    if total_questions == 0:
        return "Ошибка: вопросы не загружены"
    
    correct_answers = sum(1 for ans in exam_state["answers"].values() if ans.get("result", False))
    percentage = (correct_answers / total_questions) * 100
    
    # Определяем оценку
    if percentage >= 90:
        grade = "A"
    elif percentage >= 80:
        grade = "B"
    elif percentage >= 70:
        grade = "C"
    elif percentage >= 50:
        grade = "D"
    else:
        grade = "E"
    
    # Сохраняем экзамен в базу данных
    exams_db = load_exams_db()
    user_name = current_user["user_name"]
    
    if user_name not in exams_db:
        exams_db[user_name] = []
    
    exam_data = {
        "exam_id": exam_state["exam_id"],
        "date": datetime.now().isoformat(),
        "grade": grade,
        "percentage": percentage,
        "correct_answers": correct_answers,
        "total_questions": total_questions,
        "answers": {}
    }
    
    # Сохраняем ответы
    for idx, answer in exam_state["answers"].items():
        exam_data["answers"][str(idx)] = {
            "question": answer.get("question", ""),
            "user_answer": answer.get("text", ""),
            "reference_answer": answer.get("reference_answer", ""),
            "result": answer.get("result", False),
            "explanation": answer.get("explanation", ""),
            "audio_path": answer.get("audio_path", "")
        }
    
    exams_db[user_name].append(exam_data)
    save_exams_db(exams_db)
    
    result_message = f"""
Экзамен завершен!

Правильных ответов: {correct_answers} из {total_questions}
Процент: {percentage:.1f}%
Оценка: {grade}
"""
    
    return result_message

# ============ ФУНКЦИИ ПРОФИЛЯ ============

def load_user_profile():
    """Загрузка профиля пользователя"""
    global current_user
    
    if current_user is None:
        return "Ошибка: необходимо войти в систему", "", ""
    
    user_name = current_user["user_name"]
    name = current_user["name"]
    surname = current_user["surname"]
    
    exams_db = load_exams_db()
    user_exams = exams_db.get(user_name, [])
    
    if not user_exams:
        return f"Профиль: {name} {surname}", "Экзамены не найдены", ""
    
    # Формируем список экзаменов
    exams_list = []
    for i, exam in enumerate(user_exams, 1):
        date = exam.get("date", "")[:10] if exam.get("date") else "Неизвестно"
        grade = exam.get("grade", "N/A")
        percentage = exam.get("percentage", 0)
        exams_list.append(f"{i}. Дата: {date} | Оценка: {grade} ({percentage:.1f}%)")
    
    exams_text = "\n".join(exams_list)
    
    return f"Профиль: {name} {surname}", exams_text, ""

def show_exam_details(exam_number):
    """Показать детали экзамена"""
    global current_user
    
    if current_user is None:
        return ""
    
    try:
        exam_idx = int(exam_number) - 1
        user_name = current_user["user_name"]
        exams_db = load_exams_db()
        user_exams = exams_db.get(user_name, [])
        
        if exam_idx < 0 or exam_idx >= len(user_exams):
            return "Неверный номер экзамена"
        
        exam = user_exams[exam_idx]
        details = []
        details.append(f"Экзамен от {exam.get('date', '')[:19]}")
        details.append(f"Оценка: {exam.get('grade', 'N/A')} ({exam.get('percentage', 0):.1f}%)")
        details.append(f"Правильных ответов: {exam.get('correct_answers', 0)} из {exam.get('total_questions', 0)}")
        details.append("\n" + "="*50 + "\n")
        
        answers = exam.get("answers", {})
        for idx in sorted(answers.keys(), key=lambda x: int(x)):
            answer_data = answers[idx]
            details.append(f"Вопрос {int(idx) + 1}: {answer_data.get('question', '')}")
            details.append(f"Ваш ответ: {answer_data.get('user_answer', '')}")
            details.append(f"Правильный ответ: {answer_data.get('reference_answer', '')}")
            details.append(f"Результат: {'✓ Правильно' if answer_data.get('result') else '✗ Неправильно'}")
            if answer_data.get('explanation'):
                details.append(f"Объяснение: {answer_data.get('explanation')}")
            details.append("-" * 50)
        
        return "\n".join(details)
    except Exception as e:
        return f"Ошибка: {str(e)}"

# ============ СОЗДАНИЕ ИНТЕРФЕЙСА ============

init_databases()

with gr.Blocks(title="Система экзаменов") as app:
    gr.Markdown("# Система экзаменов")
    
    with gr.Tabs():
        # Вкладка входа/регистрации
        with gr.Tab("Вход / Регистрация"):
            gr.Markdown("## Вход или регистрация")
            gr.Markdown("Запишите аудио для идентификации")
            
            audio_input_login = gr.Audio(sources=["microphone"], type="numpy", label="Запись аудио")
            process_login_button = gr.Button("Обработать аудио", variant="primary")
            
            login_result = gr.Textbox(label="Результат", interactive=False)
            found_user_display = gr.Textbox(label="Найденный пользователь", interactive=False, visible=False)
            
            with gr.Row():
                accept_button = gr.Button("Принять", visible=False)
                register_form = gr.Column(visible=False)
            
            with register_form:
                gr.Markdown("### Регистрация нового пользователя")
                gr.Markdown("**Используется уже записанное аудио для идентификации**")
                name_input = gr.Textbox(label="Имя", placeholder="Введите ваше имя")
                surname_input = gr.Textbox(label="Фамилия", placeholder="Введите вашу фамилию")
                register_button = gr.Button("Зарегистрироваться", variant="primary")
                register_output = gr.Textbox(label="Результат регистрации", interactive=False)
            
            process_login_button.click(
                fn=process_login_audio,
                inputs=audio_input_login,
                outputs=[login_result, found_user_display, accept_button, register_form, register_output]
            )
            
            accept_button.click(
                fn=accept_user,
                outputs=login_result
            )
            
            def register_and_update(name, surname):
                result_text, form_visibility = register_new_user(name, surname, None)
                return result_text, form_visibility, result_text  # Обновляем и login_result
            
            register_button.click(
                fn=register_and_update,
                inputs=[name_input, surname_input],
                outputs=[register_output, register_form, login_result]
            )
        
        # Вкладка экзамена
        with gr.Tab("Экзамен"):
            gr.Markdown("## Прохождение экзамена")
            
            user_info_display = gr.Textbox(label="Информация о сдающем", interactive=False)
            start_exam_button = gr.Button("Начать экзамен", variant="primary")
            question_display = gr.Textbox(label="Вопрос", interactive=False)
            
            with gr.Row():
                audio_input_exam = gr.Audio(sources=["microphone"], type="numpy", label="Запись ответа")
                recognize_button = gr.Button("Распознать аудио")
            
            recognized_text = gr.Textbox(label="Распознанный текст (можно редактировать)", lines=5)
            
            with gr.Row():
                submit_button = gr.Button("Отправить ответ", variant="primary")
                explanation_button = gr.Button("Получить объяснение", interactive=False)
            
            result_display = gr.Textbox(label="Результат", interactive=False)
            explanation_display = gr.Textbox(label="Объяснение", interactive=False, visible=False)
            
            with gr.Row():
                next_question_button = gr.Button("Следующий вопрос")
                finish_exam_button = gr.Button("Завершить экзамен", variant="stop")
            
            exam_result = gr.Textbox(label="Итоговый результат", interactive=False, visible=False)
            
            start_exam_button.click(
                fn=start_exam,
                outputs=[question_display, user_info_display, recognized_text, result_display, explanation_display, 
                        submit_button, explanation_button]
            )
            
            recognize_button.click(
                fn=process_audio_recognition,
                inputs=audio_input_exam,
                outputs=recognized_text
            )
            
            submit_button.click(
                fn=submit_answer_wrapper,
                inputs=recognized_text,
                outputs=[result_display, explanation_display, submit_button, explanation_button]
            )
            
            explanation_button.click(
                fn=get_explanation_wrapper,
                outputs=[explanation_display, submit_button, explanation_button]
            )
            
            next_question_button.click(
                fn=next_question,
                outputs=[question_display, user_info_display, recognized_text, result_display, explanation_display, 
                        submit_button, explanation_button, next_question_button]
            )
            
            finish_exam_button.click(
                fn=finish_exam,
                outputs=exam_result
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=exam_result
            )
        
        # Вкладка профиля
        with gr.Tab("Профиль"):
            gr.Markdown("## Профиль сдающего")
            
            profile_info = gr.Textbox(label="Информация", interactive=False)
            refresh_profile_button = gr.Button("Обновить профиль", variant="primary")
            
            exams_list_display = gr.Textbox(label="Список экзаменов", lines=10, interactive=False)
            
            gr.Markdown("### Детали экзамена")
            exam_number_input = gr.Number(label="Номер экзамена", value=1, precision=0)
            show_details_button = gr.Button("Показать детали", variant="primary")
            exam_details_display = gr.Textbox(label="Детали экзамена", lines=20, interactive=False)
            
            refresh_profile_button.click(
                fn=load_user_profile,
                outputs=[profile_info, exams_list_display, exam_details_display]
            )
            
            show_details_button.click(
                fn=show_exam_details,
                inputs=exam_number_input,
                outputs=exam_details_display
            )

if __name__ == "__main__":
    app.launch(share=True, debug=True)
