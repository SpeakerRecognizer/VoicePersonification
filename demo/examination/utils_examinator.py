import json
import os
from pathlib import Path
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()


def _get_llm_config():
    """
    Получает конфигурацию LLM из переменных окружения.
    
    Returns:
        dict: Словарь с конфигурацией LLM
    """
    return {
        "base_url": os.getenv("LLM_BASE_URL", "http://nid-sc-29.ad.speechpro.com:17771/v1"),
        "api_key": os.getenv("LLM_API_KEY", "EMPTY"),
        "model": os.getenv("LLM_MODEL_PATH", "glm_local"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1"))
    }


def _create_llm():
    """
    Создает экземпляр LLM с конфигурацией из .env файла.
    
    Returns:
        ChatOpenAI: Экземпляр LLM
    """
    config = _get_llm_config()
    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        streaming=True,
        temperature=config["temperature"]
    )


def get_topics() -> List[str]:
    """
    Считывает темы из JSON файла и возвращает список тем для экзамена.
    
    Args:
        topics_file: Путь к файлу с темами (по умолчанию exam_topics.json)
    
    Returns:
        List[str]: Список тем для экзамена
    
    Raises:
        FileNotFoundError: Если файл с темами не найден
        json.JSONDecodeError: Если файл содержит невалидный JSON
    """
    topics_file = os.getenv("TOPIC_FILE", "exam_topics.json")
    topics_path = Path(topics_file)
    if not topics_path.exists():
        raise FileNotFoundError(f"Файл с темами не найден: {topics_file}")
    
    with open(topics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('exam_topics', [])


def generate_question_for_topic(topic: str) -> Tuple[str, str]:
    """
    Генерирует вопрос и ответ для заданной темы с помощью LLM.
    
    Args:
        topic: Тема для генерации вопроса
    
    Returns:
        Tuple[str, str]: Кортеж из (вопрос, ответ)
    """
    llm = _create_llm()
    
    prompt = f"""Сгенерируй один экзаменационный вопрос по теме "{topic}" и правильный ответ на него.

Формат ответа должен быть строго следующим (каждая строка отдельно):
ВОПРОС: [текст вопроса]
ОТВЕТ: [текст правильного ответа]

Вопрос должен быть конкретным и проверять понимание темы. Ответ должен быть полным и информативным."""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Парсим ответ
        question = None
        answer = None
        
        # Ищем вопрос и ответ в разных форматах
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            # Проверяем различные форматы
            if line.startswith('ВОПРОС:') or line.startswith('Вопрос:') or line.startswith('вопрос:'):
                question = line.split(':', 1)[1].strip() if ':' in line else line
            elif line.startswith('ОТВЕТ:') or line.startswith('Ответ:') or line.startswith('ответ:'):
                answer = line.split(':', 1)[1].strip() if ':' in line else line
                # Если ответ на следующей строке
                if not answer and i + 1 < len(lines):
                    answer = lines[i + 1].strip()
        
        # Если не удалось распарсить, пытаемся найти вопрос и ответ другим способом
        if not question or not answer:
            # Пробуем найти по знаку вопроса
            if '?' in content:
                parts = content.split('?', 1)
                if len(parts) == 2:
                    question = parts[0].strip() + '?'
                    answer = parts[1].strip()
                    # Убираем возможные префиксы
                    for prefix in ['Ответ:', 'ОТВЕТ:', 'ответ:', 'A:', 'A.', 'Ответ', 'ОТВЕТ']:
                        if answer.startswith(prefix):
                            answer = answer[len(prefix):].strip()
                        if answer.startswith(prefix + ':'):
                            answer = answer[len(prefix) + 1:].strip()
            else:
                # Если нет вопроса, создаем простой вопрос
                question = f"Расскажите о теме: {topic}"
                # Берем весь контент как ответ, но убираем возможные префиксы
                answer = content
                for prefix in ['Ответ:', 'ОТВЕТ:', 'ответ:', 'A:', 'A.']:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
        
        # Финальная проверка
        if not question or question.strip() == '':
            question = f"Что вы знаете о теме: {topic}?"
        if not answer or answer.strip() == '':
            answer = f"Правильный ответ по теме '{topic}' не был сгенерирован."
        
        return question.strip(), answer.strip()
    except Exception as e:
        # В случае ошибки возвращаем простой вопрос
        assert False, f"Ошибка при генерации вопроса: {str(e)}"


def create_question_bank() -> List[Tuple[str, str]]:
    """
    Создает банк вопрос-ответ на основе тем из файла.
    
    Args:
        topics_file: Путь к файлу с темами (по умолчанию exam_topics.json)
    
    Returns:
        List[Tuple[str, str]]: Список кортежей (вопрос, ответ) для каждой темы
    """
    topics = get_topics()
    question_bank = []
    
    for topic in topics:
        question, answer = generate_question_for_topic(topic)
        question_bank.append((question, answer))
    
    return question_bank


def check_answer(question: str, reference_answer: str, user_answer: str) -> Tuple[bool, str]:
    """
    Проверяет, правильный ли дан ответ на вопрос.
    
    Args:
        question: Текст вопроса
        reference_answer: Правильный ответ (эталон)
        user_answer: Ответ пользователя для проверки
    
    Returns:
        Tuple[bool, str]: Кортеж из (результат проверки, объяснение)
            - True если ответ правильный, False если неправильный
            - Объяснение результата проверки
    """
    llm = _create_llm()
    
    prompt = (
        f"Проверь, правильно ли пользователь ответил на вопрос: {question}. "
        f"Если правильно, напиши ВЕРНО, если неправильно, то верни НЕВЕРНО. "
        f"Ответ пользователя: {user_answer}. "
        f"Правильный ответ: {reference_answer}. "
        f"Ответ пользователя должен быть полным. "
        f"Однако если пользователь правильно понимает концепт, но отклоняется от правильного ответа, "
        f"то верни ВЕРНО, но с уточнениями."
    )
    
    try:
        response = llm.invoke(prompt)
        explanation = response.content.strip()
        
        # Проверяем наличие слова "НЕВЕРНО" в ответе
        is_correct = "НЕВЕРНО" not in explanation.upper()
        
        return is_correct, explanation
    except Exception as e:
        # В случае ошибки считаем ответ неправильным
        return False, f"Ошибка при проверке ответа: {str(e)}"


def analyze_exam_results(user_name: str, exams_data: dict) -> dict:
    """
    Анализирует результаты экзаменов пользователя и возвращает статистику.
    
    Args:
        user_name: Имя пользователя
        exams_data: Словарь с данными экзаменов пользователя (из JSON)
    
    Returns:
        dict: Словарь со статистикой:
            - total_exams: общее количество экзаменов
            - average_grade: средняя оценка
            - passed_questions_count: общее количество принятых вопросов
            - total_questions_count: общее количество вопросов
            - success_rate: процент успешных ответов
            - grade_distribution: распределение оценок
    """
    if not exams_data or user_name not in exams_data:
        return {
            "total_exams": 0,
            "average_grade": 0.0,
            "passed_questions_count": 0,
            "total_questions_count": 0,
            "success_rate": 0.0,
            "grade_distribution": {}
        }
    
    user_exams = exams_data[user_name].get("exams", [])
    
    if not user_exams:
        return {
            "total_exams": 0,
            "average_grade": 0.0,
            "passed_questions_count": 0,
            "total_questions_count": 0,
            "success_rate": 0.0,
            "grade_distribution": {}
        }
    
    total_exams = len(user_exams)
    total_grade = 0
    passed_questions_count = 0
    total_questions_count = 0
    grade_distribution = {2: 0, 3: 0, 4: 0, 5: 0}
    
    for exam in user_exams:
        grade = exam.get("grade", 0)
        total_grade += grade
        grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        passed_questions = exam.get("passed_questions", 0)
        passed_questions_count += passed_questions
        total_questions_count += 3  # В каждом экзамене 3 вопроса
    
    average_grade = total_grade / total_exams if total_exams > 0 else 0.0
    success_rate = (passed_questions_count / total_questions_count * 100) if total_questions_count > 0 else 0.0
    
    return {
        "total_exams": total_exams,
        "average_grade": round(average_grade, 2),
        "passed_questions_count": passed_questions_count,
        "total_questions_count": total_questions_count,
        "success_rate": round(success_rate, 2),
        "grade_distribution": grade_distribution
    }


def get_recommendations(user_name: str, exams_data: dict) -> str:
    """
    Генерирует рекомендации для пользователя на основе анализа его экзаменов.
    
    Args:
        user_name: Имя пользователя
        exams_data: Словарь с данными экзаменов пользователя (из JSON)
    
    Returns:
        str: Текст с рекомендациями
    """
    llm = _create_llm()
    
    analysis = analyze_exam_results(user_name, exams_data)
    
    if analysis["total_exams"] == 0:
        return "У вас пока нет сданных экзаменов. Начните с прохождения первого экзамена!"
    
    # Формируем информацию об экзаменах для промпта
    user_exams = exams_data[user_name].get("exams", [])
    exams_summary = []
    for exam in user_exams[-5:]:  # Берем последние 5 экзаменов для анализа
        exam_summary = f"Экзамен от {exam.get('date', 'неизвестно')}: оценка {exam.get('grade', 0)}, принято вопросов {exam.get('passed_questions', 0)}/3"
        exams_summary.append(exam_summary)
    
    prompt = f"""Проанализируй результаты экзаменов пользователя {user_name} и дай персональные рекомендации для улучшения.

Статистика пользователя:
- Всего экзаменов: {analysis['total_exams']}
- Средняя оценка: {analysis['average_grade']}
- Процент успешных ответов: {analysis['success_rate']}%
- Распределение оценок: {analysis['grade_distribution']}

Последние экзамены:
{chr(10).join(exams_summary)}

Дай конкретные рекомендации по улучшению результатов. Будь конструктивным и мотивирующим."""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Ошибка при генерации рекомендаций: {str(e)}"


def test_examinator():
    """
    Тест функций модуля utils_examinator.
    Проверяет:
    1. Чтение тем из файла
    2. Создание банка вопрос-ответ
    3. Проверку правильности ответа
    """
    print("=" * 70)
    print("ТЕСТ МОДУЛЯ UTILS_EXAMINATOR")
    print("=" * 70)
    
    # Тест 1: Чтение тем из файла
    print("\n[1] Тест чтения тем из файла...")
    try:
        topics = get_topics()
        print(f"    ✓ Темы успешно прочитаны: {len(topics)} тем")
        for i, topic in enumerate(topics, 1):
            print(f"    {i}. {topic}")
        
        if len(topics) == 0:
            print("    ⚠ Предупреждение: список тем пуст")
    except FileNotFoundError as e:
        print(f"    ✗ Ошибка: файл с темами не найден: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"    ✗ Ошибка: невалидный JSON в файле: {e}")
        return
    except Exception as e:
        print(f"    ✗ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Тест 2: Проверка конфигурации LLM
    print("\n[2] Тест конфигурации LLM...")
    try:
        config = _get_llm_config()
        print(f"    ✓ Конфигурация LLM получена:")
        print(f"    Base URL: {config['base_url']}")
        print(f"    Model: {config['model']}")
        print(f"    Temperature: {config['temperature']}")
        print(f"    API Key: {'установлен' if config['api_key'] else 'не установлен'}")
    except Exception as e:
        print(f"    ✗ Ошибка при получении конфигурации LLM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Тест 3: Генерация вопроса для одной темы
    print("\n[3] Тест генерации вопроса для темы...")
    if len(topics) > 0:
        test_topic = topics[0]
        print(f"    Тема: {test_topic}")
        try:
            question, answer = generate_question_for_topic(test_topic)
            print(f"    ✓ Вопрос и ответ успешно сгенерированы")
            print(f"    Вопрос: {question[:80]}...")
            print(f"    Ответ: {answer[:80]}...")
        except AssertionError as e:
            print(f"    ✗ Ошибка при генерации вопроса: {e}")
            return
        except Exception as e:
            print(f"    ✗ Неожиданная ошибка при генерации вопроса: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("    ⚠ Пропущено: нет тем для тестирования")
    
    # Тест 4: Создание банка вопрос-ответ (только для первой темы, чтобы не было долго)
    print("\n[4] Тест создания банка вопрос-ответ...")
    print("    Примечание: создается банк только для первой темы (чтобы ускорить тест)")
    try:
        # Временно сохраняем оригинальные темы
        original_topics = topics.copy()
        
        # Используем только первую тему для теста
        test_topics = [topics[0]] if len(topics) > 0 else []
        
        # Временно заменяем переменную окружения для теста
        import tempfile
        import json as json_module
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            test_data = {"exam_topics": test_topics}
            json_module.dump(test_data, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # Сохраняем оригинальное значение
        original_topic_file = os.getenv("TOPIC_FILE")
        
        try:
            # Устанавливаем временный файл
            os.environ["TOPIC_FILE"] = temp_file
            
            # Создаем банк вопросов
            question_bank = create_question_bank()
            
            print(f"    ✓ Банк вопросов успешно создан: {len(question_bank)} вопросов")
            for i, (q, a) in enumerate(question_bank, 1):
                print(f"    {i}. Вопрос: {q[:60]}...")
                print(f"       Ответ: {a[:60]}...")
        finally:
            # Восстанавливаем оригинальное значение
            if original_topic_file is None:
                os.environ.pop("TOPIC_FILE", None)
            else:
                os.environ["TOPIC_FILE"] = original_topic_file
            
            # Удаляем временный файл
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        print(f"    ✗ Ошибка при создании банка вопросов: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 5: Проверка правильности ответа
    print("\n[5] Тест проверки правильности ответа...")
    try:
        test_question = "Что такое машинное обучение?"
        test_reference = "Машинное обучение - это метод анализа данных, который автоматизирует построение аналитических моделей."
        
        # Тест с правильным ответом
        print("    [5.1] Тест с правильным ответом...")
        correct_answer = "Машинное обучение - это метод анализа данных, который автоматизирует построение аналитических моделей."
        is_correct, explanation = check_answer(test_question, test_reference, correct_answer)
        print(f"    Результат: {'✓ Правильно' if is_correct else '✗ Неправильно'}")
        print(f"    Объяснение: {explanation[:100]}...")
        
        # Тест с неправильным ответом
        print("\n    [5.2] Тест с неправильным ответом...")
        wrong_answer = "Машинное обучение - это процесс программирования компьютера."
        is_correct, explanation = check_answer(test_question, test_reference, wrong_answer)
        print(f"    Результат: {'✓ Правильно' if is_correct else '✗ Неправильно'}")
        print(f"    Объяснение: {explanation[:100]}...")
        
        print("    ✓ Проверка ответов выполнена успешно")
    except Exception as e:
        print(f"    ✗ Ошибка при проверке ответа: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    test_examinator()

