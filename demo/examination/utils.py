import json
import numpy as np
from langchain_openai import ChatOpenAI
import os
import requests

def check_verification(person_unknown_wav, name, surname):
    with open(f"verification_{name}_{surname}.json", "r") as f:
        data = json.load(f)

    person_verified_wav = data["wav"]
    return True 

def audio_to_text(path):
    asr_url = 'http://nid-sc-28.ad.speechpro.com:8976' 
    if not os.path.exists(path):
        return None
    # Очищаем путь и получаем абсолютный путь
    path = path.strip().replace('"', '').replace("'", '')
    path = path.split('.wav')[0]+'.wav'
    # Убеждаемся, что путь абсолютный
    path = os.path.abspath(path)
    
    # Кодируем путь для URL
    from urllib.parse import quote
    encoded_path = quote(path, safe='')
    print(encoded_path)
    
    url = f"{asr_url}/rec_file?path={encoded_path}".replace('%2F', '/')
    response = requests.post(url)
    text = ''
    # print(response.json())
    for i in response.json()['recognition']['asr_predicted']:
        text += i['word'] + i['punctuation_mark'] + ' '
    return text.strip()



def save_verification(person_wav, name, surname):
    data = {
        "name": name,
        "surname": surname,
        "wav": person_wav,
    }
    with open(f"verification_{name}_{surname}.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_topics():
    """Возвращает список тем для экзамена"""
    return [
        'машинное обучение',
        'цифровая обработка речевых сигналов',
        'теория принятия решений в машинном обучении',
        'доменная адаптация и калибровка модели верификации диктора'
    ]

def generate_question_for_topic(topic):
    """Генерирует вопрос и ответ для заданной темы с помощью LLM"""
    llm = ChatOpenAI(
        base_url="http://nid-sc-34:16666/v1",
        api_key="EMPTY",
        model="GLM_air_awq",
        streaming=True,
        temperature=0.1
    )
    
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
        
        return [question.strip(), answer.strip()]
    except Exception as e:
        # В случае ошибки возвращаем простой вопрос
        return [f"Расскажите о теме: {topic}", f"Правильный ответ по теме {topic} не был сгенерирован из-за ошибки: {str(e)}"]

def get_questions():
    """Генерирует вопросы для всех тем"""
    topics = get_topics()
    questions = []
    for topic in topics:
        question, answer = generate_question_for_topic(topic)
        questions.append([question, answer])
    return questions

def get_result(question, reference_answer, answer):
    llm = ChatOpenAI(
    base_url="http://nid-sc-34:16666/v1",
    api_key="EMPTY",
    model="GLM_air_awq",
    streaming=True,
    temperature=0.1
)
    response = llm.invoke(f"Проверь, правильно ли пользователь ответил на вопрос: {question}. Если правильно, напиши ВЕРНО, если неправильно, то верни НЕВЕРНО. Ответ пользователя: {answer}. Правильный ответ: {reference_answer}. Ответ пользователя должен быть полным. Однако если пользователь правльно понимает концепт, но отклоняется от правильного ответа, то верни ВЕРНО, но с уточнениями.")
    # print(response.content)
    if response.content.count("НЕВЕРНО") > 0:
        return False, response.content
    else:
        return True, response.content

# print(audio_to_text("/mnt/asr_hot/dutov/study/nirsi/samp.wav"))