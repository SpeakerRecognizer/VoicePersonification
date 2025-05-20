# 🎙️ Voice Personification

Персонификация по голосу для систем искусственного интеллекта  
_Разработка открытого модуля биометрической идентификации по голосу с использованием современных SSL-моделей_

---

## 🚀 Описание проекта

Voice Personification — это модульная система распознавания пользователя по голосу, ориентированная на интеграцию в LLM, чат-боты, образовательные платформы и иные ИИ-сервисы.  
Система поддерживает короткие аудио(от 2 секунд), сложные акустические условия и многопользовательские сценарии.

---

## 🔧 Основные возможности

- 🔒 Биометрическая верификация личности по голосу
- ⚡ Работа с короткими фрагментами речи (2–5 секунд)
- 🌍 Устойчивость к шуму, реверберации и разным языкам
- 🧠 Предобученные модели на основе совремемнных архитектур: wav2vec-BERT, Whisper, Next-TDNN
- 🎙️ Поддержка сегментации речи
- 🧩 Готова к интеграции с LLM, чат-ботами, образовательными системами

---

---

## ⚙️ Установка

```bash
git clone https://github.com/SpeakerRecognizer/VoicePersonification.git
cd VoicePersonification
pip install -r requirements.txt
```

---

## 🧪 Быстрый старт

### Загрузка и подготовка данных

Скачайте и подготовьте датасет VoxCeleb1:

```bash
bash scripts/download_and_preprocess_data.sh data/download_data data/raw_data data/scp data/protocols
```
После выполнения скрипта вы получите:
- `data/download_data/` — архивы и протоколы
- `data/raw_data/` — распакованные аудио
- `data/scp/` — файлы `wav.scp` и `utt2spk`
- `data/protocols/` — Kaldi-протоколы для тестовдав

### Тестирование моделей
Запустите тест на подготовленных данных:

```bash
python -m VoicePersonification.main \
  -cp=experiments/ecapa-tdnn \
  -cn=test
```

> Вместо `ecapa-tdnn` укажите нужную модель:  
> `experiments/itmo_personification_model_large`, `experiments/itmo_personification_model_fast`, `experiments/itmo_personification_model_segmentation`, и т.д.
---


## 🧠 Наши модели

| Название модели                        | Особенности                                                                          | Архитектура    | Размер |
|----------------------------------------|---------------------------------------------------------------------------------------|----------------|----------|
| `itmo_personification_model_large`  | Основная модель с высокой точностью и устойчивостью; использует SSL-предобучение     | wav2vec-BERT   |  XX        |
| `itmo_personification_model_fast`      | Лёгкая и быстрая модель для верификации                        | Next-TDNN      |       XX   |
| `itmo_personification_model_segmentation` | Позволяет точно выделять участки речи отдельных говорящих для улучшения верификации | Whisper        |     XX     |
| `ecapa-tdnn baseline`                  | Бейзлайн-модель для сравнения  | ECAPA-TDNN     |   XX       |

---

## 📊 Сравнение 

Сравнение проводилось по метрике **EER (Equal Error Rate, %)**
| Модель                                      | VoxCeleb1 | Voices | NIST SRE 2016 | NIST SRE 2019 |
|--------------------------------------------|-----------|--------|----------------|----------------|
| `itmo personification model large`      | XX.XX     | XX.XX  | XX.XX          | XX.XX          |
| `itmo personification model fast`          | XX.XX     | XX.XX  | XX.XX          | XX.XX          |
| `itmo personification model segmentation`                       | XX.XX     | XX.XX  | XX.XX          | XX.XX          |
| `ecapa-tdnn baseline` | XX.XX        | XX.XX     | XX.XX             | XX.XX            |

---

## 📒 Туториалы и примеры

- [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) — быстрый старт и верификация
- [`examples/`](examples/) — примеры аудиофайлов enrollment и test

---

## 🎓 Курс по распознаванию диктора

Если вам интересно, как работает голосовая биометрия изнутри — загляните в репозиторий нашего [курса](https://github.com/itmo-mbss-lab/sr_labs_book/blob/main/README.md)!  
Там мы рассказываем, как строятся системы распознавания дикторов, как оценивается их качество, и как они применяются в реальных сценариях.  
Все задания, материалы и код доступны в репозитории.

> 💡 Курс читается в рамках магисторской программы "Речевые технологии и машинное обучение" в Университете ИТМО, но мы будем рады, если он окажется полезен и за ее пределами.

---


## 📚 Публикации

Ниже — некоторые статьи, опубликованные в рамках или при поддержке этого проекта.  

- 🔖 [Joint Voice Activity Detection and Quality Estimation for Efficient Speech Preprocessing](https://ieeexplore.ieee.org/document/10977856)
  
- 🔖 [Robust Speaker Recognition for Whispered Speech](https://ieeexplore.ieee.org/document/10977907)  

- 🔖 [Accurate Speaker Counting, Diarization and Separation for Advanced Recognition of Multichannel Multispeaker Conversations](https://www.sciencedirect.com/science/article/abs/pii/S0885230825000051)


---

## 📄 Лицензия

Этот проект распространяется под лицензией  
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Вы можете:

- 📤 **Распространять** — копировать и публиковать проект  
- 🎛️ **Модифицировать** — изменять, адаптировать и использовать в некоммерческих целях  

При соблюдении условий:

- **Указание авторства** — необходимо указать авторов проекта  
- **Только некоммерческое использование** — использование в коммерческих целях запрещено без отдельного разрешения  

> ☑️ **Исключение**: коммерческое использование разрешено только Университету ИТМО.  
> Все остальные должны получить отдельное согласие от авторов.

📄 [Полный текст лицензии](https://creativecommons.org/licenses/by-nc/4.0/)  
📬 По вопросам коммерческого использования: **VoicePersonificationITMO@gmail.com**

## 📝 Цитирование

Если вы используете этот проект в научной публикации, пожалуйста, цитируйте нас следующим образом:

```bibtex
@misc{itmo_voice_personification_2025,
  title     = {Voice Personification for Artificial Intelligence Systems},
  author    = {Khmelev, N. and Novoselov, S. Zorkina, A. and et.al},
  year      = {2025},
  note      = {ITMO University, open-source implementation},
  howpublished = {\url{https://github.com/YOUR_ORG/voice-personification}}
}
```
## 🙏 Благодарности
Проект был выполнен при поддержке Университета ИТМО (Россия)
в рамках гранта «Научно-исследовательская работа в сфере искусственного интеллекта»,
проект № 640110 — «Персонификация по голосу для систем искусственного интеллекта».



