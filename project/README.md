# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

## 1. Паспорт проекта

- **Название проекта:** Ticket Priority Triage — классификация приоритета обращений в поддержку
- **Автор:** `Ходеев Евгений Игоревич`
- **Группа:** `ФВБО-01-22`
- **Контакт:** `<@legendjeka>`

**Цель проекта:**  
Построить воспроизводимый учебный ML-проект, который обучает модель классификации приоритета обращений в поддержку и предоставляет минимальный рабочий сервис для демонстрации инференса.

**Краткая идея:**  
По тексту обращения и нескольким структурированным признакам модель определяет приоритет заявки: `low`, `medium`, `high`. Используются синтетические, не конфиденциальные данные, пайплайн scikit-learn и HTTP-сервис с endpoint'ами `/health` и `/predict`. Проект демонстрирует работу с данными, baseline, улучшенную модель, метрики, артефакты, конфигурацию, тесты и воспроизводимый запуск.

---

## 2. Структура проекта

Все файлы проекта размещены внутри папки `project/`, как требуется в задании. В архиве нет кода или материалов проекта вне этой папки.

```text
project/
├── artifacts/              # обученная модель, метрики, model card, примеры предсказаний
├── configs/                # config.json и .env.example
├── data/                   # синтетический датасет sample_tickets.csv
├── examples/               # пример JSON-запроса для инференса
├── notebooks/              # EDA и эксперименты
├── src/                    # основной код проекта
│   ├── config.py           # загрузка конфигурации
│   ├── data.py             # генерация и валидация данных
│   ├── evaluate.py         # оценка сохранённой модели
│   ├── modeling.py         # пайплайн модели и функции предсказания
│   ├── predict.py          # CLI-инференс
│   ├── service.py          # HTTP-сервис без Flask/FastAPI
│   └── train.py            # обучение baseline и финальной модели
├── tests/                  # pytest-тесты
├── Dockerfile              # сценарий развёртывания через Docker
├── README.md               # инструкция по запуску
├── report.md               # отчёт по проекту
├── requirements.txt        # зависимости
├── SECURITY.md             # заметки по безопасности
└── self-checklist.md       # самопроверка по критериям
```

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.11` рекомендуется; Dockerfile использует `python:3.11-slim`.
- Системные зависимости не требуются.
- Docker опционален.

### 3.2. Установка окружения

```bash
cd project
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Сгенерировать данные

В репозитории уже есть `data/sample_tickets.csv`. При необходимости его можно пересоздать:

```bash
python -m src.data --output data/sample_tickets.csv --rows 600 --seed 42
```

### 4.2. Запустить обучение модели

```bash
python -m src.train --config configs/config.json
```

После запуска появятся/обновятся:

- `artifacts/model.joblib` — финальная модель;
- `artifacts/metrics.json` — метрики baseline и финальной модели;
- `artifacts/model_card.md` — краткое описание модели.

Текущие результаты на отложенной выборке:

| Модель | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline: most frequent class | 0.4133 | 0.1950 |
| Final: TF-IDF + OneHot + Logistic Regression | 0.9400 | 0.9419 |

### 4.3. Запустить CLI-предсказание

```bash
python -m src.predict \
  --input examples/sample_input.json \
  --output artifacts/sample_predictions.csv
```

### 4.4. Запустить сервис

Сервис реализован на стандартной библиотеке Python (`http.server`), без Flask/FastAPI. Это сохраняет минимальный стек и одновременно выполняет требование проекта про сервис, `/health` и `/predict`.

```bash
python -m src.service --config configs/config.json
```

По умолчанию сервис запускается на порту `8000`.

Проверка работоспособности:

```bash
curl http://127.0.0.1:8000/health
```

Пример запроса к `/predict`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_input.json
```

Формат входа для одного объекта:

```json
{
  "text": "Production outage: all users cannot log in.",
  "channel": "phone",
  "customer_tier": "enterprise",
  "product_area": "login",
  "sentiment_score": -0.88,
  "account_age_days": 980
}
```

Формат ответа:

```json
{
  "count": 1,
  "predictions": [
    {
      "priority": "high",
      "probabilities": {
        "high": 0.82,
        "low": 0.04,
        "medium": 0.14
      }
    }
  ]
}
```

### 4.5. Запустить через Docker

```bash
docker build -t ticket-priority-triage .
docker run -p 8000:8000 ticket-priority-triage
```

---

## 5. Данные

Источник данных — синтетическая генерация в `src/data.py`. Реальные персональные данные, конфиденциальные обращения и секреты не используются.

Файл `data/sample_tickets.csv` содержит 600 строк и следующие признаки:

- `text` — текст обращения;
- `channel` — канал обращения: `email`, `chat`, `phone`, `web`;
- `customer_tier` — тариф клиента: `free`, `standard`, `premium`, `enterprise`;
- `product_area` — зона продукта;
- `sentiment_score` — числовая оценка тональности от -1 до 1;
- `account_age_days` — возраст аккаунта в днях;
- `priority` — целевая переменная: `low`, `medium`, `high`.

В данных специально добавлен небольшой шум разметки, чтобы задача не была идеально простой.

---

## 6. Тесты

Запуск тестов:

```bash
pytest tests
```

Проверяются:

- схема и классы синтетических данных;
- обучение baseline и финальной модели;
- качество финальной модели относительно baseline;
- CLI/API-логика предсказаний;
- нормализация JSON-пayload для сервиса.

---

## 7. Демонстрация на защите

План демонстрации:

1. Показать структуру проекта: `src/`, `data/`, `notebooks/`, `artifacts/`, `tests/`.
2. Открыть `notebooks/01_eda.ipynb` и показать распределение классов и основные признаки.
3. Запустить обучение:

   ```bash
   python -m src.train --config configs/config.json
   ```

4. Показать сравнение baseline и финальной модели в `artifacts/metrics.json`.
5. Запустить сервис:

   ```bash
   python -m src.service --config configs/config.json
   ```

6. Проверить:

   ```bash
   curl http://127.0.0.1:8000/health
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/sample_input.json
   ```

---

## 8. Ограничения и дальнейшая работа

Ограничения текущей версии:

- данные синтетические, поэтому качество не переносится напрямую на реальные обращения;
- модель решает только задачу классификации приоритета;
- нет авторизации и постоянного хранения запросов;
- сервис предназначен для демонстрации инженерного пайплайна, а не для промышленной эксплуатации.

Возможные улучшения:

- заменить синтетику открытым реальным датасетом поддержки;
- добавить мониторинг распределения входных признаков;
- добавить версионирование моделей;
- расширить набор экспериментов и подобрать гиперпараметры через cross-validation;
- добавить более строгую валидацию входных данных.

---

## 9. Оценка проекта по чеклисту

Проект закрывает 10/10 пунктов самопроверки из `self-checklist.md`:

- сервис запускается;
- `/predict` использует реальную модель;
- есть EDA и эксперименты;
- есть baseline и улучшенная модель;
- код вынесен в `src/`;
- есть Dockerfile;
- есть `.env.example`, секретов нет;
- есть логи и `/health`;
- выбор модели обоснован в отчёте;
- README и отчёт описывают демонстрацию.
