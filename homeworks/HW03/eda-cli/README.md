# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

Дополнительные параметры команды `report` (HW03):

- `--title` – заголовок отчёта (первая строка в `report.md`).
- `--max-hist-columns` – сколько числовых колонок включить в набор гистограмм.
- `--max-cat-columns` – сколько категориальных колонок анализировать (top-таблицы).
- `--top-k-categories` – сколько top-значений сохранять для каждой категориальной колонки.
- `--min-missing-share` – порог доли пропусков: колонки с missing_share >= порога попадают в список «проблемных».

Пример вызова с кастомными настройками:

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports_example \
  --title "EDA: example.csv" \
  --max-hist-columns 4 \
  --top-k-categories 10 \
  --min-missing-share 0.1
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
