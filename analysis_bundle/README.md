# Пакет расчётов и очищенных данных

Эта папка — единая навигационная точка для передачи проекта. Она собрана без
дублирования больших CSV: каталоги `cleaned_data/` и `results/`, а также
`code_pipeline/` и `code_current_hypothesis/` являются junction-ссылками на
рабочие каталоги проекта. Поэтому изменения видны в обоих местах, а pipeline
продолжает работать из корня репозитория.

## Содержимое

- `ANALYSIS_REPORT.md` — расчёты, выводы, ограничения и методика;
- `CLEANED_DATA_README.md` — схема и правила очистки;
- `code_pipeline/` — `clean_data.py`, `preprocessing.py`, источники данных и
  служебная визуализация;
- `code_current_hypothesis/` — подготовка клиентов, группы, проверка гипотезы,
  модель FuelRate/FineRate и презентационные графики;
- `cleaned_data/` — очищенные CSV, недельная панель и аудит возвратов/дублей;
- `results/` — таблицы, графики и validation всех расчётных веток.

## Быстрые ссылки

- Основные числа: `results/presentation_math_model/statistics_for_slides.csv`;
- Robustness: `results/presentation_math_model/robustness_results.csv`;
- Очищенная недельная панель: `cleaned_data/client_week_panel.csv`;
- Очищенные события топлива: `cleaned_data/fuel_clean.csv`;
- Очищенные события штрафов: `cleaned_data/fines_clean.csv`;
- Клиентские признаки: `cleaned_data/clients_clean.csv`.

## Запуск

Команды запускаются из корня проекта, а не из этой папки:

```bash
uv run python main.py
uv run python -m current_hypothesis.presentation_charts
```

Junction-ссылки рассчитаны на Windows и локальный checkout. При переносе
только этой папки в другое место нужно сохранить весь репозиторий целиком или
заново создать ссылки на `pipeline/`, `current_hypothesis/`, `data/processed/`
и `outputs/`.
