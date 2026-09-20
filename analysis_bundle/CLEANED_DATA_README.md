# Очищенные данные

Эта папка создаётся `pipeline/clean_data.py` из трёх файлов в `data/raw/`:

- `clients_demographics.csv` → `clients_clean.csv` и `clients_demographics_clean.csv`;
- `fines_2026.csv` → `fines_clean.csv`;
- `fuel_transaction.csv` → `fuel_clean.csv`, `fuel_refund_matches.csv` и
  `fuel_negative_unresolved.csv`.

## Основные файлы

| Файл | Строки | Ключ | Роль |
|---|---:|---|---|
| `clients_clean.csv` | 28 190 | `client_id × auto_document_id` | клиентские и автомобильные признаки плюс аудит конфликтов |
| `fines_clean.csv` | 72 959 | `bill_id` | очищенные штрафные события |
| `fuel_clean.csv` | 379 699 | `order_id` | топливные события с признаками возврата |
| `client_week_panel.csv` | 590 525 | `client_id × week` | аналитическая панель для расчётов до/после |
| `client_auto_conflicts.csv` | 94 | исходная пара | все повторяющиеся client×auto строки |
| `client_auto_conflicts_resolved.csv` | 47 | исходная пара | аудит свёртки повторов |
| `fuel_refund_matches.csv` | 23 | отрицательная операция | однозначные возвраты |
| `fuel_negative_unresolved.csv` | 1 477 | отрицательная операция | неразрешённые возвраты, сохранённые для аудита |

`client_week_panel.csv` содержит 25 675 уникальных клиентов и 23 недели
(`2026-03-30`—`2026-08-31`), включая нулевые недели. Основное аналитическое
окно использует полные недели в периоде 1 апреля—31 августа 2026 года.

## Аудит и повторная сборка

- [`cleaning_report.md`](cleaning_report.md) — количественный отчёт очистки;
- [`cleaning_metadata.json`](cleaning_metadata.json) — параметры и контрольные размеры;
- `data/reference/client_week_panel_v2.csv` — контрольный снимок, не источник анализа.

Повторная сборка:

```bash
uv run python main.py
```

Очистка не изменяет `data/raw/`. Все CSV в этой папке включены в Git LFS;
после клонирования репозитория требуется `git lfs pull`.
