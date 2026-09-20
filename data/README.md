# Данные

- `raw/` — исходные `clients_demographics.csv`, `fines_2026.csv` и
  `fuel_transaction.csv`. Главный пайплайн их только читает.
- `processed/` — результаты `pipeline/clean_data.py`; их можно полностью
  пересоздать командой `uv run python main.py`.
- `reference/client_week_panel_v2.csv` — прежний контрольный снимок недельной
  панели. Он не используется как источник аналитики.
