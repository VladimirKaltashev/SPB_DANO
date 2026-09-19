"""Обязательный PNG-backend для всех отчётов, в том числе без GUI."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spb-matplotlib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:
    raise RuntimeError(
        "Не удалось загрузить Matplotlib: графики обязательны. "
        "Запустите uv sync, затем uv run python main.py в папке проекта."
    ) from error

__all__ = ["plt"]
