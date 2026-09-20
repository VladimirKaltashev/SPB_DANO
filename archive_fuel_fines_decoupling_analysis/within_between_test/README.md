# Совместный тест within–between

Одна OLS-модель в параметризации Mundlak: штрафы на отклонение покупок клиента
от его среднего, само среднее и Month FE. Когорта и литры/100 сохранены.
Wald проверяет равенство двух наклонов с полной совместной ковариацией.

Основные ошибки по регионам; альтернатива по клиентам. Используется ранг
совместной модели K=7 для CR1, F(1,G−1) для Wald, t(G−1) для интервалов.
Предыдущие результаты не изменяются; новых графиков и Stage 2 нет.

```bash
.venv/bin/python archive_fuel_fines_decoupling_analysis/within_between_test/diagnostic.py
```

[Отчёт](outputs/report.md) · [Wald-тесты](outputs/wald_equality_tests.csv)
