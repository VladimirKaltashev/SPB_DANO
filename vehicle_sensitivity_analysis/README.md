# Vehicle sensitivity — только Stage 1

Быстрая проверка неоднородной связи PriceShock с покупками топлива и штрафами
по объёму двигателя; мощность — альтернативный proxy. Предыдущие анализы не меняются.

Запуск из корня проекта:

```bash
.venv/bin/python vehicle_sensitivity_analysis/diagnostic.py
```

Главные результаты: [diagnostic_summary.md](outputs/diagnostic_summary.md),
[diagnostic_results.csv](outputs/diagnostic_results.csv), два PNG в `outputs/`.
Аудит распределений — `distribution_audit.csv`, корреляций — `correlations.csv`;
эффекты при −1/0/+1 SD — `marginal_response_table.csv`. Источники и проверки
сохранены в `metadata.json` и `validation.json`.

Дизайн зафиксирован в `outputs/analysis_plan.json` до моделей: все пять месяцев,
все 16 регионов, общая выборка клиентов с известными engine_volume и horsepower,
OLS с отдельными RegionFE и MonthFE, SE по регионам с поправкой и t(15).
Четыре основные модели в уровнях, две заранее заданные проверки log(1+литры).
Без подбора границ, исключения экстремумов, автоматического Stage 2 или mediation.

Используются проверенные очищенные месячные данные предыдущего анализа,
сверенные с полным `client_week_panel_v2.csv`; PriceShock не пересчитывается.
Медианы характеристик по автомобилям клиента воспроизводят прежнее назначение.
Пропуски не заполняются. Нули покупок остаются в выборке.
Исторические штрафы сохраняются на уровне автомобиля и не добавляются в модели.

Коэффициенты и кластерные ошибки всех шести interaction перепроверяются
независимым NumPy-расчётом. SHA-256 и перечень файлов трёх предыдущих анализов
проверяются после запуска. Требуются имеющиеся pandas, numpy, scipy, statsmodels,
matplotlib; выходные CSV не являются Excel-книгами.
