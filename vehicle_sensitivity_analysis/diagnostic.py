"""Stage 1 only: python vehicle_sensitivity_analysis/diagnostic.py."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
GROUPS = ["LOW", "MEDIUM", "HIGH"]
VARIABLES = ["engine_volume", "horsepower"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def table(frame):
    def fmt(x):
        if pd.isna(x):
            return "—"
        return f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x).replace("|", "/")
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False, name=None)]])


def describe(series, variable, population):
    q = series.quantile([.01, .05, .25, .5, .75, .95, .99])
    return dict(population=population, variable=variable, N=len(series),
        missing_share=series.isna().mean(), zero_share=series.eq(0).mean(),
        mean=series.mean(), std=series.std(ddof=0), min=series.min(),
        **{name: q.loc[p] for name, p in zip(["p1", "p5", "p25", "median", "p75", "p95", "p99"], q.index)},
        max=series.max(), skew=series.skew())


def load_data():
    previous = ROOT / "fines_per_liter_analysis/outputs"
    archive = ROOT / "archive_monthly_risk_clustering/outputs"
    usage = json.loads((ROOT / "usage_composition_analysis/outputs/metadata.json").read_text())
    sources = usage["sources"]
    for source in sources.values():
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Verified source changed: {source['path']}")
    archive_meta = json.loads((archive / "metadata.json").read_text())
    for key in ["panel_v2", "fines", "fuel", "vehicles"]:
        assert sources[key]["sha256"] == archive_meta["sources"][key]["sha256"]
    dtypes = {"client_id": "string", "region": "string", "auto_document_id": "string"}
    dm = pd.read_csv(previous / "tables/driver_month.csv", dtype=dtypes)
    archived = pd.read_csv(archive / "tables/driver_region_month.csv", dtype=dtypes)
    prices = pd.read_csv(previous / "tables/region_month_prices.csv", dtype=dtypes)
    vehicles = pd.read_csv(sources["vehicles"]["path"], dtype=dtypes)
    assert not dm.duplicated(["client_id", "month"]).any()
    assert not vehicles.duplicated(["client_id", "auto_document_id"]).any()
    matched = dm.merge(archived[["client_id", "region", "month", "fuel_liters", "fines_count"]],
        on=["client_id", "region", "month"], how="outer", validate="one_to_one", suffixes=("", "_archive"))
    for outcome in ["fuel_liters", "fines_count"]:
        assert np.allclose(matched[outcome], matched[outcome + "_archive"])
    assert np.allclose(prices.price_shock, prices.avg_fuel_price / prices.baseline_price_region - 1)
    assert set(prices.baseline_month) == {"2026-04"}
    # Reproduce the existing parser without importing or rewriting the old analysis.
    parsed = vehicles.engine_type.astype("string").str.extract(r"^\s*([\d.,]+)\s*\(\s*([\d.,]+)\s*л\.\s*с\.\s*\)\s*$")
    for i, name in enumerate(VARIABLES):
        vehicles[name] = pd.to_numeric(parsed[i].str.replace(",", "."), errors="coerce")
    invalid = vehicles.engine_volume.lt(0) | vehicles.horsepower.le(0)
    vehicles.loc[invalid, VARIABLES] = np.nan
    vehicles["auto_price"] = pd.to_numeric(vehicles.price, errors="raise")
    vehicles["flag_review"] = vehicles.engine_volume.eq(0) | vehicles.engine_volume.gt(8) | vehicles.horsepower.gt(1000)
    attributes = archived.drop_duplicates("client_id")[["client_id", "region", "car_count",
        "subscription_creation_date", *VARIABLES, "n_cars_with_missing_engine", "has_extreme_engine"]].copy()
    rebuilt = vehicles.groupby("client_id")[VARIABLES].median()
    for name in VARIABLES:
        assert np.allclose(attributes.set_index("client_id")[name].sort_index(), rebuilt[name].sort_index(), equal_nan=True)
        assert archived.groupby("client_id")[name].nunique().le(1).all()
    auto = vehicles.groupby("client_id").agg(auto_price=("auto_price", "median"),
        observed_car_count=("auto_document_id", "nunique"),
        car_identifiers=("auto_document_id", lambda x: json.dumps(sorted(x.tolist()))))
    attributes = attributes.join(auto, on="client_id")
    assert attributes.car_count.eq(attributes.observed_car_count).all()
    dm = dm.merge(attributes.drop(columns="region"), on="client_id", validate="many_to_one")
    dm = dm.merge(prices[["region", "month", "price_shock"]], on=["region", "month"], validate="many_to_one")
    assert not dm.price_shock.isna().any()
    assert np.isfinite(dm[["fuel_liters", "fines_count", "price_shock"]]).all().all()
    assert dm.fuel_liters.ge(0).all() and dm.fines_count.ge(0).all()
    assert dm.groupby("client_id").size().eq(5).all()
    assert set(dm.month) == {f"2026-{m:02d}" for m in range(4, 9)} and dm.region.nunique() == 16
    attributes["analytical_eligible"] = attributes[VARIABLES].notna().all(axis=1)
    eligible = attributes.loc[attributes.analytical_eligible].copy()
    sample = dm.loc[dm.client_id.isin(eligible.client_id)].copy().reset_index(drop=True)
    standardization = []
    for name in VARIABLES:
        mean, sd = eligible[name].mean(), eligible[name].std(ddof=0)
        assert sd > 0
        sample[name + "_z"] = (sample[name] - mean) / sd
        standardization.append(dict(variable=name, mean=mean, std_ddof0=sd, n_clients=len(eligible)))
    sample["log_fuel"] = np.log1p(sample.fuel_liters)
    q1, q2 = eligible.engine_volume.quantile([1 / 3, 2 / 3])
    assert q1 < q2
    sample["engine_group"] = np.select([sample.engine_volume.le(q1), sample.engine_volume.le(q2)], ["LOW", "MEDIUM"], default="HIGH")
    # Keep all vehicle-level historic-fine fields with their original unit; do not sum potentially duplicated client history.
    vehicles.to_csv(OUT / "vehicle_attributes.csv", index=False)
    vehicles.loc[vehicles.flag_review].to_csv(OUT / "vehicle_values_for_review.csv", index=False)
    attributes.to_csv(OUT / "client_attributes.csv", index=False)
    sample.to_csv(OUT / "diagnostic_client_month.csv", index=False)
    pd.DataFrame(standardization).to_csv(OUT / "standardization.csv", index=False)
    audit = [describe(frame[name], name, population) for population, frame in
        [("all_clients", attributes), ("analytical_clients", eligible), ("vehicle_rows", vehicles)]
        for name in [*VARIABLES, "auto_price"]]
    audit += [describe(frame[name], name, population) for population, frame in
        [("all_client_months", dm), ("analytical_client_months", sample)] for name in ["fuel_liters", "fines_count"]]
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT / "distribution_audit.csv", index=False)
    correlations = []
    for population, frame in [("all_clients", attributes), ("analytical_clients", eligible)]:
        for left, right in [("engine_volume", "horsepower"), ("engine_volume", "auto_price"), ("horsepower", "auto_price")]:
            pair = frame[[left, right]].dropna()
            correlations.append(dict(population=population, left=left, right=right, N=len(pair), pearson_r=pair[left].corr(pair[right])))
    correlations = pd.DataFrame(correlations)
    correlations.to_csv(OUT / "correlations.csv", index=False)
    group_info = sample.drop_duplicates("client_id").groupby("engine_group").agg(
        n_clients=("client_id", "size"), minimum=("engine_volume", "min"), maximum=("engine_volume", "max"), mean=("engine_volume", "mean")).reindex(GROUPS).reset_index()
    group_info.to_csv(OUT / "engine_tertiles.csv", index=False)
    meta = dict(sources=sources, all_clients=len(attributes), all_client_months=len(dm),
        analytical_clients=len(eligible), analytical_client_months=len(sample), excluded_missing_clients=len(attributes)-len(eligible),
        regions=sample.region.nunique(), months=sorted(sample.month.unique()),
        multi_car_clients=int(attributes.car_count.gt(1).sum()), multi_car_share=float(attributes.car_count.gt(1).mean()),
        analytical_multi_car_clients=int(eligible.car_count.gt(1).sum()),
        vehicle_rows=len(vehicles), invalid_parsed_vehicle_rows=int(invalid.sum()),
        missing_parsed_vehicle_rows=int(vehicles[VARIABLES].isna().any(axis=1).sum()),
        clients_with_some_missing_car_features=int(attributes.n_cars_with_missing_engine.gt(0).sum()),
        zero_engine_vehicle_rows=int(vehicles.engine_volume.eq(0).sum()),
        engine_over_8_vehicle_rows=int(vehicles.engine_volume.gt(8).sum()), horsepower_over_1000_vehicle_rows=int(vehicles.horsepower.gt(1000).sum()),
        tertile_q1=float(q1), tertile_q2=float(q2), full_total_liters=float(dm.fuel_liters.sum()),
        full_total_fines=int(dm.fines_count.sum()), analytical_total_liters=float(sample.fuel_liters.sum()), analytical_total_fines=int(sample.fines_count.sum()))
    return sample, audit, correlations, group_info, meta


def fit_models(sample):
    fe = pd.get_dummies(sample[["region", "month"]], drop_first=True, dtype=float)
    results, all_coefficients, magnitudes, checks = [], [], [], []
    for variable in VARIABLES:
        z = sample[variable + "_z"]
        X = pd.DataFrame({"const": 1., "price_shock": sample.price_shock, "vehicle_z": z,
                          "interaction": sample.price_shock * z}).join(fe)
        assert np.linalg.matrix_rank(X.to_numpy()) == X.shape[1]
        for outcome in ["fuel_liters", "fines_count", "log_fuel"]:
            fit = sm.OLS(sample[outcome], X).fit()
            robust = fit.get_robustcov_results(cov_type="cluster", groups=sample.region,
                use_correction=True, df_correction=True, use_t=True)
            ci = robust.conf_int()
            for i, term in enumerate(X.columns):
                all_coefficients.append(dict(outcome=outcome, vehicle_variable=variable, term=term,
                    beta=robust.params[i], SE=robust.bse[i], p_value=robust.pvalues[i],
                    ci95_low=ci[i, 0], ci95_high=ci[i, 1]))
            j = X.columns.get_loc("interaction")
            results.append(dict(outcome=outcome, vehicle_variable=variable, interaction_beta=robust.params[j],
                SE=robust.bse[j], p_value=robust.pvalues[j], ci95_low=ci[j, 0], ci95_high=ci[j, 1],
                N=int(fit.nobs), n_clients=sample.client_id.nunique(), n_regions=sample.region.nunique(),
                R_squared=fit.rsquared, sign="positive" if robust.params[j] > 0 else "negative",
                interaction_per_10pp_shock=robust.params[j] * .1))
            # Independent NumPy OLS and region-cluster sandwich, including small-sample correction.
            xn, y = X.to_numpy(), sample[outcome].to_numpy()
            beta = np.linalg.lstsq(xn, y, rcond=None)[0]
            residual = y - xn @ beta
            scores = pd.DataFrame(xn * residual[:, None]).groupby(sample.region.to_numpy()).sum().to_numpy()
            bread = np.linalg.inv(xn.T @ xn)
            g, n, k = len(scores), len(xn), xn.shape[1]
            covariance = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - k)
            se = np.sqrt(covariance[j, j]); pvalue = 2 * t.sf(abs(beta[j] / se), g - 1)
            assert np.allclose([beta[j], se, pvalue], [robust.params[j], robust.bse[j], robust.pvalues[j]], rtol=1e-7, atol=1e-9)
            checks.append(dict(outcome=outcome, vehicle_variable=variable, independent_OLS_and_cluster_SE="PASS"))
            for value in [-1, 0, 1]:
                contrast = np.zeros(k); contrast[1] = .1; contrast[j] = .1 * value
                effect = float(contrast @ robust.params)
                effect_se = float(np.sqrt(contrast @ robust.cov_params() @ contrast))
                half = t.ppf(.975, g - 1) * effect_se
                magnitudes.append(dict(outcome=outcome, vehicle_variable=variable, vehicle_z=value,
                    change_per_10pp_shock=effect, SE=effect_se, ci95_low=effect-half, ci95_high=effect+half,
                    percent_change_in_geometric_1_plus_liters=100*np.expm1(effect) if outcome == "log_fuel" else np.nan))
            print(f"{outcome} × {variable}: interaction={robust.params[j]:.6g}, p={robust.pvalues[j]:.6g}", flush=True)
    result = pd.DataFrame(results)
    order = {"fuel_liters": 0, "fines_count": 1, "log_fuel": 2}
    result = result.assign(_order=result.outcome.map(order)).sort_values(["_order", "vehicle_variable"]).drop(columns="_order")
    result.to_csv(OUT / "diagnostic_results.csv", index=False)
    pd.DataFrame(all_coefficients).to_csv(OUT / "all_model_coefficients.csv", index=False)
    magnitude = pd.DataFrame(magnitudes)
    magnitude.to_csv(OUT / "marginal_response_table.csv", index=False)
    price_fe = sm.OLS(sample.price_shock, sm.add_constant(fe)).fit()
    return result, magnitude, checks, float(price_fe.resid.std(ddof=0))


def draw_plots(sample, groups, results):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    colors = {"LOW": "#397CB4", "MEDIUM": "#DB9A36", "HIGH": "#A74666"}
    panel = sample.groupby(["region", "month", "engine_group"], as_index=False).agg(
        n_drivers=("client_id", "size"), fuel_liters=("fuel_liters", "mean"),
        fines_per_driver=("fines_count", "mean"), price_shock=("price_shock", "first"))
    assert panel.groupby(["region", "engine_group"]).n_drivers.nunique().eq(1).all()
    panel.to_csv(OUT / "engine_tertile_region_month.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for outcome, source_outcome, filename, title, ylabel in [
        ("fuel_liters", "fuel_liters", "01_engine_volume_fuel_response.png", "Объём двигателя и наблюдаемые покупки топлива", "Средние литры на клиента в месяц"),
        ("fines_per_driver", "fines_count", "02_engine_volume_fines_response.png", "Объём двигателя и штрафная активность", "Штрафов на клиента в месяц")]:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), sharex=True, sharey=True)
        for ax, group in zip(axes, GROUPS):
            cell = panel.loc[panel.engine_group.eq(group)]
            row = groups.set_index("engine_group").loc[group]
            ax.scatter(cell.price_shock, cell[outcome], c=colors[group], alpha=.65, s=29, edgecolor="white", linewidth=.3)
            ax.set_title(f"{group}: {row.minimum:g}–{row.maximum:g} л\nn = {int(row.n_clients):,} клиентов", fontsize=11)
            ax.set_xlabel("PriceShock к апрелю")
            ax.set_xticks([0, .05, .10, .15])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(alpha=.18)
        axes[0].set_ylabel(ylabel)
        stat = results.loc[results.outcome.eq(source_outcome) & results.vehicle_variable.eq("engine_volume")].iloc[0]
        fig.suptitle(title, fontsize=17)
        fig.text(.5, .075, f"Основной continuous interaction с RegionFE + MonthFE: β = {stat.interaction_beta:+.3f}; p = {stat.p_value:.4g}.", ha="center", fontsize=10)
        fig.text(.5, .025, "Точки — исходные средние региона × месяца; без поправки на FE. Терцили только для визуализации, тест — на client × month.", ha="center", fontsize=9)
        fig.tight_layout(rect=[0, .12, 1, .91])
        fig.savefig(OUT / filename, dpi=180, facecolor="white")
        plt.close(fig)


def write_summary(results, magnitudes, audit, correlations, groups, meta):
    def row(outcome, variable):
        return results.loc[results.outcome.eq(outcome) & results.vehicle_variable.eq(variable)].iloc[0]
    engine, engine_log = row("fuel_liters", "engine_volume"), row("log_fuel", "engine_volume")
    hp, hp_log = row("fuel_liters", "horsepower"), row("log_fuel", "horsepower")
    fine = row("fines_count", "engine_volume")
    e_signal = engine.p_value < .05
    e_stable = e_signal and engine_log.p_value < .05 and engine.sign == engine_log.sign
    hp_agrees = engine.sign == hp.sign and engine_log.sign == hp_log.sign
    verdict = dict(H1="YES" if e_stable else "INCONCLUSIVE",
        direction=("larger engine responds MORE" if engine.interaction_beta < 0 else "larger engine responds LESS") if e_stable else "NO CLEAR DIFFERENCE",
        H2="YES" if fine.p_value < .05 else "INCONCLUSIVE",
        HORSEPOWER="YES" if hp_agrees and hp.p_value < .05 and hp_log.p_value < .05 else "MIXED",
        SHOULD_WE_CONTINUE="YES" if e_stable else "MAYBE" if e_signal or engine_log.p_value < .05 else "NO",
        stage_2_executed=False)
    dump("verdict.json", verdict)
    selected = results[["outcome", "vehicle_variable", "interaction_beta", "SE", "p_value", "ci95_low", "ci95_high", "N", "R_squared", "sign"]]
    selected = selected.assign(interpretation=[
        "Различие не установлено" if r.p_value >= .05 else
        "Более положительный наклон при большей характеристике" if r.interaction_beta > 0 else
        "Более отрицательный наклон при большей характеристике" for r in results.itertuples()])
    descriptions = []
    for r in results.itertuples():
        unit = "л/клиент" if r.outcome == "fuel_liters" else "штрафа/клиент" if r.outcome == "fines_count" else "log(1 + литры)"
        descriptions.append(f"- {r.outcome} × {r.vehicle_variable}: при +1 SD характеристики разница ответа на +10 п.п. PriceShock = {r.interaction_beta*.1:+.6g} {unit}; p={r.p_value:.6g}.")
    text = f"""# Vehicle sensitivity: Stage 1

## Шесть диагностических моделей

{table(selected)}

β — разница наклонов PriceShock при +1 SD характеристики; PriceShock записан долей (0,10 = +10 п.п.). Все шесть моделей используют одну выборку.

## Краткий verdict

H1: {verdict['H1']}. Направление: {verdict['direction']}.

H2: {verdict['H2']}.

HORSEPOWER: {verdict['HORSEPOWER']}.

SHOULD WE CONTINUE THIS HYPOTHESIS? {verdict['SHOULD_WE_CONTINUE']}.

В уровнях EngineVolume × PriceShock = {engine.interaction_beta:+.3f} (p={engine.p_value:.4g}); в log(1+литры) = {engine_log.interaction_beta:+.3f} (p={engine_log.p_value:.4g}). Horsepower даёт соответственно {hp.interaction_beta:+.3f} (p={hp.p_value:.4g}) и {hp_log.interaction_beta:+.3f} (p={hp_log.p_value:.4g}). В текущем запуске значимый положительный сигнал есть только в логарифмическом исходе, тогда как отрицательные оценки в литрах статистически неотличимы от нуля на уровне 5%. Это смешанный результат по масштабу исхода, а не противоположные механизмы engine_volume и horsepower. Log(1+литры) уменьшает влияние крупных покупок и иначе учитывает переходы к нулю, поэтому эти модели оценивают разные аспекты наблюдаемых покупок. Для штрафов различие по engine_volume не установлено (p={fine.p_value:.4g}); большого полного анализа на основании этого этапа не требуется, возможна только целевая дальнейшая проверка после обсуждения.

Эффекты уровней и log(1+литры) показаны совместно; трансформация не выбирается по значимости. Незначимость не доказывает отсутствия неоднородности. Направление взаимодействия описывает различие условных наклонов, а не обязательно абсолютное падение у всех автомобилей. Только Stage 1 выполнен; Stage 2 и mediation не запускались.

## Очищенные источники и сопоставимость

Использованы результаты очистки ochistka.ipynb / preprocessing.py: fuel_clean.csv (physical_fuel_volume_main), fines_clean.csv и clients_demographics_clean.csv. Очищенные операции ранее сверены с client_week_panel_v2.csv для каждого клиента × недели без расхождений. Месячные суммы и региональный PriceShock взяты из того же проверенного расчёта fines_per_liter_analysis и проверены по SHA-256; исходные файлы не перезаписываются. PriceShock = средневзвешенная региональная цена месяца / апрельская цена региона − 1; период апрель–август 2026.

Полная исходная панель: **{meta['all_clients']:,} клиентов, {meta['all_client_months']:,} клиент-месяцев**. Аналитическая: **{meta['analytical_clients']:,} клиентов, {meta['analytical_client_months']:,} клиент-месяцев, {meta['regions']} регионов, 5 месяцев**. Исключены только {meta['excluded_missing_clients']} клиентов с пропусками engine_volume / horsepower; все месяцы оставшихся клиентов, включая нулевые покупки, сохранены. Ограничения по положительным апрельским покупкам из usage_composition_analysis не переносятся. Полные суммы: {meta['full_total_liters']:,.2f} л и {meta['full_total_fines']:,} штрафов.

## Data audit

Характеристики ниже посчитаны по уникальным клиентам до исключений; распределения по аналитическим клиентам и отдельным автомобилям также сохранены в distribution_audit.csv.

{table(audit.loc[audit.population.eq('all_clients')])}

{table(correlations.loc[correlations.population.eq('analytical_clients')])}

Auto_price — медианная записанная стоимость автомобилей клиента, только аудит, не контроль модели и не доход. Исторические штрафы сохранены на исходном уровне клиент–автомобиль в vehicle_attributes.csv; неоднозначные суммы по нескольким машинам не строились и в модели не включались. Идентификаторы всех машин и дата подписки сохранены в аналитической панели.

### Несколько автомобилей и качество измерения

У **{meta['multi_car_clients']:,} клиентов ({meta['multi_car_share']:.2%})** несколько машин; в аналитической выборке таких {meta['analytical_multi_car_clients']:,}. Как в предыдущем анализе, engine_volume и horsepower — отдельные медианы известных значений автомобилей клиента, а не первая произвольная машина. Такие медианы могут описывать разные машины или синтетический промежуточный автомобиль. Топливные покупки привязаны к клиенту, поэтому нельзя уверенно назначить их конкретному двигателю. У {meta['clients_with_some_missing_car_features']} клиентов есть хотя бы одна машина с отсутствующими характеристиками: медиана использует только известные значения. Все назначения перепроверены по очищенной таблице машин.

Сохранены {meta['vehicle_rows']:,} строк клиент–автомобиль. Не распознаны характеристики в {meta['missing_parsed_vehicle_rows']} строках; {meta['invalid_parsed_vehicle_rows']} распознанных строк нарушали прежнее правило объём ≥0, мощность >0. Отдельные флаги: нулевой объём — {meta['zero_engine_vehicle_rows']} строк, объём >8 л — {meta['engine_over_8_vehicle_rows']}, мощность >1000 л.с. — {meta['horsepower_over_1000_vehicle_rows']}. Это сигналы для проверки записи, не автоматическое основание удалить клиента. Нулевой объём может означать электрический автомобиль; без типа силовой установки это не подтверждено. Экстремальные положительные значения не объявляются технически невозможными только по величине. Основной очищенный вариант оставлен без новой обрезки, winsorization и восстановления предполагаемых правильных значений. Список — vehicle_values_for_review.csv.

В текущих исходниках особенно подозрительны записи FORD 2010 «16.0 (100.00 л.с.)», DAEWOO 2007 «8.5 (84.00 л.с.)», а также SUZUKI с 1074/1728 л.с. и LAND ROVER с 1904 л.с. Они похожи на ошибки ввода или единиц, но корректные значения не восстановлены по догадке. Следовательно, «очищенный датасет» здесь означает выполненную документированную очистку операций и дубликатов, а не гарантию отсутствия всех ошибок автомобильных характеристик. Эти ограничения существенны для новой гипотезы и явно сохраняются в Stage 1.

### Распределения исходов в аналитической выборке

{table(audit.loc[audit.population.eq('analytical_client_months')])}

Доля нулей и асимметрия fines_count показаны явно; OLS оставлен как сопоставимый быстрый диагностический тест. Count models на этом этапе не перебирались.

## Спецификация и размер эффекта

OLS: Outcome_it = const + PriceShock_rt + VehicleZ_i + PriceShock_rt × VehicleZ_i + RegionFE + MonthFE. EngineVolume и Horsepower оцениваются отдельно, не как два независимых механизма. Стандартизация один раз по уникальным клиентам общей аналитической выборки, SD с ddof=0; параметры в standardization.csv. Region × Month FE не используются. Ошибки кластеризованы по региону с finite-sample correction и t(15), 95% ДИ. **16 кластеров — небольшое число: обычный cluster-robust inference может быть неточным.**

После вычитания RegionFE и MonthFE SD PriceShock = {meta['price_shock_residual_sd_after_FE']:.6g}. Это остаточная вариация, идентифицирующая общий ценовой коэффициент. В отличие от прежней агрегированной регрессии, здесь добавлены FE, веса задаются клиентами и исключены отсутствующие характеристики; общий коэффициент цены не обязан воспроизвести −590.126. FE не делают оценку причинной и не исключают неоднородные временные тренды типов автомобилей.

{chr(10).join(descriptions)}

Ниже предсказанное изменение исхода при +10 п.п. PriceShock на −1 SD, среднем и +1 SD характеристики (FE зафиксированы). ДИ используют ковариацию основного наклона и interaction. Это контрасты модели, а не утверждение, что в каждом регионе наблюдалось именно такое изменение цены.

{table(magnitudes.drop(columns='percent_change_in_geometric_1_plus_liters'))}

Для log_fuel контраст относится к log(1+литры). 100×(exp(контраст)−1), сохранённый в CSV, относится к геометрическому масштабу 1+литры; его нельзя автоматически назвать процентным изменением среднего числа литров.

## Только две диагностические визуализации

Границы терцилей по характеристике до моделей: {meta['tertile_q1']:g} и {meta['tertile_q2']:g} л. Одинаковые значения двигателя остаются вместе; поэтому размеры групп могут отличаться. Группы фиксированы во времени и нужны только для визуализации.

{table(groups)}

![Engine volume — fuel](01_engine_volume_fuel_response.png)

![Engine volume — fines](02_engine_volume_fines_response.png)

Точки показывают ненормированные региональные средние; их вид не заменяет continuous interaction с FE. Наблюдаемые покупки топлива не равны пробегу; 25,3% штрафов в предыдущем анализе относятся к месяцам без покупки топлива в сервисе. Штрафная активность не является чистым измерением стиля вождения или страхового риска. Все p-value номинальные; основная переменная engine_volume, horsepower — альтернативный коррелированный proxy.
"""
    (OUT / "diagnostic_summary.md").write_text(text, encoding="utf-8")
    return verdict


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT / "analysis_plan.json").exists(), "Read and freeze the Stage 1 design before models"
    print("Verifying cleaned inputs and auditing vehicle assignment...", flush=True)
    sample, audit, correlations, groups, meta = load_data()
    # Persist and display audit before fitting any model.
    dump("data_audit.json", meta)
    print(audit.loc[audit.population.isin(['all_clients', 'analytical_client_months'])].to_string(index=False), flush=True)
    print(correlations.to_string(index=False), flush=True)
    print("Fitting six predeclared Stage 1 models...", flush=True)
    results, magnitudes, checks, price_sd = fit_models(sample)
    meta['price_shock_residual_sd_after_FE'] = price_sd
    draw_plots(sample, groups, results)
    verdict = write_summary(results, magnitudes, audit, correlations, groups, meta)
    manifest = json.loads((OUT / "previous_analyses_manifest.json").read_text())
    current = {str(p.relative_to(ROOT)): digest(p) for d in ['archive_monthly_risk_clustering', 'fines_per_liter_analysis', 'usage_composition_analysis']
               for p in (ROOT / d).rglob('*') if p.is_file()}
    assert current == manifest, "Previous analyses changed"
    assert len(results) == 6 and results.N.nunique() == 1 and results.n_regions.eq(16).all()
    assert len(list(OUT.glob('*.png'))) == 2
    meta['diagnostic_code_sha256'] = digest(Path(__file__))
    dump('metadata.json', meta)
    dump('validation.json', dict(status='PASS', cleaned_source_hashes=len(meta['sources']),
        previous_files_unchanged=len(manifest), independent_inference_checks=checks,
        old_monthly_totals='PASS', vehicle_assignment_reconciliation='PASS', price_shock_formula='PASS',
        common_complete_case_sample='PASS', exactly_two_png_figures=True))
    print(results.to_string(index=False), flush=True)
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
