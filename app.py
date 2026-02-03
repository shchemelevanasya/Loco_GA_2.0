"""
app.py
Streamlit UI (русский) для запуска GA, загрузки данных, выбора операторов и экспорта результатов.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from io import BytesIO, StringIO
from loco_ga import GeneticAlgorithm, GAConfig, sample_data, Trip, Loco, LocoType, evaluate_individual, compute_fitness
import plots

st.set_page_config(page_title="GA Назначение локомотивов", layout="wide")

st.title("Генетический алгоритм назначения локомотивов")
st.markdown("Интерфейс на русском. Загрузите таблицы рейсов и локомотивов или используйте примерные данные.")

# Sidebar: data input and GA parameters
with st.sidebar:
    st.header("Данные")
    uploaded_trips = st.file_uploader("Загрузить CSV рейсов (columns: id,start_time,end_time,origin,destination,distance)", type=["csv"])
    uploaded_locos = st.file_uploader("Загрузить CSV локомотивов (columns: id,type_id,resource_remaining,home_depot)", type=["csv"])
    use_sample = st.button("Использовать примерные данные")
    st.markdown("---")
    st.header("Параметры GA")
    pop_size = st.number_input("Размер популяции", value=80, min_value=4)
    generations = st.number_input("Поколений", value=50, min_value=1)
    crossover_rate = st.slider("Вероятность кроссовера", 0.0, 1.0, 0.8)
    base_mut = st.slider("Базовая вероятность мутации", 0.0, 0.5, 0.05)
    elite = st.number_input("Элитизм (кол-во)", value=2, min_value=0)
    minimize = st.checkbox("Минимизировать целевую функцию (по умолчанию)", value=True)
    st.markdown("---")
    st.header("Операторы")
    # Select operators
    crossover_ops = st.multiselect("Кроссоверы", ["one_point","two_point","uniform","priority"], default=["one_point","two_point","uniform","priority"])
    mut_ops = st.multiselect("Мутации", ["swap_locos","replace_loco","range_shuffle"], default=["swap_locos","replace_loco","range_shuffle"])
    st.markdown("Базовые вероятности кроссоверов (не обязаны суммироваться — они нормализуются автоматически):")
    base_cross_probs = {}
    for op in ["one_point","two_point","uniform","priority"]:
        base_cross_probs[op] = st.number_input(f"p_{op}", value=0.25, min_value=0.0, max_value=1.0, step=0.01)
    st.markdown("Базовые вероятности мутаций:")
    base_mut_probs = {}
    for op in ["swap_locos","replace_loco","range_shuffle"]:
        base_mut_probs[op] = st.number_input(f"p_{op}", value=0.4 if op!="range_shuffle" else 0.2, min_value=0.0, max_value=1.0, step=0.01)

# Load or generate data
if use_sample or (uploaded_trips is None and uploaded_locos is None):
    trips, locos, loco_types = sample_data(n_trips=60, n_locos=12)
    st.success("Загружены примерные данные.")
else:
    # parse uploaded CSVs
    trips_df = pd.read_csv(uploaded_trips)
    locos_df = pd.read_csv(uploaded_locos)
    # minimal validation
    trips = [Trip(id=str(r['id']), start_time=float(r['start_time']), end_time=float(r['end_time']), origin=str(r.get('origin','')), destination=str(r.get('destination','')), distance=float(r.get('distance',0.0))) for idx,r in trips_df.iterrows()]
    loco_types_list = {}
    # For types: we try to infer from locos_df (if contains type columns)
    types_in_df = loco_types_list
    # For simplicity: create types mapping from unique type_id with default numeric parameters if not provided
    unique_types = locos_df['type_id'].unique()
    loco_types = {}
    for t in unique_types:
        loco_types[str(t)] = LocoType(id=str(t), power=float(3.0), avg_reposition_speed=float(50.0), resource_range=(6.0,12.0))
    locos = []
    for idx,row in locos_df.iterrows():
        locos.append(Loco(id=str(row['id']), type_id=str(row['type_id']), resource_remaining=float(row.get('resource_remaining',8.0)), home_depot=str(row.get('home_depot',''))))

st.write(f"Рейсов: {len(trips)}, локомотивов: {len(locos)}, типов: {len(loco_types)}")

# Run GA
config = GAConfig(
    population_size=int(pop_size),
    generations=int(generations),
    crossover_rate=float(crossover_rate),
    base_mutation_prob=float(base_mut),
    elite_size=int(elite),
    maximize=not minimize,
    allowed_crossover_ops=crossover_ops,
    allowed_mutation_ops=mut_ops,
    crossover_base_probs=base_cross_probs,
    mutation_base_probs=base_mut_probs,
    seed=42
)

run_button = st.button("Запустить GA")
if run_button:
    ga = GeneticAlgorithm(trips, locos, loco_types, config)
    with st.spinner("Запуск генетического алгоритма..."):
        ga.run()
    st.success(f"Генетический алгоритм завершён за {ga.total_time:.2f} с.")
    # best individual
    best = sorted(ga.population, key=lambda x: x.fitness, reverse=ga.config.maximize)[0]
    st.subheader("Лучшее решение")
    st.write(f"Пригодность: {best.fitness:.4f}")
    st.write("Компоненты:")
    st.json(best.components)
    st.write(f"Использовано локомотивов: {best.used_locos_count}")
    # show genome snippet
    df_assign = pd.DataFrame({"trip_id":[t.id for t in trips], "loco_idx": best.genome})
    st.dataframe(df_assign.head(50))
    # Evolution plot
    fig, buf = plots.plot_evolution(ga.history, dpi=200)
    st.subheader("Кривая эволюции")
    st.pyplot(fig)
    st.download_button("Скачать график эволюции (PNG)", buf, file_name="evolution.png", mime="image/png")
    # weights plot
    fig_w, buf_w = plots.plot_weights_over_time(ga.history, dpi=200)
    if fig_w is not None:
        st.subheader("Динамика весов фитнес-компонентов")
        st.pyplot(fig_w)
        st.download_button("Скачать веса (PNG)", buf_w, file_name="weights.png", mime="image/png")
    # history table and csv
    hist_df = pd.DataFrame({
        "best": ga.history["best"],
        "mean": ga.history["mean"],
        "median": ga.history["median"],
        "std": ga.history["std"],
        "time_sec": ga.history["time"]
    })
    st.subheader("Статистика по поколениям")
    st.dataframe(hist_df)
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать статистику (CSV)", csv, file_name="ga_history.csv", mime="text/csv")
    # Diagram assignments (simple timeline)
    st.subheader("Диаграмма назначений (временная шкала)")
    # prepare timeline plot simple
    import matplotlib.pyplot as plt
    fig2, ax2 = plt.subplots(figsize=(12,4), dpi=200)
    colors = plt.cm.get_cmap("tab20", len(locos))
    for i, trip in enumerate(trips):
        lidx = int(best.genome[i])
        ax2.barh(y=lidx, width=(trip.end_time - trip.start_time), left=trip.start_time, color=colors(lidx), alpha=0.8)
        ax2.text(trip.start_time + 0.01, lidx + 0.1, f"{trip.id}", fontsize=6)
    ax2.set_xlabel("Время (ч)")
    ax2.set_ylabel("Индекс локомотива")
    ax2.set_title("Временная диаграмма назначений")
    st.pyplot(fig2)
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=300)
    buf2.seek(0)
    st.download_button("Скачать диаграмму назначений (PNG)", buf2, file_name="assignment_timeline.png", mime="image/png")
    # Sensitivity (fast mode)
    st.subheader("Анализ чувствительности весов (быстрый режим)")
    if st.button("Запустить быстрый анализ чувствительности"):
        st.info("Запускаю быстрый анализ: немного изменяю веса и оцениваю отклик лучшего решения.")
        base_weights = ga.history["weights"][-1] if ga.history["weights"] else {}
        deltas = [-0.5, -0.25, 0.0, 0.25, 0.5]
        sens = []
        for k in base_weights.keys():
            for d in deltas:
                wmod = base_weights.copy()
                wmod[k] = max(0.0, wmod[k] * (1.0 + d))
                # renormalize
                s = sum(wmod.values())
                for kk in wmod: wmod[kk] /= max(s,1e-9)
                # compute score for best.ind using new weights manually
                vec = np.array([best.components.get("idle_time",0.0), best.components.get("empty_run_time",0.0), best.components.get("train_wait_time",0.0), best.components.get("loco_wait_time",0.0), best.used_locos_count])
                keys = ["idle_time","empty_run_time","train_wait_time","loco_wait_time","used_locos_count"]
                wvec = np.array([wmod[k] for k in keys])
                score = float(np.dot(vec, wvec))
                sens.append({"component":k,"delta":d,"score":score})
        sens_df = pd.DataFrame(sens)
        st.dataframe(sens_df)
        st.download_button("Скачать чувствительность (CSV)", sens_df.to_csv(index=False).encode('utf-8'), file_name="sensitivity.csv", mime="text/csv")