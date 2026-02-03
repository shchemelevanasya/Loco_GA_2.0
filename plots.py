"""
plots.py
Визуализация результатов (Seaborn style), экспорт PNG / CSV.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io
from typing import List, Dict, Any
sns.set_theme(style="darkgrid")
FONT_SCALE = 1.1
sns.set_context("notebook", font_scale=FONT_SCALE)

def plot_evolution(history: Dict[str, Any], dpi=200, figsize=(10,5)):
    """
    Построение кривых: best, mean, median, ±std (тенями). Отмечаем улучшения.
    history: dict with keys best, mean, median, std (lists), weights optional
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    gens = np.arange(len(history["best"]))
    best = np.array(history["best"])
    mean = np.array(history["mean"])
    median = np.array(history["median"])
    std = np.array(history["std"])
    ax.plot(gens, best, label="Лучший", color="tab:green")
    ax.plot(gens, mean, label="Средний", color="tab:blue")
    ax.plot(gens, median, label="Медиана", color="tab:orange")
    ax.fill_between(gens, mean-std, mean+std, color="tab:blue", alpha=0.2, label="±std")
    # mark improvements in best
    improvements = np.where(np.concatenate([[True], best[1:] < best[:-1]]))[0] if np.mean(best)>0 else np.where(np.concatenate([[True], best[1:] > best[:-1]]))[0]
    for g in improvements:
        ax.scatter(g, best[g], color="red", s=10)
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Пригодность")
    ax.set_title("Кривая эволюции")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return fig, buf

def plot_weights_over_time(history: Dict[str,Any], dpi=200, figsize=(8,4)):
    # history["weights"] is list of dicts
    if not history.get("weights"):
        return None, None
    df = pd.DataFrame(history["weights"])
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    df.plot.area(ax=ax)
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Вес компонента")
    ax.set_title("Динамика весов (адаптивные)")
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return fig, buf

def save_fig_to_bytes(fig, fmt="png", dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi)
    buf.seek(0)
    return buf

def df_summary_table(population_history: List[Dict[str,Any]]):
    # expects history per generation metrics; convert to DataFrame
    return pd.DataFrame(population_history)