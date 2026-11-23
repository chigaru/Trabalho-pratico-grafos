import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = "results/parte2_tempos_tabela_final.csv"
OUT = "figures/parte2_resumo_tempos_log.png"

df = pd.read_csv(CSV)

df = df.drop_duplicates(subset=["Algoritmo", "Tamanho (n)"], keep="last").copy()

df["familia"] = df["Algoritmo"].str.split(" \\(", n=1, regex=True).str[0]
df["rotulo"] = df.apply(lambda r: f"{r['Algoritmo']} · n={int(r['Tamanho (n)']):,}", axis=1)
df = df.sort_values("Tempo Médio (ms)").reset_index(drop=True)

familias = sorted(df["familia"].unique())
cmap = plt.get_cmap("tab10")
cores = {f: cmap(i % 10) for i, f in enumerate(familias)}

fig, ax = plt.subplots(figsize=(11, max(6, 0.27 * len(df) + 1.5)), dpi=150)
ax.barh(df["rotulo"], df["Tempo Médio (ms)"], xerr=df["Margem IC 95% (ms)"],
        color=[cores[f] for f in df["familia"]], edgecolor="#2d3748", linewidth=0.5,
        error_kw={"elinewidth": 1.0, "capsize": 2})
ax.set_xscale("log")
ax.set_xlim(left=df["Tempo Médio (ms)"].min() * 0.6)
ax.set_xlabel("Tempo médio de execução (ms, escala log) ± IC 95%")
ax.set_title("Tempos de execução dos algoritmos da Parte II\n(tamanhos de entrada diferentes: comparar dentro de cada algoritmo)",
             fontsize=11, fontweight="bold")
ax.tick_params(axis="y", labelsize=7)
ax.grid(True, axis="x", which="both", linestyle=":", alpha=0.5)
ax.legend(handles=[Patch(facecolor=cores[f], edgecolor="#2d3748", label=f) for f in familias],
          title="Algoritmo", fontsize=8, title_fontsize=9, loc="lower right")
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"Salvo: {OUT} ({len(df)} barras)")
