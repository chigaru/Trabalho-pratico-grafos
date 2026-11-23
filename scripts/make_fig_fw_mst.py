import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/parte2_tempos_tabela_final.csv")

df_fw = df[df["Algoritmo"].str.contains("Floyd", case=False, na=False)].copy()
df_fw = df_fw.sort_values("Tamanho (n)")

fw_n = df_fw["Tamanho (n)"].values
fw_y = df_fw["Tempo Médio (ms)"].values
fw_err = df_fw["Margem IC 95% (ms)"].values

c_fw = np.sum(fw_y * (fw_n ** 3)) / np.sum(fw_n ** 6)
y_pred = c_fw * (fw_n ** 3)
ss_res = np.sum((fw_y - y_pred) ** 2)
ss_tot = np.sum((fw_y - np.mean(fw_y)) ** 2)
r2_fw = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
ax.errorbar(fw_n, fw_y, yerr=fw_err, fmt="o", color="#2b6cb0", ecolor="#e53e3e",
            capsize=5, elinewidth=1.8, markersize=7, label="Tempo Observado (Média ± IC 95%)")

x_smooth = np.linspace(min(fw_n), max(fw_n), 200)
y_smooth = c_fw * (x_smooth ** 3)
ax.plot(x_smooth, y_smooth, "--", color="#2c7a7b", linewidth=2.0,
        label=f"Curva Teórica Ajustada $c \\cdot n^3$\n($c = {c_fw:.3e}$, $R^2 = {r2_fw:.4f}$)")

n_800_idx = np.where(fw_n == 800)[0]
if len(n_800_idx) > 0:
    idx800 = n_800_idx[0]
    ax.annotate(f"n=800: {fw_y[idx800]:.1f} ± {fw_err[idx800]:.2f} ms",
                xy=(fw_n[idx800], fw_y[idx800]),
                xytext=(fw_n[idx800] - 250, fw_y[idx800] - 15),
                arrowprops=dict(facecolor="#2b6cb0", shrink=0.08, width=1.2, headwidth=5),
                fontsize=9.5, fontweight="bold", color="#1a365d")

ax.set_xlabel("Ordem do Subgrafo ($n$ nós)", fontsize=11, fontweight="bold")
ax.set_ylabel("Tempo de Execução (ms)", fontsize=11, fontweight="bold")
ax.set_title("Floyd-Warshall: Tempo Observado × Complexidade Teórica $O(n^3)$", fontsize=12, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig("figures/parte2_floyd_warshall.png", bbox_inches="tight")
plt.close()
print("Salvo: figures/parte2_floyd_warshall.png")

df_prim = df[df["Algoritmo"].str.startswith("Prim (MST)")].sort_values("Tamanho (n)")
df_kruskal = df[df["Algoritmo"].str.startswith("Kruskal (MST)")].sort_values("Tamanho (n)")

mst_n = df_prim["Tamanho (n)"].values
prim_y = df_prim["Tempo Médio (ms)"].values
prim_err = df_prim["Margem IC 95% (ms)"].values

kruskal_y = df_kruskal["Tempo Médio (ms)"].values
kruskal_err = df_kruskal["Margem IC 95% (ms)"].values

fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=200)
ax.errorbar(mst_n, prim_y, yerr=prim_err, fmt="o-", color="#2b6cb0", ecolor="#2b6cb0",
            capsize=5, elinewidth=1.6, markersize=6, label="Prim: $O(m \\log n)$")
ax.errorbar(mst_n, kruskal_y, yerr=kruskal_err, fmt="s-", color="#38a169", ecolor="#38a169",
            capsize=5, elinewidth=1.6, markersize=6, label="Kruskal: $O(m \\cdot \\alpha(n))$")

ax.set_xlabel("Ordem do Subgrafo ($n$ nós)", fontsize=11, fontweight="bold")
ax.set_ylabel("Tempo de Execução (ms)", fontsize=11, fontweight="bold")
ax.set_title("Árvore Geradora Mínima: Prim vs Kruskal em Subgrafos Crescentes", fontsize=12, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig("figures/parte2_mst_prim_kruskal.png", bbox_inches="tight")
plt.close()
print("Salvo: figures/parte2_mst_prim_kruskal.png")
