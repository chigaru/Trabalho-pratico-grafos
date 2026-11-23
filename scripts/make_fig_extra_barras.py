import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/parte3_extra_tabela_comparativa.csv")

model_map = {
    "Real": "Real",
    "ER": "ER",
    "BA": "BA",
    "WS_b0.01": "WS\nβ=0.01",
    "WS_b0.1": "WS\nβ=0.1",
    "WS_b0.5": "WS\nβ=0.5"
}

df["NomeExibicao"] = df["Modelo"].map(model_map)
df = df.dropna(subset=["NomeExibicao"])

model_names = df["NomeExibicao"].tolist()
L_vals = df["L (amostr.)"].astype(float).tolist()
C_vals = df["C médio"].astype(float).tolist()

colors_lc = ["#2b6cb0", "#e53e3e", "#2f855a", "#805ad5", "#d69e2e", "#dd6b20"]

fig, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

bars_l = ax_l.bar(model_names, L_vals, color=colors_lc, edgecolor="#1a202c", linewidth=1)
ax_l.set_ylabel("Distância média ($L$)", fontsize=11, fontweight="bold")
ax_l.set_title("Distância Média ($L$) por Modelo", fontsize=12, fontweight="bold")
ax_l.set_ylim(0, max(L_vals) * 1.15)
for bar in bars_l:
    h = bar.get_height()
    ax_l.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}",
              ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_l.grid(axis="y", linestyle=":", alpha=0.5)

bars_c = ax_c.bar(model_names, C_vals, color=colors_lc, edgecolor="#1a202c", linewidth=1)
ax_c.set_ylabel("Coeficiente de Clusterização ($C$)", fontsize=11, fontweight="bold")
ax_c.set_title("Clusterização Média ($C$) por Modelo (escala log)", fontsize=12, fontweight="bold")
ax_c.set_yscale("log")
max_c = max(C_vals)
ax_c.set_ylim(1e-3, 5.0)

for bar in bars_c:
    h = bar.get_height()
    if h > 0:
        ax_c.text(bar.get_x() + bar.get_width() / 2, h * 1.25, f"{h:.4f}",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax_c.grid(axis="y", linestyle=":", alpha=0.5)

fig.suptitle("Comparação Estrutural: Grafo Real vs. Modelos Teóricos (G2 — ego-Gplus)",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
fig_bars_path = os.path.join("figures", "parte3_extra_L_C_barras.png")
plt.savefig(fig_bars_path, bbox_inches="tight")
plt.close()
print("Salvo: figures/parte3_extra_L_C_barras.png")
