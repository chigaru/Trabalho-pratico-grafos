import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

with open(os.path.join(RESULTS_DIR, "parte3_extra_modelos.json"), "r", encoding="utf-8") as f:
    res = json.load(f)

df = pd.read_csv(os.path.join(RESULTS_DIR, "parte3_extra_tabela_comparativa.csv"))
print(df.to_string(index=False))

val = res["validacao_igraph_n2000"]
for model, info in val.items():
    status = "✓" if info["match_m"] and info["match_C"] else "✗"
    print(f"{model}: m_manual={info['m_manual']:,} vs igraph={info['m_igraph']:,} ({status})")
    print(f"  C_manual={info['C_manual']:.6f} vs igraph={info['C_igraph']:.6f}")

fig_ccdf = os.path.join(FIGURES_DIR, "parte3_extra_ccdf_graus.png")
if os.path.exists(fig_ccdf):
    from IPython.display import Image, display
    display(Image(filename=fig_ccdf))
else:
    print(f"Figura não encontrada: {fig_ccdf}")

fig_bars = os.path.join(FIGURES_DIR, "parte3_extra_L_C_barras.png")
if os.path.exists(fig_bars):
    from IPython.display import Image, display
    display(Image(filename=fig_bars))
else:
    print(f"Figura não encontrada: {fig_bars}")

params = res["parametros"]
print(f"n = {params['n']:,}")
print(f"m_real = {params['m_real']:,}")
print(f"m_attach (BA) = {params['m_attach_BA']}")
print(f"k (WS) = {params['k_WS']}")
print(f"betas (WS) = {params['betas_WS']}")
print(f"seed = {params['seed']}")
print(f"k_APL = {params['k_APL']}")

for name in ["ER", "BA", "WS_b0.01", "WS_b0.1", "WS_b0.5"]:
    if name in res["modelos"]:
        info = res["modelos"][name]
        print(f"\n{name}:")
        print(f"  Geração: {info.get('t_gen_s', '?')}s")
        print(f"  APL: {info.get('t_apl_s', '?')}s")
        print(f"  Clustering: {info.get('t_clust_s', '?')}s")
        print(f"  Memória CSR: {info.get('mem_csr_mb', '?')} MB")
