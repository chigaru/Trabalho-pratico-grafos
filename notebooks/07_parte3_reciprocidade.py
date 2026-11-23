import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

with open(os.path.join(RESULTS_DIR, "parte3_reciprocidade.json"), "r", encoding="utf-8") as f:
    res = json.load(f)

print(json.dumps(res, indent=2, ensure_ascii=False))

rg = res["reciprocidade_global"]
print(f"Pares recíprocos: {rg['pares_reciprocos']:,}")
print(f"Fração de arcos em laços bidirecionais: {rg['fracao_pct']:.2f}%")
print(f"Validação interna: arcos − pares = {res['validacao_interna']['arcos_unicos_menos_pares']:,}")
print(f"  Esperado: {res['validacao_interna']['m_undirected_esperado']:,}")
print(f"  Status: {'PASSOU ✓' if res['validacao_interna']['fechou'] else 'FALHOU ✗'}")

rb = res["baseline_aleatorio"]
print(f"Densidade do dígrafo: ρ = {rb['densidade_digrafo']:.8f}")
print(f"Reciprocidade esperada em dígrafo aleatório: {rb['reciprocidade_esperada_aleatoria']:.8f}")
print(f"Razão observado/esperado: {rb['razao_observado_esperado']:.2f}×")
print()
print("Interpretação: a reciprocidade observada (≈21%) é cerca de 100× maior")
print("que a esperada em um dígrafo aleatório com mesma densidade (~0.21%),")
print("indicando forte tendência à mutualidade nas conexões do Google+.")

fig_path = os.path.join(FIGURES_DIR, "parte3_reciprocidade_graus_dirigidos.png")
if os.path.exists(fig_path):
    from IPython.display import Image, display
    display(Image(filename=fig_path))
else:
    print(f"Figura não encontrada: {fig_path}")

gd = res["graus_dirigidos"]
print(f"In-degree médio: {gd['in_degree_medio']:.2f}")
print(f"Out-degree médio: {gd['out_degree_medio']:.2f}")
print(f"In-degree máximo: {gd['in_degree_max']:,}")
print(f"Out-degree máximo: {gd['out_degree_max']:,}")
print(f"Correlação in vs out: r = {gd['correlacao_in_out']:.4f}")

rn = res["reciprocidade_por_no"]
print(f"Nós com reciprocidade total: {rn['nos_reciprocidade_total']:,} ({rn['frac_reciprocidade_total_pct']:.2f}%)")
print(f"Nós sem nenhum arco recíproco: {rn['nos_sem_reciprocidade']:,} ({rn['frac_sem_reciprocidade_pct']:.2f}%)")

df_bands = pd.read_csv(os.path.join(RESULTS_DIR, "parte3_reciprocidade_por_faixa.csv"))
print(df_bands.to_string(index=False))

fig_path2 = os.path.join(FIGURES_DIR, "parte3_reciprocidade_por_faixa.png")
if os.path.exists(fig_path2):
    from IPython.display import Image, display
    display(Image(filename=fig_path2))

scc = res["scc"]
print(f"Total de SCCs: {scc['total_sccs']:,}")
print(f"Maior SCC: {scc['maior_scc_nos']:,} nós ({scc['maior_scc_frac_pct']:.2f}%)")
print(f"Reciprocidade na maior SCC: {scc['frac_reciprocidade_na_scc_pct']:.2f}% dos arcos recíprocos de saída")
print()
print("A maior SCC concentra a vasta maioria da reciprocidade.")
print("Isto é natural: a SCC exige conectividade bidirecional entre seus membros,")
print("logo pares recíprocos tendem a residir dentro dela.")
