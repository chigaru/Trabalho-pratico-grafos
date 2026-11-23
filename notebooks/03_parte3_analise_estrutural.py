import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from IPython.display import Image, display
except ImportError:
    class Image:
        def __init__(self, filename):
            self.filename = filename
        def __repr__(self):
            return f"[Image: {self.filename}]"
    display = print

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from src.graph_loader import load_csr_graph

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

print("Ambiente configurado com sucesso para a Parte III!")

SW_PATH = os.path.join(RESULTS_DIR, "parte3_small_world.json")
with open(SW_PATH, "r", encoding="utf-8") as f:
    sw_data = json.load(f)

sw_rows = [
    {
        "Métrica": "Comprimento Médio dos Caminhos (APL, L)",
        "Grafo Real G (ego-Gplus)": f"{sw_data['grafo_real']['L_G']:.4f}",
        "ER Teórico": f"{sw_data['teorico']['L_rand']:.4f}",
        "ER Empírico (G(n, m) Numba)": f"{sw_data['empirico']['L_rand']:.4f}",
        "Razão (G / ER Empírico)": f"{sw_data['empirico']['ratio_L']:.4f} (~1x)"
    },
    {
        "Métrica": "Coeficiente de Clusterização Médio (C)",
        "Grafo Real G (ego-Gplus)": f"{sw_data['grafo_real']['C_G']:.6f}",
        "ER Teórico": f"{sw_data['teorico']['C_rand']:.6f}",
        "ER Empírico (G(n, m) Numba)": f"{sw_data['empirico']['C_rand']:.6f}",
        "Razão (G / ER Empírico)": f"{sw_data['empirico']['ratio_C']:.2f}x (>> 1)"
    },
    {
        "Métrica": "Índice de Mundo Pequeno (sigma)",
        "Grafo Real G (ego-Gplus)": "—",
        "ER Teórico": f"{sw_data['teorico']['sigma']:.2f}",
        "ER Empírico (G(n, m) Numba)": f"{sw_data['empirico']['sigma']:.2f}",
        "Razão (G / ER Empírico)": f"sigma = {sw_data['empirico']['sigma']:.2f} >> 1"
    }
]
df_sw = pd.DataFrame(sw_rows)
display(df_sw)

display(Image(filename=os.path.join(FIGURES_DIR, "parte3_small_world_comparacao.png")))

PL_PATH = os.path.join(RESULTS_DIR, "parte3_power_law.json")
with open(PL_PATH, "r", encoding="utf-8") as f:
    pl_data = json.load(f)

opt_c = pl_data["otimo_clauset"]
ref_1500 = pl_data["referencia_1500"]

tail_rows = [
    {
        "Limiar de Cauda (k_min)": f"k_min = {opt_c['k_min']} (Mínimo KS Clauset)",
        "Nós na Cauda": f"{opt_c['n_tail']:,} ({opt_c['fraction_tail_pct']:.2f}%)",
        "Power Law (gamma, KS)": f"gamma = {opt_c['power_law']['gamma']:.4f} (KS = {opt_c['power_law']['ks']:.4f})",
        "Log-Normal (mu, sigma, KS)": f"mu = {opt_c['log_normal']['mu']:.2f}, s = {opt_c['log_normal']['sigma']:.2f} (KS = {opt_c['log_normal']['ks']:.4f})",
        "Exponencial (scale, KS)": f"scale = {opt_c['exponencial']['scale']:.2f} (KS = {opt_c['exponencial']['ks']:.4f})",
        "AIC (PL vs LN)": f"PL: {opt_c['power_law']['aic']:.0f} | LN: {opt_c['log_normal']['aic']:.0f} (Delta = {opt_c['power_law']['aic'] - opt_c['log_normal']['aic']:.1f})"
    },
    {
        "Limiar de Cauda (k_min)": f"k_min = {ref_1500['k_min']} (Regime Típico de Cauda)",
        "Nós na Cauda": f"{ref_1500['n_tail']:,} ({ref_1500['fraction_tail_pct']:.2f}%)",
        "Power Law (gamma, KS)": f"gamma = {ref_1500['power_law']['gamma']:.4f} (KS = {ref_1500['power_law']['ks']:.4f})",
        "Log-Normal (mu, sigma, KS)": f"mu = {ref_1500['log_normal']['mu']:.2f}, s = {ref_1500['log_normal']['sigma']:.2f} (KS = {ref_1500['log_normal']['ks']:.4f})",
        "Exponencial (scale, KS)": f"scale = {ref_1500['exponencial']['scale']:.2f} (KS = {ref_1500['exponencial']['ks']:.4f})",
        "AIC (PL vs LN)": f"PL: {ref_1500['power_law']['aic']:.0f} | LN: {ref_1500['log_normal']['aic']:.0f} (Delta = {ref_1500['power_law']['aic'] - ref_1500['log_normal']['aic']:.1f})"
    }
]
df_tail = pd.DataFrame(tail_rows)
display(df_tail)

display(Image(filename=os.path.join(FIGURES_DIR, "parte3_power_law_ks_scan.png")))
display(Image(filename=os.path.join(FIGURES_DIR, "parte3_distribuicao_graus_ajuste.png")))

TABELA_COMP_PATH = os.path.join(RESULTS_DIR, "parte3_tabela_comparativa.csv")
df_comp = pd.read_csv(TABELA_COMP_PATH)
display(df_comp)

display(Image(filename=os.path.join(FIGURES_DIR, "parte3_robustez_boxplots.png")))

RESUMO_PATH = os.path.join(RESULTS_DIR, "parte3_resumo.md")
with open(RESUMO_PATH, "r", encoding="utf-8") as f:
    resumo_texto = f.read()

print("=== RASCUNHOS FINAIS DAS 3 PERGUNTAS DA PARTE III ===")
for bloco in resumo_texto.split("### Rascunho da Resposta"):
    bloco_strip = bloco.strip()
    if bloco_strip.startswith("(Pergunta"):
        partes_traco = bloco_strip.split("---")
        corpo_pergunta = partes_traco[0].strip()
        linhas = [l.strip() for l in corpo_pergunta.split("\n") if l.strip()]
        header = linhas[0]
        texto = " ".join(linhas[1:])
        print(f"\n{header}\n{texto}\n")
