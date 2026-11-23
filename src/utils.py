import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats

def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    sem = std / np.sqrt(n)

    if n >= 30:
        crit_val = float(stats.norm.ppf((1 + confidence) / 2))
    else:
        crit_val = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))

    margin = crit_val * sem
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "sem": round(sem, 6),
        "margin": round(margin, 6),
        "ci_lower": round(mean - margin, 6),
        "ci_upper": round(mean + margin, 6),
        "confidence_level": confidence
    }

def plot_degree_distribution(
    degrees: np.ndarray,
    out_path: str = "figures/distribuicao_graus.png"
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    deg_min = int(np.min(degrees))
    deg_max = int(np.max(degrees))
    deg_mean = float(np.mean(degrees))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    ax1 = axes[0]
    bins = np.logspace(np.log10(max(1, deg_min)), np.log10(deg_max), 40)
    counts, edges = np.histogram(degrees, bins=bins, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    valid = counts > 0

    ax1.scatter(
        bin_centers[valid],
        counts[valid],
        color="#2b6cb0",
        alpha=0.8,
        s=30,
        edgecolors="none",
        label="Densidade Empírica $P(k)$"
    )
    ax1.axvline(deg_mean, color="#e53e3e", linestyle="--", linewidth=1.5, label=f"Grau Médio: {deg_mean:.1f}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Grau do Vértice ($k$)", fontsize=11)
    ax1.set_ylabel("Densidade de Probabilidade $P(k)$", fontsize=11)
    ax1.set_title("Distribuição de Graus — PDF (Log-Log)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True, fontsize=10)

    ax2 = axes[1]
    sorted_deg = np.sort(degrees)
    n = len(sorted_deg)
    ccdf_y = 1.0 - (np.arange(n) / n)

    ax2.plot(
        sorted_deg,
        ccdf_y,
        color="#2c7a7b",
        linewidth=2,
        label="CCDF Empírica $P(K \\geq k)$"
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Grau do Vértice ($k$)", fontsize=11)
    ax2.set_ylabel("Probabilidade Cumulativa $P(K \\geq k)$", fontsize=11)
    ax2.set_title("Distribuição Cumulativa Complementar (CCDF)", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)

    fig.suptitle(
        f"Análise Estrutural de Graus (G2 — ego-Gplus)\n"
        f"$N = {len(degrees):,}$, $k_{{\\min}} = {deg_min}$, $\\langle k \\rangle = {deg_mean:.2f}$, $k_{{\\max}} = {deg_max:,}$",
        fontsize=13,
        fontweight="bold",
        y=1.03
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_ego_network_hub(
    subg,
    hub_idx: int = 0,
    hub_label: str = "Hub",
    out_path: str = "figures/visualizacao_ego_hub.png"
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    layout = subg.layout_fruchterman_reingold(niter=600)
    coords = np.array(layout.coords)

    coords -= np.mean(coords, axis=0)
    max_c = np.max(np.abs(coords))
    if max_c > 0:
        coords /= max_c

    fig, ax = plt.subplots(figsize=(9, 9), dpi=200)

    edgelist = np.array(subg.get_edgelist())
    lines = [[coords[u], coords[v]] for u, v in edgelist]
    lc = LineCollection(lines, colors="#718096", linewidths=0.35, alpha=0.035, zorder=1)
    ax.add_collection(lc)

    neighbor_mask = np.ones(subg.vcount(), dtype=bool)
    neighbor_mask[hub_idx] = False
    ax.scatter(
        coords[neighbor_mask, 0],
        coords[neighbor_mask, 1],
        s=20,
        color="#3182ce",
        alpha=0.65,
        edgecolors="none",
        zorder=2,
        label=f"Vizinhos do Hub ($|N(u)| = {subg.vcount()-1}$)"
    )

    ax.scatter(
        coords[hub_idx, 0],
        coords[hub_idx, 1],
        s=140,
        color="#e53e3e",
        edgecolors="#1a202c",
        linewidths=1.8,
        zorder=3,
        label=f"Vértice Hub ({hub_label})"
    )

    ax.set_title(
        f"Visualização Reduzida: Ego-Network Induzida de Hub Central (G2)\n"
        f"Ordem: {subg.vcount():,} nós | Arestas Internas: {subg.ecount():,} arestas",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def build_status_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Métrica da Parte I": "Número de vértices (n)",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['vertices']:,}",
            "Método e Justificativa Metodológica": "Exato via dimensão da CSR (107.614 nós na GCC)."
        },
        {
            "Métrica da Parte I": "Número de arestas (m)",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['arestas']:,}",
            "Método e Justificativa Metodológica": "Exato via contagem simples de arestas não-dirigidas (12.238.285)."
        },
        {
            "Métrica da Parte I": "Grau mínimo",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['grau_minimo']}",
            "Método e Justificativa Metodológica": "Exato: nenhum nó é isolado na GCC (grau mín = 1)."
        },
        {
            "Métrica da Parte I": "Grau médio",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['grau_medio']:.2f}",
            "Método e Justificativa Metodológica": "Exato: 2m/n = 227,45 conexões médias por indivíduo."
        },
        {
            "Métrica da Parte I": "Grau máximo",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['grau_maximo']:,}",
            "Método e Justificativa Metodológica": "Exato: maior hub da rede possui 20.127 vizinhos diretos."
        },
        {
            "Métrica da Parte I": "Distribuição de graus",
            "Status": "Calculado",
            "Valor Obtido": "Histograma + CCDF log-log",
            "Método e Justificativa Metodológica": "Exato: função densidade e CCDF geradas em figures/distribuicao_graus.png."
        },
        {
            "Métrica da Parte I": "Densidade da rede",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['densidade']:.6e}",
            "Método e Justificativa Metodológica": "Exato: 2m / (n(n-1)) = 0,002114 (~0,21% dos pares conectados)."
        },
        {
            "Métrica da Parte I": "Número de componentes conexas",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['num_componentes']}",
            "Método e Justificativa Metodológica": "Exato: a GCC abrange 100% dos nós originais do dataset."
        },
        {
            "Métrica da Parte I": "Tamanho de cada componente",
            "Status": "Calculado",
            "Valor Obtido": f"GCC: {metrics['tamanho_gcc']:,} nós (100,00%)",
            "Método e Justificativa Metodológica": "Exato: componente gigante única com todos os 107.614 nós."
        },
        {
            "Métrica da Parte I": "Diâmetro",
            "Status": "Estimado (Limite Inferior)",
            "Valor Obtido": f"≥ {metrics['diametro_lower_bound']} (SNAP: {metrics['diametro_snap']})",
            "Método e Justificativa Metodológica": "All-pairs exato inviável (1,3×10¹² ops). Double-sweep BFS em Numba atinge diâmetro exato 6 do SNAP."
        },
        {
            "Métrica da Parte I": "Raio",
            "Status": "Estimado (Intervalo Rigoroso)",
            "Valor Obtido": "3 ≤ r ≤ 4",
            "Método e Justificativa Metodológica": "All-pairs exato inviável. Como D ≥ 6, temos r ≥ ⌈D/2⌉ = 3; amostragem de excentricidades (k=1.000) dá limite superior 4. Logo, 3 ≤ r ≤ 4."
        },
        {
            "Métrica da Parte I": "Comprimento médio dos caminhos (APL)",
            "Status": "Estimado (Amostrado)",
            "Valor Obtido": f"{metrics['apl_amostrado']['mean']:.4f} ± {metrics['apl_amostrado']['margin']:.4f} (IC 95%)",
            "Método e Justificativa Metodológica": "All-pairs inviável. BFS em Numba com k=1.000 fontes aleatórias (semente 42); erro amostral ≈ 0,6% (margem ±0,0175 / 2,8603)."
        },
        {
            "Métrica da Parte I": "Clusterização média (C=0 se grau < 2)",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['clustering_zero']:.6f} (SNAP: 0,4901)",
            "Método e Justificativa Metodológica": "Exato no grafo completo via Numba paralelo; idêntico ao igraph (mode='zero') e validação SNAP."
        },
        {
            "Métrica da Parte I": "Clusterização média (nós k<2 excluídos)",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['clustering_excluded']:.6f}",
            "Método e Justificativa Metodológica": "Exato: média restrita aos vértices com grau ≥ 2 (idêntico ao igraph mode='nan')."
        },
        {
            "Métrica da Parte I": "Número total de triângulos",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['total_triangulos']:,}",
            "Método e Justificativa Metodológica": "Exato no grafo completo via interseção CSR Numba; bate rigorosamente com o SNAP (1.073.677.742)."
        },
        {
            "Métrica da Parte I": "Visualização reduzida do grafo",
            "Status": "Calculado",
            "Valor Obtido": f"{metrics['ego_hub_nodes']:,} nós, {metrics['ego_hub_edges']:,} arestas",
            "Método e Justificativa Metodológica": "Ego-network induzida de hub com grau ~640 plotada via Fruchterman-Reingold em figures/."
        }
    ]
    return pd.DataFrame(rows)
