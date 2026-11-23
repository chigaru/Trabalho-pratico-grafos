import os
import sys
import json
import pickle
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sys.path.append(os.path.abspath("."))
from src.graph_loader import load_csr_graph

def plot_ego_legivel(
    subg,
    hub_sub_idx: int,
    hub_orig_id: int,
    out_path: str,
    seed: int = 42
):
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(seed)
    init_coords = rng.standard_normal((subg.vcount(), 2)).tolist()
    layout = subg.layout_fruchterman_reingold(niter=1500, seed=init_coords)
    coords = np.array(layout.coords)
    coords -= np.mean(coords, axis=0)
    max_c = np.max(np.abs(coords))
    if max_c > 0:
        coords /= max_c

    sub_degs = np.array(subg.degree())
    n_nodes = subg.vcount()
    n_edges = subg.ecount()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    edgelist = np.array(subg.get_edgelist())
    lines = [[coords[u], coords[v]] for u, v in edgelist]
    lc = LineCollection(lines, colors="#4a5568", linewidths=0.3, alpha=0.15, zorder=1)
    ax.add_collection(lc)

    neighbor_mask = np.ones(n_nodes, dtype=bool)
    neighbor_mask[hub_sub_idx] = False

    node_sizes = 20 + 80 * (sub_degs / np.max(sub_degs))

    sc = ax.scatter(
        coords[neighbor_mask, 0],
        coords[neighbor_mask, 1],
        c=sub_degs[neighbor_mask],
        s=node_sizes[neighbor_mask],
        cmap="viridis",
        alpha=0.85,
        edgecolors="none",
        zorder=2
    )

    ax.scatter(
        coords[hub_sub_idx, 0],
        coords[hub_sub_idx, 1],
        s=160,
        color="#e53e3e",
        edgecolors="#1a202c",
        linewidths=2.0,
        zorder=4,
        label=f"Nó Central #{hub_orig_id} ($k_{{sub}}={sub_degs[hub_sub_idx]}$)"
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Grau no Subgrafo ($k_{sub}$)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    dens_pct_str = f"{2*n_edges/(n_nodes*(n_nodes-1))*100:.2f}%"
    ax.set_title(
        f"Figura A: Ego-Network Legível (Nó Central #{hub_orig_id})\n"
        f"V = {n_nodes} nós, E = {n_edges:,} arestas internas (Densidade $\\rho$ = {dens_pct_str})",
        fontsize=11,
        fontweight="bold",
        pad=10
    )
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

def plot_ego_hub_denso(
    subg,
    hub_sub_idx: int,
    hub_orig_id: int,
    out_path: str,
    n_sample_edges: int = 4000,
    seed: int = 42
):
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(seed)
    init_coords = rng.standard_normal((subg.vcount(), 2)).tolist()
    layout = subg.layout_fruchterman_reingold(niter=1000, seed=init_coords)
    coords = np.array(layout.coords)
    coords -= np.mean(coords, axis=0)
    max_c = np.max(np.abs(coords))
    if max_c > 0:
        coords /= max_c

    sub_degs = np.array(subg.degree())
    n_nodes = subg.vcount()
    n_edges = subg.ecount()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    rng = np.random.default_rng(seed)
    edgelist = np.array(subg.get_edgelist())
    sample_idx = rng.choice(len(edgelist), size=min(n_sample_edges, len(edgelist)), replace=False)
    sampled_lines = [[coords[edgelist[i][0]], coords[edgelist[i][1]]] for i in sample_idx]

    lc = LineCollection(sampled_lines, colors="#2d3748", linewidths=0.25, alpha=0.05, zorder=1)
    ax.add_collection(lc)

    neighbor_mask = np.ones(n_nodes, dtype=bool)
    neighbor_mask[hub_sub_idx] = False

    node_sizes = 15 + 60 * (sub_degs / np.max(sub_degs))

    sc = ax.scatter(
        coords[neighbor_mask, 0],
        coords[neighbor_mask, 1],
        c=sub_degs[neighbor_mask],
        s=node_sizes[neighbor_mask],
        cmap="plasma",
        alpha=0.75,
        edgecolors="none",
        zorder=2
    )

    ax.scatter(
        coords[hub_sub_idx, 0],
        coords[hub_sub_idx, 1],
        s=180,
        color="#3182ce",
        edgecolors="#1a202c",
        linewidths=2.0,
        zorder=4,
        label=f"Super-Hub #{hub_orig_id} ($k={sub_degs[hub_sub_idx]}$)"
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Grau no Subgrafo ($k_{sub}$)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    dens = 2 * n_edges / (n_nodes * (n_nodes - 1))
    dens_pct_str = f"{dens * 100:.2f}%"
    ax.set_title(
        f"Figura B: Visão Geral do Núcleo Denso (Hub #{hub_orig_id})\n"
        f"V = {n_nodes} nós, E = {n_edges:,} arestas ({n_sample_edges:,} amostradas, $\\alpha=0.05$, Densidade $\\rho$ = {dens_pct_str})",
        fontsize=11,
        fontweight="bold",
        pad=10
    )
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

def main():
    print("=== GERANDO FIGURAS DA PARTE I (EGO-NETWORKS) ===")
    npz_path = "data/processed/gplus_undirected_gcc.npz"
    pickle_path = "data/processed/gplus_undirected_gcc.pickle"

    csr, node_ids = load_csr_graph(npz_path)
    degrees = csr.indptr[1:] - csr.indptr[:-1]

    with open(pickle_path, "rb") as f:
        g = pickle.load(f)
    g.vs["orig_id"] = list(range(g.vcount()))

    candidates = np.where((degrees >= 80) & (degrees <= 150))[0]
    rng = np.random.default_rng(42)
    node_a = int(rng.choice(candidates))
    deg_a = int(degrees[node_a])

    nbrs_a = g.neighbors(node_a)
    ego_nodes_a = [node_a] + nbrs_a
    sub_a = g.induced_subgraph(ego_nodes_a)
    n_a = sub_a.vcount()
    m_a = sub_a.ecount()
    dens_a = 2 * m_a / (n_a * (n_a - 1))

    print(f"Figura A escolhida:")
    print(f"  Nó Central: #{node_a} | Grau global: {deg_a}")
    print(f"  Nós no subgrafo: {n_a} | Arestas internas: {m_a:,} | Densidade: {dens_a:.4f} ({dens_a*100:.2f}%)")

    fig_a_path = "figures/visualizacao_ego_legivel.png"
    hub_idx_a = sub_a.vs["orig_id"].index(node_a)
    print(f"  Índice do nó central no subgrafo: {hub_idx_a} (grau no subgrafo = {sub_a.degree(hub_idx_a)})")
    plot_ego_legivel(sub_a, hub_sub_idx=hub_idx_a, hub_orig_id=node_a, out_path=fig_a_path)
    print(f"  Salvo: {fig_a_path}")

    node_b = 2
    deg_b = int(degrees[node_b])
    nbrs_b = g.neighbors(node_b)
    ego_nodes_b = [node_b] + nbrs_b
    sub_b = g.induced_subgraph(ego_nodes_b)
    n_b = sub_b.vcount()
    m_b = sub_b.ecount()
    dens_b = 2 * m_b / (n_b * (n_b - 1))

    print(f"\nFigura B:")
    print(f"  Nó Central: #{node_b} | Grau global: {deg_b}")
    print(f"  Nós no subgrafo: {n_b} | Arestas internas: {m_b:,} | Densidade: {dens_b:.4f} ({dens_b*100:.2f}%)")

    fig_b_path = "figures/visualizacao_ego_hub_denso.png"
    hub_idx_b = sub_b.vs["orig_id"].index(node_b)
    print(f"  Índice do nó central no subgrafo: {hub_idx_b} (grau no subgrafo = {sub_b.degree(hub_idx_b)})")
    plot_ego_hub_denso(sub_b, hub_sub_idx=hub_idx_b, hub_orig_id=node_b, out_path=fig_b_path, n_sample_edges=4000, seed=42)
    print(f"  Salvo: {fig_b_path}")

    info = {
        "figura_a_legivel": {
            "descricao": "Ego-network induzida legível com todas as arestas internas",
            "node_central_id": node_a,
            "grau_global": deg_a,
            "num_nos": n_a,
            "num_arestas": m_a,
            "densidade": round(dens_a, 6),
            "densidade_pct": round(dens_a * 100, 2),
            "arquivo": fig_a_path
        },
        "figura_b_denso": {
            "descricao": "Ego-network induzida do super-hub com amostragem de 4.000 arestas",
            "node_central_id": node_b,
            "grau_global": deg_b,
            "num_nos": n_b,
            "num_arestas_total": m_b,
            "arestas_amostradas": 4000,
            "densidade": round(dens_b, 6),
            "densidade_pct": round(dens_b * 100, 2),
            "arquivo": fig_b_path
        }
    }

    out_json = "results/parte1_ego_networks.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nDados registrados em {out_json} com sucesso!")

if __name__ == "__main__":
    main()
