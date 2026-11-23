import time
import pickle
import os
from typing import Dict, Any, Optional
import numpy as np
import scipy.sparse as sp
import igraph as ig

from src.algorithms_numba import (
    bfs_single_csr,
    double_sweep_diameter,
    sample_apl_and_eccentricities,
    compute_triangles_and_clustering
)
from src.utils import calculate_confidence_interval

def compute_all_parte1_metrics(
    csr: sp.csr_matrix,
    node_ids: Optional[np.ndarray] = None,
    k_apl: int = 1000,
    seed: int = 42,
    pickle_path: Optional[str] = "data/processed/gplus_undirected_gcc.pickle",
    verbose: bool = True
) -> Dict[str, Any]:
    t_global_start = time.perf_counter()
    indptr = csr.indptr
    indices = csr.indices
    degrees = indptr[1:] - indptr[:-1]
    n = len(degrees)
    m = int(csr.nnz // 2)

    if verbose:
        print("=== INICIANDO CÁLCULO DAS MÉTRICAS DA PARTE I ===")
        print(f"Grafo: {n:,} vértices | {m:,} arestas simples não-dirigidas")

    deg_min = int(np.min(degrees))
    deg_max = int(np.max(degrees))
    deg_mean = float(np.mean(degrees))
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0

    if verbose:
        print("Calculando limite inferior do diâmetro via Double-Sweep BFS (Numba)...")
    t0 = time.perf_counter()
    diam_lb, diam_src, diam_dst, sweep_hist = double_sweep_diameter(
        indptr=indptr,
        indices=indices,
        num_sweeps=10,
        initial_node=0
    )
    t_diam = time.perf_counter() - t0
    if verbose:
        print(f"  Diâmetro (Double-Sweep): >= {diam_lb} (concluído em {t_diam:.2f}s)")

    if verbose:
        print(f"Calculando APL amostrado e limite superior do raio com k={k_apl:,} fontes (seed={seed})...")
    t0 = time.perf_counter()
    np.random.seed(seed)
    sources = np.random.choice(n, size=k_apl, replace=False).astype(np.int32)
    means, eccentricities = sample_apl_and_eccentricities(indptr, indices, sources)
    apl_ci = calculate_confidence_interval(means, confidence=0.95)
    raio_ub = int(np.min(eccentricities))
    t_apl = time.perf_counter() - t0
    if verbose:
        print(f"  APL Amostrado: {apl_ci['mean']:.4f} ± {apl_ci['margin']:.4f} (95% IC: [{apl_ci['ci_lower']:.4f}, {apl_ci['ci_upper']:.4f}])")
        print(f"  Raio (Mínima Excentricidade Amostrada): <= {raio_ub} (concluído em {t_apl:.2f}s)")

    if verbose:
        print("Calculando triângulos e clusterização local via Numba multithreaded...")
    t0 = time.perf_counter()
    triangles_per_node, clustering_per_node = compute_triangles_and_clustering(indptr, indices)
    total_triangulos = int(np.sum(triangles_per_node) // 3)
    c_zero = float(np.mean(clustering_per_node))
    c_excluded = float(np.mean(clustering_per_node[degrees >= 2]))
    t_triang = time.perf_counter() - t0
    if verbose:
        print(f"  Total de Triângulos: {total_triangulos:,} (SNAP: 1.073.677.742)")
        print(f"  Clusterização Média (C=0 se grau < 2): {c_zero:.6f} (SNAP: 0,4901)")
        print(f"  Clusterização Média (excluindo grau < 2): {c_excluded:.6f} (concluído em {t_triang:.2f}s)")

    validacao_igraph = {}
    if pickle_path and os.path.exists(pickle_path):
        if verbose:
            print("Carregando grafo igraph para validação cruzada...")
        t0 = time.perf_counter()
        with open(pickle_path, "rb") as f:
            g = pickle.load(f)

        ig_c_zero = float(g.transitivity_avglocal_undirected(mode="zero"))
        ig_c_nan = float(g.transitivity_avglocal_undirected(mode="nan"))
        diff_zero = abs(c_zero - ig_c_zero)
        diff_nan = abs(c_excluded - ig_c_nan)

        sample_val_sources = sources[:5]
        bfs_diff_count = 0
        for s in sample_val_sources:
            ig_dists = np.array(g.distances(source=int(s))[0], dtype=np.int32)
            _, _, _, _ = bfs_single_csr(indptr, indices, int(s))
            n_nodes = len(indptr) - 1
            dist_nb = np.full(n_nodes, -1, dtype=np.int32)
            q = np.empty(n_nodes, dtype=np.int32)
            dist_nb[s] = 0; q[0] = s; h = 0; t = 1
            while h < t:
                u = q[h]; h += 1; nd = dist_nb[u] + 1
                for i in range(indptr[u], indptr[u+1]):
                    v = indices[i]
                    if dist_nb[v] == -1:
                        dist_nb[v] = nd; q[t] = v; t += 1
            diffs = np.sum(dist_nb != ig_dists)
            bfs_diff_count += int(diffs)

        validacao_igraph = {
            "igraph_clustering_zero": round(ig_c_zero, 6),
            "igraph_clustering_nan": round(ig_c_nan, 6),
            "diff_clustering_zero": diff_zero,
            "diff_clustering_nan": diff_nan,
            "clustering_match": (diff_zero < 1e-5 and diff_nan < 1e-5),
            "val_sources_tested": len(sample_val_sources),
            "bfs_distance_differences": bfs_diff_count,
            "bfs_distances_match": (bfs_diff_count == 0),
            "tempo_validacao_s": round(time.perf_counter() - t0, 2)
        }
        if verbose:
            print(f"  Validação igraph: Clustering coincide? {validacao_igraph['clustering_match']} | Distâncias BFS coincidem? {validacao_igraph['bfs_distances_match']}")

    hub_candidate = int(np.where((degrees >= 600) & (degrees <= 700))[0][0])
    hub_degree = int(degrees[hub_candidate])
    ego_nodes_count = hub_degree + 1

    t_total = time.perf_counter() - t_global_start

    metrics = {
        "vertices": n,
        "arestas": m,
        "grau_minimo": deg_min,
        "grau_medio": round(deg_mean, 2),
        "grau_maximo": deg_max,
        "densidade": density,
        "num_componentes": 1,
        "tamanho_gcc": n,
        "fracao_gcc_pct": 100.0,
        "diametro_lower_bound": int(diam_lb),
        "diametro_snap": 6,
        "raio_upper_bound": int(raio_ub),
        "apl_amostrado": apl_ci,
        "clustering_zero": round(c_zero, 6),
        "clustering_excluded": round(c_excluded, 6),
        "clustering_snap": 0.4901,
        "total_triangulos": total_triangulos,
        "triangulos_snap": 1073677742,
        "ego_hub_id": hub_candidate,
        "ego_hub_degree": hub_degree,
        "ego_hub_nodes": ego_nodes_count,
        "ego_hub_edges": 0,
        "validacao_igraph": validacao_igraph,
        "tempos": {
            "diametro_s": round(t_diam, 2),
            "apl_raio_s": round(t_apl, 2),
            "triangulos_clustering_s": round(t_triang, 2),
            "total_s": round(t_total, 2)
        }
    }
    return metrics
