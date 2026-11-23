import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import igraph as ig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sys.path.append(os.path.abspath("."))
from src.graph_loader import load_csr_graph
from src.utils import calculate_confidence_interval
from src.algorithms_numba import (
    generate_symmetric_weights_numba,
    bfs_distances_csr,
    dfs_iterative_csr,
    check_eulerian_numba,
    dijkstra_csr,
    bellman_ford_csr,
    floyd_warshall_csr,
    tarjan_scc_iterative,
    prim_mst_csr,
    kruskal_mst
)

@njit
def get_bfs_order(indptr, indices, source=0):
    N = len(indptr) - 1
    order = np.empty(N, dtype=np.int32)
    seen = np.zeros(N, dtype=np.bool_)
    queue = np.empty(N, dtype=np.int32)
    seen[source] = True
    queue[0] = source
    head = 0
    tail = 1
    while head < tail:
        u = queue[head]
        order[head] = u
        head += 1
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if not seen[v]:
                seen[v] = True
                queue[tail] = v
                tail += 1
    return order[:tail]

def fit_complexity_curve(x_data, y_data, theoretical_fn):
    f_x = np.array([theoretical_fn(x) for x in x_data], dtype=np.float64)
    y = np.array(y_data, dtype=np.float64)

    denom = np.sum(f_x ** 2)
    if denom > 0:
        c = np.sum(y * f_x) / denom
    else:
        c = 0.0

    y_pred = c * f_x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(c), float(r2), y_pred

def extract_subgraph_csr(visited_order, n, indptr, indices, weights):
    sub_nodes = visited_order[:n]
    node_set = set(sub_nodes)
    node_map = {orig: new for new, orig in enumerate(sub_nodes)}

    u_list = []
    v_list = []
    w_list = []

    for u in sub_nodes:
        new_u = node_map[u]
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if v in node_set:
                u_list.append(new_u)
                v_list.append(node_map[v])
                w_list.append(weights[i])

    sub_csr = csr_matrix((w_list, (u_list, v_list)), shape=(n, n))
    m_undir = sub_csr.nnz // 2
    return sub_csr.indptr, sub_csr.indices, sub_csr.data.astype(np.int32), m_undir

def main():
    t_start = time.perf_counter()
    print("=== PIPELINE DA PARTE II: ALGORITMOS E COMPLEXIDADE ===")

    print("\n1. Carregando dados do grafo...")
    csr_undir, _ = load_csr_graph("data/processed/gplus_undirected_gcc.npz")
    indptr = csr_undir.indptr
    indices = csr_undir.indices
    N = len(indptr) - 1
    M_undir = len(indices) // 2
    print(f"   Grafo não-dirigido: {N:,} vértices, {M_undir:,} arestas simples.")

    data_dir = np.load("data/processed/gplus_directed.npz")
    d_indptr = data_dir["indptr"]
    d_indices = data_dir["indices"]
    N_dir = len(d_indptr) - 1
    M_dir = len(d_indices)
    print(f"   Dígrafo original: {N_dir:,} vértices, {M_dir:,} arcos dirigidos.")

    print("\n2. Gerando pesos inteiros simétricos (1..10, seed 42)...")
    weights = generate_symmetric_weights_numba(indptr, indices, seed=42)
    print(f"   Pesos gerados com sucesso: min={weights.min()}, max={weights.max()}, shape={len(weights):,}")

    print("\n3. Gerando subgrafos aninhados via BFS a partir do nó 0 (semente fixa)...")
    t0 = time.perf_counter()
    bfs_order = get_bfs_order(indptr, indices, 0)
    print(f"   Ordem BFS completa gerada em {time.perf_counter() - t0:.2f}s ({len(bfs_order):,} nós).")

    subgraph_sizes = [100, 200, 400, 500, 600, 800, 1000, 2000, 3000, 5000]
    subgraphs_info = {}
    subgraphs_data = {}

    for n in subgraph_sizes:
        sub_indptr, sub_indices, sub_weights, sub_m = extract_subgraph_csr(
            bfs_order, n, indptr, indices, weights
        )
        density = (2.0 * sub_m) / (n * (n - 1)) if n > 1 else 0.0
        subgraphs_info[n] = {
            "n": n,
            "m": sub_m,
            "densidade": round(density, 6),
            "densidade_pct": round(density * 100, 2)
        }
        subgraphs_data[n] = {
            "indptr": sub_indptr,
            "indices": sub_indices,
            "weights": sub_weights,
            "m": sub_m
        }
        print(f"   Subgrafo n={n:4d}: m={sub_m:7,d} arestas | densidade={density*100:5.2f}%")

    with open("results/parte2_subgrafos.json", "w", encoding="utf-8") as f:
        json.dump(subgraphs_info, f, indent=2, ensure_ascii=False)

    print("\n4. Executando validação de correção prévia dos algoritmos...")
    val_report = {}

    is_eul, has_path, is_conn, odd_c, even_c = check_eulerian_numba(indptr, indices)
    val_report["eulerianidade"] = {
        "conexo": bool(is_conn),
        "nos_grau_impar": int(odd_c),
        "nos_grau_par": int(even_c),
        "e_euleriano": bool(is_eul),
        "tem_caminho_euleriano": bool(has_path),
        "conclusao": "Grafo não-euleriano: embora conexo, possui 55.624 vértices de grau ímpar (exigência: 0)."
    }
    print(f"   [Eulerianidade] Conexo={is_conn}, Nós ímpares={odd_c:,}, Nós pares={even_c:,} -> Euleriano={is_eul}")

    dist_bfs_0 = bfs_distances_csr(indptr, indices, 0)
    weights_one = np.ones(len(indices), dtype=np.int32)
    dist_dijk_one_0 = dijkstra_csr(indptr, indices, weights_one, 0)
    diff_bfs_dijk = int(np.max(np.abs(dist_bfs_0.astype(np.int64) - dist_dijk_one_0)))
    val_report["dijkstra_pesos_1_vs_bfs"] = {
        "diff_maxima": diff_bfs_dijk,
        "match": bool(diff_bfs_dijk == 0)
    }
    print(f"   [Dijkstra(w=1) == BFS] Divergência máxima: {diff_bfs_dijk} (Match perfeito: {diff_bfs_dijk == 0})")

    with open("data/processed/gplus_undirected_gcc.pickle", "rb") as f:
        g_ig = pickle.load(f)

    bfs_ig_diffs = []
    test_sources = [0, 42, 100]
    for s in test_sources:
        d_man = bfs_distances_csr(indptr, indices, s)
        d_ig = np.array(g_ig.distances(source=s)[0], dtype=np.int32)
        bfs_ig_diffs.append(int(np.max(np.abs(d_man - d_ig))))
    val_report["bfs_vs_igraph"] = {
        "fontes_testadas": test_sources,
        "max_diff_por_fonte": bfs_ig_diffs,
        "match": bool(all(d == 0 for d in bfs_ig_diffs))
    }
    print(f"   [BFS vs igraph] Fontes {test_sources}: divergências {bfs_ig_diffs}")

    sub200 = subgraphs_data[200]
    D_fw_200 = floyd_warshall_csr(sub200["indptr"], sub200["indices"], sub200["weights"])
    fw_vs_dijk_match = True
    for s in range(200):
        d_s = dijkstra_csr(sub200["indptr"], sub200["indices"], sub200["weights"], s)
        if not np.array_equal(d_s, D_fw_200[s]):
            fw_vs_dijk_match = False
            break

    d_bf_0, rounds_bf = bellman_ford_csr(sub200["indptr"], sub200["indices"], sub200["weights"], 0)
    d_dijk_0 = dijkstra_csr(sub200["indptr"], sub200["indices"], sub200["weights"], 0)
    bf_vs_dijk_match = bool(np.array_equal(d_bf_0, d_dijk_0))

    val_report["floyd_warshall_vs_dijkstra"] = {
        "subgrafo_n": 200,
        "match": fw_vs_dijk_match
    }
    val_report["bellman_ford_vs_dijkstra"] = {
        "subgrafo_n": 200,
        "rounds_convergencia": int(rounds_bf),
        "match": bf_vs_dijk_match
    }
    print(f"   [FW vs Dijkstra n=200] Match: {fw_vs_dijk_match}")
    print(f"   [BF vs Dijkstra n=200] Match: {bf_vs_dijk_match} (convergência em {rounds_bf} rodadas)")

    w_prim, _ = prim_mst_csr(sub200["indptr"], sub200["indices"], sub200["weights"])

    u_list, v_list, w_list = [], [], []
    for u in range(200):
        for i in range(sub200["indptr"][u], sub200["indptr"][u+1]):
            v = sub200["indices"][i]
            if u < v:
                u_list.append(u)
                v_list.append(v)
                w_list.append(sub200["weights"][i])
    u_arr = np.array(u_list, dtype=np.int32)
    v_arr = np.array(v_list, dtype=np.int32)
    w_arr = np.array(w_list, dtype=np.int32)

    w_kruskal, _ = kruskal_mst(200, u_arr, v_arr, w_arr)

    sub200_mat = csr_matrix((sub200["weights"], sub200["indices"], sub200["indptr"]), shape=(200, 200))
    sp_mst = minimum_spanning_tree(sub200_mat)
    sp_weight = int(sp_mst.sum())

    mst_match = (w_prim == w_kruskal == sp_weight)
    val_report["mst_prim_kruskal_scipy"] = {
        "subgrafo_n": 200,
        "peso_prim": int(w_prim),
        "peso_kruskal": int(w_kruskal),
        "peso_scipy": sp_weight,
        "match": mst_match
    }
    print(f"   [MST Prim == Kruskal == SciPy] Pesos: Prim={w_prim}, Kruskal={w_kruskal}, SciPy={sp_weight} -> Match={mst_match}")

    scc_arr, scc_count = tarjan_scc_iterative(d_indptr, d_indices)
    scc_sizes = np.bincount(scc_arr)
    largest_scc = int(scc_sizes.max())
    val_report["tarjan_scc"] = {
        "total_sccs": int(scc_count),
        "maior_scc": largest_scc,
        "maior_scc_snap": 69501,
        "fracao_scc_pct": round(largest_scc / N_dir * 100, 2),
        "match_snap": bool(largest_scc == 69501)
    }
    print(f"   [Tarjan Dígrafo] SCCs={scc_count:,}, Maior SCC={largest_scc:,} (SNAP: 69.501) -> Match={largest_scc == 69501}")

    with open("results/parte2_validacao.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2, ensure_ascii=False)

    print("\n5. Executando medições de tempo e benchmarks estatísticos...")
    rng = np.random.default_rng(42)
    benchmarks = {}
    summary_table_rows = []

    print("\n -> Benchmark Floyd-Warshall (O(n³))...")
    _ = floyd_warshall_csr(subgraphs_data[100]["indptr"], subgraphs_data[100]["indices"], subgraphs_data[100]["weights"])

    fw_sizes = [100, 200, 400, 600, 800]
    fw_results = []
    for n in fw_sizes:
        sub = subgraphs_data[n]
        reps = 30 if n < 600 else 10
        times = []
        for _ in range(reps):
            t_run_0 = time.perf_counter()
            _ = floyd_warshall_csr(sub["indptr"], sub["indices"], sub["weights"])
            times.append(time.perf_counter() - t_run_0)

        ci_res = calculate_confidence_interval(np.array(times) * 1000, confidence=0.95)
        ci_res["n_nodes"] = n
        ci_res["m_edges"] = sub["m"]
        ci_res["reps"] = reps
        ci_res["stat"] = "Z" if reps >= 30 else "t-Student"
        fw_results.append(ci_res)
        print(f"    n={n:3d} (m={sub['m']:5,d}, {reps} reps, {ci_res['stat']}): {ci_res['mean']:.2f} ± {ci_res['margin']:.2f} ms")

        summary_table_rows.append({
            "Algoritmo": "Floyd-Warshall",
            "Tamanho (n)": n,
            "Arestas (m)": sub["m"],
            "Repetições": reps,
            "Estatística": ci_res["stat"],
            "Tempo Médio (ms)": ci_res["mean"],
            "Margem IC 95% (ms)": ci_res["margin"],
            "Tempo (Média ± IC 95%)": f"{ci_res['mean']:.2f} ± {ci_res['margin']:.2f} ms",
            "Complexidade Teórica": "O(n³)"
        })

    benchmarks["floyd_warshall"] = fw_results

    print("\n -> Benchmark Bellman-Ford (O(n * m)) com parada antecipada...")
    _ = bellman_ford_csr(subgraphs_data[500]["indptr"], subgraphs_data[500]["indices"], subgraphs_data[500]["weights"], 0)

    bf_sizes = [500, 1000, 2000, 3000]
    bf_results = []
    for n in bf_sizes:
        sub = subgraphs_data[n]
        reps = 30 if n < 2000 else 10
        src_nodes = rng.integers(0, n, size=reps)
        times = []
        rounds_list = []
        for s in src_nodes:
            t_run_0 = time.perf_counter()
            _, r = bellman_ford_csr(sub["indptr"], sub["indices"], sub["weights"], int(s))
            times.append(time.perf_counter() - t_run_0)
            rounds_list.append(r)

        ci_res = calculate_confidence_interval(np.array(times) * 1000, confidence=0.95)
        ci_res["n_nodes"] = n
        ci_res["m_edges"] = sub["m"]
        ci_res["reps"] = reps
        ci_res["stat"] = "Z" if reps >= 30 else "t-Student"
        ci_res["mean_rounds"] = float(np.mean(rounds_list))
        bf_results.append(ci_res)
        print(f"    n={n:4d} (m={sub['m']:6,d}, rounds={ci_res['mean_rounds']:.1f}, {reps} reps, {ci_res['stat']}): {ci_res['mean']:.2f} ± {ci_res['margin']:.2f} ms")

        summary_table_rows.append({
            "Algoritmo": "Bellman-Ford",
            "Tamanho (n)": n,
            "Arestas (m)": sub["m"],
            "Repetições": reps,
            "Estatística": ci_res["stat"],
            "Tempo Médio (ms)": ci_res["mean"],
            "Margem IC 95% (ms)": ci_res["margin"],
            "Tempo (Média ± IC 95%)": f"{ci_res['mean']:.2f} ± {ci_res['margin']:.2f} ms",
            "Complexidade Teórica": "O(n · m)"
        })

    benchmarks["bellman_ford"] = bf_results

    print("\n -> Benchmark Prim vs Kruskal (MST)...")
    mst_sizes = [500, 1000, 2000, 3000, 5000]
    prim_results = []
    kruskal_results = []

    _ = prim_mst_csr(subgraphs_data[500]["indptr"], subgraphs_data[500]["indices"], subgraphs_data[500]["weights"])

    for n in mst_sizes:
        sub = subgraphs_data[n]
        reps = 30 if n <= 2000 else 10

        u_l, v_l, w_l = [], [], []
        sub_ind = sub["indptr"]
        sub_indices = sub["indices"]
        sub_w = sub["weights"]
        for u in range(n):
            for i in range(sub_ind[u], sub_ind[u+1]):
                v = sub_indices[i]
                if u < v:
                    u_l.append(u)
                    v_l.append(v)
                    w_l.append(sub_w[i])
        u_arr = np.array(u_l, dtype=np.int32)
        v_arr = np.array(v_l, dtype=np.int32)
        w_arr = np.array(w_l, dtype=np.int32)

        if n == mst_sizes[0]:
            _ = kruskal_mst(n, u_arr, v_arr, w_arr)

        times_prim = []
        times_kruskal = []
        for _ in range(reps):
            t_p0 = time.perf_counter()
            _ = prim_mst_csr(sub_ind, sub_indices, sub_w)
            times_prim.append(time.perf_counter() - t_p0)

            t_k0 = time.perf_counter()
            _ = kruskal_mst(n, u_arr, v_arr, w_arr)
            times_kruskal.append(time.perf_counter() - t_k0)

        ci_prim = calculate_confidence_interval(np.array(times_prim) * 1000, confidence=0.95)
        ci_prim["n_nodes"] = n
        ci_prim["m_edges"] = sub["m"]
        ci_prim["reps"] = reps
        ci_prim["stat"] = "Z" if reps >= 30 else "t-Student"
        prim_results.append(ci_prim)

        ci_kruskal = calculate_confidence_interval(np.array(times_kruskal) * 1000, confidence=0.95)
        ci_kruskal["n_nodes"] = n
        ci_kruskal["m_edges"] = sub["m"]
        ci_kruskal["reps"] = reps
        ci_kruskal["stat"] = "Z" if reps >= 30 else "t-Student"
        kruskal_results.append(ci_kruskal)

        print(f"    n={n:4d} (m={sub['m']:7,d}, {reps} reps, {ci_prim['stat']}):")
        print(f"       Prim:    {ci_prim['mean']:6.2f} ± {ci_prim['margin']:5.2f} ms")
        print(f"       Kruskal: {ci_kruskal['mean']:6.2f} ± {ci_kruskal['margin']:5.2f} ms")

        summary_table_rows.append({
            "Algoritmo": "Prim (MST)",
            "Tamanho (n)": n,
            "Arestas (m)": sub["m"],
            "Repetições": reps,
            "Estatística": ci_prim["stat"],
            "Tempo Médio (ms)": ci_prim["mean"],
            "Margem IC 95% (ms)": ci_prim["margin"],
            "Tempo (Média ± IC 95%)": f"{ci_prim['mean']:.2f} ± {ci_prim['margin']:.2f} ms",
            "Complexidade Teórica": "O(m log n)"
        })
        summary_table_rows.append({
            "Algoritmo": "Kruskal (MST)",
            "Tamanho (n)": n,
            "Arestas (m)": sub["m"],
            "Repetições": reps,
            "Estatística": ci_kruskal["stat"],
            "Tempo Médio (ms)": ci_kruskal["mean"],
            "Margem IC 95% (ms)": ci_kruskal["margin"],
            "Tempo (Média ± IC 95%)": f"{ci_kruskal['mean']:.2f} ± {ci_kruskal['margin']:.2f} ms",
            "Complexidade Teórica": "O(m · α(n))"
        })

    benchmarks["prim"] = prim_results
    benchmarks["kruskal"] = kruskal_results

    print("\n -> Executando Kruskal 1 vez no Grafo Completo (n=107.614, m=12.238.285)...")
    t_ext0 = time.perf_counter()
    full_u_l, full_v_l, full_w_l = [], [], []
    for u in range(N):
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if u < v:
                full_u_l.append(u)
                full_v_l.append(v)
                full_w_l.append(weights[i])
    full_u = np.array(full_u_l, dtype=np.int32)
    full_v = np.array(full_v_l, dtype=np.int32)
    full_w = np.array(full_w_l, dtype=np.int32)
    print(f"    Arestas do grafo completo extraídas em {time.perf_counter() - t_ext0:.2f}s ({len(full_w):,} arestas).")

    times_kruskal_full = []
    for rep in range(5):
        t_kf0 = time.perf_counter()
        full_mst_weight, full_mst_edges = kruskal_mst(N, full_u, full_v, full_w)
        times_kruskal_full.append(time.perf_counter() - t_kf0)

    ci_kruskal_full = calculate_confidence_interval(np.array(times_kruskal_full) * 1000, confidence=0.95)
    ci_kruskal_full["n_nodes"] = N
    ci_kruskal_full["m_edges"] = M_undir
    ci_kruskal_full["mst_weight"] = int(full_mst_weight)
    ci_kruskal_full["mst_edges"] = int(full_mst_edges)
    ci_kruskal_full["reps"] = 5
    ci_kruskal_full["stat"] = "t-Student"
    benchmarks["kruskal_full"] = ci_kruskal_full

    print(f"    Kruskal no Grafo Completo: Peso da MST = {full_mst_weight:,} ({full_mst_edges:,} arestas)")
    print(f"    Tempo de execução (5 reps): {ci_kruskal_full['mean']:.2f} ± {ci_kruskal_full['margin']:.2f} ms")

    summary_table_rows.append({
        "Algoritmo": "Kruskal (Grafo Completo)",
        "Tamanho (n)": N,
        "Arestas (m)": M_undir,
        "Repetições": 5,
        "Estatística": "t-Student",
        "Tempo Médio (ms)": ci_kruskal_full["mean"],
        "Margem IC 95% (ms)": ci_kruskal_full["margin"],
        "Tempo (Média ± IC 95%)": f"{ci_kruskal_full['mean']:.2f} ± {ci_kruskal_full['margin']:.2f} ms",
        "Complexidade Teórica": "O(m · α(n))"
    })

    print("\n -> Benchmark BFS, DFS e Dijkstra no Grafo Completo (30 fontes aleatórias, IC-Z)...")
    test_srcs = rng.integers(0, N, size=30)

    times_bfs_full = []
    times_dfs_full = []
    times_dijk_full = []

    _ = bfs_distances_csr(indptr, indices, 0)
    _ = dfs_iterative_csr(indptr, indices, 0)
    _ = dijkstra_csr(indptr, indices, weights, 0)

    for s in test_srcs:
        t0 = time.perf_counter()
        _ = bfs_distances_csr(indptr, indices, int(s))
        times_bfs_full.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = dfs_iterative_csr(indptr, indices, int(s))
        times_dfs_full.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = dijkstra_csr(indptr, indices, weights, int(s))
        times_dijk_full.append(time.perf_counter() - t0)

    ci_bfs_full = calculate_confidence_interval(np.array(times_bfs_full) * 1000, confidence=0.95)
    ci_dfs_full = calculate_confidence_interval(np.array(times_dfs_full) * 1000, confidence=0.95)
    ci_dijk_full = calculate_confidence_interval(np.array(times_dijk_full) * 1000, confidence=0.95)

    benchmarks["bfs_full"] = ci_bfs_full
    benchmarks["dfs_full"] = ci_dfs_full
    benchmarks["dijkstra_full"] = ci_dijk_full

    print(f"    BFS (Grafo Completo):      {ci_bfs_full['mean']:.2f} ± {ci_bfs_full['margin']:.2f} ms (IC-Z)")
    print(f"    DFS (Grafo Completo):      {ci_dfs_full['mean']:.2f} ± {ci_dfs_full['margin']:.2f} ms (IC-Z)")
    print(f"    Dijkstra (Grafo Completo): {ci_dijk_full['mean']:.2f} ± {ci_dijk_full['margin']:.2f} ms (IC-Z)")

    summary_table_rows.append({
        "Algoritmo": "BFS (Grafo Completo)",
        "Tamanho (n)": N,
        "Arestas (m)": M_undir,
        "Repetições": 30,
        "Estatística": "Z",
        "Tempo Médio (ms)": ci_bfs_full["mean"],
        "Margem IC 95% (ms)": ci_bfs_full["margin"],
        "Tempo (Média ± IC 95%)": f"{ci_bfs_full['mean']:.2f} ± {ci_bfs_full['margin']:.2f} ms",
        "Complexidade Teórica": "O(n + m)"
    })
    summary_table_rows.append({
        "Algoritmo": "DFS (Grafo Completo)",
        "Tamanho (n)": N,
        "Arestas (m)": M_undir,
        "Repetições": 30,
        "Estatística": "Z",
        "Tempo Médio (ms)": ci_dfs_full["mean"],
        "Margem IC 95% (ms)": ci_dfs_full["margin"],
        "Tempo (Média ± IC 95%)": f"{ci_dfs_full['mean']:.2f} ± {ci_dfs_full['margin']:.2f} ms",
        "Complexidade Teórica": "O(n + m)"
    })
    summary_table_rows.append({
        "Algoritmo": "Dijkstra (Grafo Completo)",
        "Tamanho (n)": N,
        "Arestas (m)": M_undir,
        "Repetições": 30,
        "Estatística": "Z",
        "Tempo Médio (ms)": ci_dijk_full["mean"],
        "Margem IC 95% (ms)": ci_dijk_full["margin"],
        "Tempo (Média ± IC 95%)": f"{ci_dijk_full['mean']:.2f} ± {ci_dijk_full['margin']:.2f} ms",
        "Complexidade Teórica": "O(m log n)"
    })

    print("\n -> Benchmark Tarjan no Dígrafo (10 repetições, IC t-Student)...")
    _ = tarjan_scc_iterative(d_indptr, d_indices)
    times_tarjan = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = tarjan_scc_iterative(d_indptr, d_indices)
        times_tarjan.append(time.perf_counter() - t0)

    ci_tarjan = calculate_confidence_interval(np.array(times_tarjan) * 1000, confidence=0.95)
    benchmarks["tarjan_digraph"] = ci_tarjan
    print(f"    Tarjan (Dígrafo Completo): {ci_tarjan['mean']:.2f} ± {ci_tarjan['margin']:.2f} ms (IC t-Student)")

    summary_table_rows.append({
        "Algoritmo": "Tarjan (Dígrafo Completo)",
        "Tamanho (n)": N_dir,
        "Arestas (m)": M_dir,
        "Repetições": 10,
        "Estatística": "t-Student",
        "Tempo Médio (ms)": ci_tarjan["mean"],
        "Margem IC 95% (ms)": ci_tarjan["margin"],
        "Tempo (Média ± IC 95%)": f"{ci_tarjan['mean']:.2f} ± {ci_tarjan['margin']:.2f} ms",
        "Complexidade Teórica": "O(V + E)"
    })

    df_tempos = pd.DataFrame(summary_table_rows)
    df_tempos.to_csv("results/parte2_tempos_tabela.csv", index=False, encoding="utf-8-sig")
    with open("results/parte2_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em results/parte2_benchmarks.json e results/parte2_tempos_tabela.csv")

    print("\n6. Gerando figuras de análise de complexidade (figures/)...")

    fw_n = [item["n_nodes"] for item in fw_results]
    fw_y = [item["mean"] for item in fw_results]
    fw_err = [item["margin"] for item in fw_results]
    c_fw, r2_fw, _ = fit_complexity_curve(fw_n, fw_y, lambda x: x**3)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    ax.errorbar(fw_n, fw_y, yerr=fw_err, fmt="o", color="#2b6cb0", ecolor="#e53e3e",
                capsize=5, elinewidth=1.8, markersize=7, label="Tempo Observado (Média ± IC 95%)")

    x_smooth = np.linspace(min(fw_n), max(fw_n), 200)
    y_smooth = c_fw * (x_smooth ** 3)
    ax.plot(x_smooth, y_smooth, "--", color="#2c7a7b", linewidth=2.0,
            label=f"Curva Teórica Ajustada $c \\cdot n^3$\n($c = {c_fw:.3e}$, $R^2 = {r2_fw:.4f}$)")

    ax.set_xlabel("Ordem do Subgrafo ($n$ nós)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tempo de Execução (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Floyd-Warshall: Tempo Observado × Complexidade Teórica $O(n^3)$", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/parte2_floyd_warshall.png", bbox_inches="tight")
    plt.close()
    print("   Salvo: figures/parte2_floyd_warshall.png")

    bf_n = [item["n_nodes"] for item in bf_results]
    bf_m = [item["m_edges"] for item in bf_results]
    bf_y = [item["mean"] for item in bf_results]
    bf_err = [item["margin"] for item in bf_results]

    bf_prod = np.array([n * m for n, m in zip(bf_n, bf_m)])
    c_bf, r2_bf, _ = fit_complexity_curve(bf_prod, bf_y, lambda x: x)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    ax.errorbar(bf_n, bf_y, yerr=bf_err, fmt="s", color="#d69e2e", ecolor="#e53e3e",
                capsize=5, elinewidth=1.8, markersize=7, label="Tempo Observado (Média ± IC 95%)")

    bf_y_curve = [c_bf * (n * m) for n, m in zip(bf_n, bf_m)]
    ax.plot(bf_n, bf_y_curve, "--", color="#c53030", linewidth=2.0,
            label=f"Curva Teórica Ajustada $c \\cdot (n \\cdot m)$\n($c = {c_bf:.3e}$, $R^2 = {r2_bf:.4f}$)")

    ax.set_xlabel("Ordem do Subgrafo ($n$ nós)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tempo de Execução (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Bellman-Ford (Parada Antecipada): Tempo Observado × $O(n \\cdot m)$", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/parte2_bellman_ford.png", bbox_inches="tight")
    plt.close()
    print("   Salvo: figures/parte2_bellman_ford.png")

    mst_n = [item["n_nodes"] for item in prim_results]
    mst_m = [item["m_edges"] for item in prim_results]
    prim_y = [item["mean"] for item in prim_results]
    prim_err = [item["margin"] for item in prim_results]
    kruskal_y = [item["mean"] for item in kruskal_results]
    kruskal_err = [item["margin"] for item in kruskal_results]

    prim_prod = [m * np.log2(n) for n, m in zip(mst_n, mst_m)]
    c_p, r2_p, _ = fit_complexity_curve(prim_prod, prim_y, lambda x: x)

    c_k, r2_k, _ = fit_complexity_curve(mst_m, kruskal_y, lambda x: x)

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=200)
    ax.errorbar(mst_n, prim_y, yerr=prim_err, fmt="o-", color="#2b6cb0", ecolor="#2b6cb0",
                capsize=5, elinewidth=1.6, markersize=6, label=f"Prim: $O(m \\log n)$ ($R^2={r2_p:.4f}$)")
    ax.errorbar(mst_n, kruskal_y, yerr=kruskal_err, fmt="s-", color="#38a169", ecolor="#38a169",
                capsize=5, elinewidth=1.6, markersize=6, label=f"Kruskal: $O(m \\cdot \\alpha(n))$ ($R^2={r2_k:.4f}$)")

    ax.set_xlabel("Ordem do Subgrafo ($n$ nós)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tempo de Execução (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Árvore Geradora Mínima: Prim vs Kruskal em Subgrafos Crescentes", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/parte2_mst_prim_kruskal.png", bbox_inches="tight")
    plt.close()
    print("   Salvo: figures/parte2_mst_prim_kruskal.png")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)

    alg_labels = [
        "BFS\n(Completo)",
        "DFS\n(Completo)",
        "Dijkstra\n(Completo)",
        "Kruskal\n(Completo)",
        "Tarjan\n(Dígrafo)",
        "Prim\n(n=5k)",
        "Bellman-Ford\n(n=3k)",
        "Floyd-Warshall\n(n=800)"
    ]
    alg_means = [
        ci_bfs_full["mean"],
        ci_dfs_full["mean"],
        ci_dijk_full["mean"],
        ci_kruskal_full["mean"],
        ci_tarjan["mean"],
        prim_results[-1]["mean"],
        bf_results[-1]["mean"],
        fw_results[-1]["mean"]
    ]
    alg_errs = [
        ci_bfs_full["margin"],
        ci_dfs_full["margin"],
        ci_dijk_full["margin"],
        ci_kruskal_full["margin"],
        ci_tarjan["margin"],
        prim_results[-1]["margin"],
        bf_results[-1]["margin"],
        fw_results[-1]["margin"]
    ]
    colors = ["#3182ce", "#2b6cb0", "#4299e1", "#38a169", "#805ad5", "#dd6b20", "#e53e3e", "#d69e2e"]

    bars = ax.bar(alg_labels, alg_means, yerr=alg_errs, capsize=5, color=colors, alpha=0.85, edgecolor="#1a202c")
    ax.set_ylabel("Tempo de Execução Médio (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Visão Geral de Desempenho dos Algoritmos Implementados (Média ± IC 95%)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f} ms",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/parte2_resumo_tempos.png", bbox_inches="tight")
    plt.close()
    print("   Salvo: figures/parte2_resumo_tempos.png")

    t_tot = time.perf_counter() - t_start
    print(f"\n=== PIPELINE DA PARTE II FINALIZADO COM SUCESSO EM {t_tot:.2f}s ===")

if __name__ == "__main__":
    main()
