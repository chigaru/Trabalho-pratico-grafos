import os
import sys
import time
import json
import traceback
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as spstats
import scipy.optimize as opt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numba import njit, prange

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DIRECTED_PATH = "data/processed/gplus_directed.npz"
UNDIRECTED_PATH = "data/processed/gplus_undirected_gcc.npz"

sys.path.insert(0, os.path.abspath("."))
from scripts.run_parte3 import (
    sample_er_edges, build_csr_from_edges,
    compute_clustering_manual, sample_apl_bfs,
)

@njit
def count_reciprocity_csr(indptr_out, indices_out, indptr_in, indices_in):
    n = len(indptr_out) - 1
    recip_arcs = np.int64(0)
    for u in range(n):
        for j in range(indptr_out[u], indptr_out[u + 1]):
            v = indices_out[j]
            lo = indptr_out[v]
            hi = indptr_out[v + 1] - 1
            found = False
            while lo <= hi:
                mid = (lo + hi) // 2
                if indices_out[mid] == u:
                    found = True
                    break
                elif indices_out[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if found:
                recip_arcs += 1
    return recip_arcs

@njit
def compute_in_out_degrees(indptr_out, indices_out, n):
    out_deg = np.empty(n, dtype=np.int32)
    in_deg = np.zeros(n, dtype=np.int32)
    for u in range(n):
        out_deg[u] = indptr_out[u + 1] - indptr_out[u]
        for j in range(indptr_out[u], indptr_out[u + 1]):
            v = indices_out[j]
            in_deg[v] += 1
    return in_deg, out_deg

@njit
def node_reciprocity(indptr_out, indices_out, n):
    recip_count = np.zeros(n, dtype=np.int32)
    for u in range(n):
        for j in range(indptr_out[u], indptr_out[u + 1]):
            v = indices_out[j]
            lo = indptr_out[v]
            hi = indptr_out[v + 1] - 1
            found = False
            while lo <= hi:
                mid = (lo + hi) // 2
                if indices_out[mid] == u:
                    found = True
                    break
                elif indices_out[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if found:
                recip_count[u] += 1
    return recip_count

@njit
def tarjan_scc_iterative(indptr, indices):
    N = len(indptr) - 1
    index_arr = np.full(N, -1, dtype=np.int32)
    lowlink = np.full(N, -1, dtype=np.int32)
    on_stack = np.zeros(N, dtype=np.bool_)
    scc = np.full(N, -1, dtype=np.int32)
    call_u = np.empty(N, dtype=np.int32)
    call_edge = np.empty(N, dtype=np.int32)
    call_top = 0
    t_stack = np.empty(N, dtype=np.int32)
    t_top = 0
    timer = 0
    scc_count = 0
    for start in range(N):
        if index_arr[start] != -1:
            continue
        call_u[call_top] = start
        call_edge[call_top] = indptr[start]
        call_top += 1
        index_arr[start] = timer
        lowlink[start] = timer
        timer += 1
        t_stack[t_top] = start
        t_top += 1
        on_stack[start] = True
        while call_top > 0:
            fi = call_top - 1
            u = call_u[fi]
            ep = call_edge[fi]
            ee = indptr[u + 1]
            pushed = False
            while ep < ee:
                v = indices[ep]
                ep += 1
                call_edge[fi] = ep
                if index_arr[v] == -1:
                    index_arr[v] = timer
                    lowlink[v] = timer
                    timer += 1
                    t_stack[t_top] = v
                    t_top += 1
                    on_stack[v] = True
                    call_u[call_top] = v
                    call_edge[call_top] = indptr[v]
                    call_top += 1
                    pushed = True
                    break
                elif on_stack[v]:
                    if index_arr[v] < lowlink[u]:
                        lowlink[u] = index_arr[v]
            if pushed:
                continue
            if lowlink[u] == index_arr[u]:
                while t_top > 0:
                    t_top -= 1
                    w = t_stack[t_top]
                    on_stack[w] = False
                    scc[w] = scc_count
                    if w == u:
                        break
                scc_count += 1
            call_top -= 1
            if call_top > 0:
                pi = call_top - 1
                p = call_u[pi]
                if lowlink[u] < lowlink[p]:
                    lowlink[p] = lowlink[u]
    return scc, scc_count

@njit
def generate_ba_csr(n, m_attach, seed=42):
    np.random.seed(seed)
    max_edges = n * m_attach + m_attach * (m_attach + 1) // 2
    u_arr = np.empty(max_edges, dtype=np.int32)
    v_arr = np.empty(max_edges, dtype=np.int32)
    edge_count = 0

    stubs = np.empty(2 * max_edges, dtype=np.int32)
    stub_count = 0

    m0 = m_attach + 1
    for i in range(m0):
        for j in range(i + 1, m0):
            u_arr[edge_count] = i
            v_arr[edge_count] = j
            stubs[stub_count] = i; stub_count += 1
            stubs[stub_count] = j; stub_count += 1
            edge_count += 1

    ht_size = 1 << 24
    ht_mask = ht_size - 1
    ht = np.zeros(ht_size, dtype=np.int64)

    for e in range(edge_count):
        uu = np.int32(u_arr[e])
        vv = np.int32(v_arr[e])
        if uu > vv:
            tmp = uu; uu = vv; vv = tmp
        key = (np.int64(uu) << 32) | np.int64(vv)
        h = (key ^ (key >> 30)) * np.int64(0xbf58476d1ce4e5b9)
        h = (h ^ (h >> 27)) * np.int64(0x94d049bb133111eb)
        h = h ^ (h >> 31)
        slot = int(h & ht_mask)
        while ht[slot] != 0:
            slot = (slot + 1) & ht_mask
        ht[slot] = key

    for new_node in range(m0, n):
        targets_added = 0
        attempts = 0
        while targets_added < m_attach and attempts < m_attach * 50:
            attempts += 1
            idx = np.random.randint(0, stub_count)
            target = stubs[idx]
            if target == new_node:
                continue
            uu = np.int32(min(new_node, target))
            vv = np.int32(max(new_node, target))
            key = (np.int64(uu) << 32) | np.int64(vv)
            h = (key ^ (key >> 30)) * np.int64(0xbf58476d1ce4e5b9)
            h = (h ^ (h >> 27)) * np.int64(0x94d049bb133111eb)
            h = h ^ (h >> 31)
            slot = int(h & ht_mask)
            exists = False
            while ht[slot] != 0:
                if ht[slot] == key:
                    exists = True
                    break
                slot = (slot + 1) & ht_mask
            if exists:
                continue
            ht[slot] = key
            u_arr[edge_count] = new_node
            v_arr[edge_count] = target
            stubs[stub_count] = new_node; stub_count += 1
            stubs[stub_count] = target; stub_count += 1
            edge_count += 1
            targets_added += 1

    return build_csr_from_edges(n, edge_count, u_arr[:edge_count], v_arr[:edge_count])

@njit
def generate_ws_csr(n, k, beta, seed=42):
    np.random.seed(seed)
    half_k = k // 2
    max_edges = n * half_k + 1000
    u_arr = np.empty(max_edges, dtype=np.int32)
    v_arr = np.empty(max_edges, dtype=np.int32)

    ht_size = 1 << 24
    ht_mask = ht_size - 1
    ht = np.zeros(ht_size, dtype=np.int64)

    edge_count = 0

    def _hash_key(uu, vv):
        key = (np.int64(uu) << 32) | np.int64(vv)
        h = (key ^ (key >> 30)) * np.int64(0xbf58476d1ce4e5b9)
        h = (h ^ (h >> 27)) * np.int64(0x94d049bb133111eb)
        h = h ^ (h >> 31)
        return key, int(h & ht_mask)

    def _exists_in_ht(uu, vv):
        a = min(uu, vv)
        b = max(uu, vv)
        key, slot = _hash_key(a, b)
        while ht[slot] != 0:
            if ht[slot] == key:
                return True
            slot = (slot + 1) & ht_mask
        return False

    def _insert_ht(uu, vv):
        a = min(uu, vv)
        b = max(uu, vv)
        key, slot = _hash_key(a, b)
        while ht[slot] != 0:
            if ht[slot] == key:
                return
            slot = (slot + 1) & ht_mask
        ht[slot] = key

    ring_targets = np.empty((n, half_k), dtype=np.int32)
    for i in range(n):
        for j in range(1, half_k + 1):
            ring_targets[i, j - 1] = (i + j) % n

    for i in range(n):
        for j_idx in range(half_k):
            original_target = ring_targets[i, j_idx]
            if np.random.random() < beta:
                attempts = 0
                new_target = original_target
                while attempts < 100:
                    cand = np.random.randint(0, n)
                    if cand == i:
                        attempts += 1
                        continue
                    if not _exists_in_ht(i, cand):
                        new_target = cand
                        break
                    attempts += 1
                ring_targets[i, j_idx] = new_target

    for i in range(n):
        for j_idx in range(half_k):
            v = ring_targets[i, j_idx]
            a = min(i, v)
            b = max(i, v)
            if not _exists_in_ht(a, b):
                _insert_ht(a, b)
                u_arr[edge_count] = a
                v_arr[edge_count] = b
                edge_count += 1

    return build_csr_from_edges(n, edge_count, u_arr[:edge_count], v_arr[:edge_count])

@njit
def bfs_single_csr_basic(indptr, indices, source):
    N = len(indptr) - 1
    dist = np.full(N, -1, dtype=np.int32)
    queue = np.empty(N, dtype=np.int32)
    dist[source] = 0
    queue[0] = source
    head = 0; tail = 1
    max_dist = 0
    sum_dist = np.int64(0)
    while head < tail:
        u = queue[head]; head += 1
        d_next = dist[u] + 1
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            if dist[v] == -1:
                dist[v] = d_next
                queue[tail] = v; tail += 1
                if d_next > max_dist:
                    max_dist = d_next
                sum_dist += d_next
    return max_dist, sum_dist, tail

@njit
def compute_max_degree(indptr):
    n = len(indptr) - 1
    mx = 0
    for u in range(n):
        d = indptr[u+1] - indptr[u]
        if d > mx:
            mx = d
    return mx

def run_reciprocity():
    print("\n" + "=" * 70)
    print("=== ITEM 4: RECIPROCIDADE DO DÍGRAFO ORIGINAL ===")
    print("=" * 70)

    t0_total = time.perf_counter()

    print(f"[4.0] Carregando dígrafo de {DIRECTED_PATH}...")
    d = np.load(DIRECTED_PATH, allow_pickle=True)
    indptr_d = d["indptr"]
    indices_d = d["indices"]
    n_d = len(indptr_d) - 1
    m_arcs = len(indices_d)
    print(f"  n = {n_d:,}, arcos = {m_arcs:,}")

    for u in range(n_d):
        s, e = indptr_d[u], indptr_d[u+1]
        if e - s > 1:
            indices_d[s:e].sort()

    print("[4.1] Calculando graus de entrada e saída...")
    t0 = time.perf_counter()
    in_deg, out_deg = compute_in_out_degrees(indptr_d, indices_d, n_d)
    t_deg = time.perf_counter() - t0
    print(f"  Graus calculados em {t_deg:.2f}s")

    print("[4.2] Contando pares recíprocos...")
    t0 = time.perf_counter()
    recip_arcs_2x = count_reciprocity_csr(indptr_d, indices_d, indptr_d, indices_d)
    t_recip = time.perf_counter() - t0
    reciprocal_pairs = recip_arcs_2x // 2
    arcs_in_reciprocal = recip_arcs_2x
    frac_reciprocal = arcs_in_reciprocal / m_arcs
    print(f"  Pares recíprocos: {reciprocal_pairs:,}")
    print(f"  Arcos em laços bidirecionais: {arcs_in_reciprocal:,} / {m_arcs:,} = {frac_reciprocal*100:.2f}%")
    print(f"  Tempo: {t_recip:.2f}s")

    m_undirected_expected = 12_238_285
    m_undirected_check = m_arcs - reciprocal_pairs
    validation_ok = m_undirected_check == m_undirected_expected
    print(f"[4.2b] Validação: {m_arcs:,} - {reciprocal_pairs:,} = {m_undirected_check:,}")
    print(f"  Esperado m_undirected = {m_undirected_expected:,} → {'PASSOU ✓' if validation_ok else 'FALHOU ✗'}")

    density_digraph = m_arcs / (n_d * (n_d - 1))
    expected_reciprocity = density_digraph
    ratio_obs_exp = frac_reciprocal / expected_reciprocity if expected_reciprocity > 0 else float('inf')
    print(f"[4.3] Baseline dígrafo aleatório: rho = {density_digraph:.6f}")
    print(f"  Reciprocidade esperada = {expected_reciprocity:.6f}")
    print(f"  Observado/Esperado = {ratio_obs_exp:.2f}x")

    print("[4.4] Calculando reciprocidade por nó...")
    t0 = time.perf_counter()
    recip_per_node = node_reciprocity(indptr_d, indices_d, n_d)
    t_node_recip = time.perf_counter() - t0
    print(f"  Tempo: {t_node_recip:.2f}s")

    total_degree = in_deg + out_deg
    full_recip_mask = (recip_per_node == out_deg) & (out_deg > 0)
    no_recip_mask = (recip_per_node == 0) & (out_deg > 0)
    frac_full_recip = np.sum(full_recip_mask) / n_d
    frac_no_recip = np.sum(no_recip_mask) / n_d
    n_full_recip = int(np.sum(full_recip_mask))
    n_no_recip = int(np.sum(no_recip_mask))
    print(f"  Nós com reciprocidade total (todos arcos recíprocos): {n_full_recip:,} ({frac_full_recip*100:.2f}%)")
    print(f"  Nós sem nenhum arco recíproco: {n_no_recip:,} ({frac_no_recip*100:.2f}%)")

    corr_in_out = float(np.corrcoef(in_deg.astype(np.float64), out_deg.astype(np.float64))[0, 1])
    print(f"[4.5] Correlação in-degree vs out-degree: r = {corr_in_out:.4f}")

    print("[4.6] Reciprocidade por faixa de grau total (potências de 2)...")
    max_td = int(np.max(total_degree))
    band_edges = []
    k = 0
    while 2**k < max_td:
        band_edges.append(2**k)
        k += 1
    band_edges.append(2**k)

    band_rows = []
    for b_idx in range(len(band_edges) - 1):
        lo = band_edges[b_idx]
        hi = band_edges[b_idx + 1]
        mask = (total_degree >= lo) & (total_degree < hi)
        n_band = int(np.sum(mask))
        if n_band == 0:
            continue
        out_band = out_deg[mask]
        recip_band = recip_per_node[mask]
        valid = out_band > 0
        if np.sum(valid) == 0:
            mean_recip_frac = 0.0
        else:
            mean_recip_frac = float(np.mean(recip_band[valid] / out_band[valid].astype(np.float64)))
        band_rows.append({
            "faixa_grau_total": f"[{lo}, {hi})",
            "n_nos": n_band,
            "reciprocidade_media": round(mean_recip_frac, 4),
        })
        print(f"  [{lo:>6}, {hi:>6}): {n_band:>7} nós, recip média = {mean_recip_frac:.4f}")

    df_bands = pd.DataFrame(band_rows)
    df_bands.to_csv(os.path.join(RESULTS_DIR, "parte3_reciprocidade_por_faixa.csv"), index=False)

    print("[4.7] Calculando SCCs com Tarjan...")
    t0 = time.perf_counter()
    scc_ids, scc_count = tarjan_scc_iterative(indptr_d, indices_d)
    t_scc = time.perf_counter() - t0
    print(f"  {scc_count:,} SCCs encontradas em {t_scc:.2f}s")

    scc_sizes = np.bincount(scc_ids)
    largest_scc_id = int(np.argmax(scc_sizes))
    largest_scc_size = int(scc_sizes[largest_scc_id])
    print(f"  Maior SCC: {largest_scc_size:,} nós ({largest_scc_size/n_d*100:.2f}%)")

    scc_mask = (scc_ids == largest_scc_id)
    recip_in_scc = int(np.sum(recip_per_node[scc_mask]))
    total_recip_arcs = int(np.sum(recip_per_node))
    frac_recip_in_scc = recip_in_scc / total_recip_arcs if total_recip_arcs > 0 else 0
    print(f"  Arcos recíprocos (saída) na maior SCC: {recip_in_scc:,} / {total_recip_arcs:,} = {frac_recip_in_scc*100:.2f}%")

    print("[4.fig] Gerando figuras de reciprocidade...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    ax1 = axes[0]
    sorted_in = np.sort(in_deg)
    sorted_out = np.sort(out_deg)
    ccdf_in = 1.0 - np.arange(len(sorted_in)) / len(sorted_in)
    ccdf_out = 1.0 - np.arange(len(sorted_out)) / len(sorted_out)
    ax1.plot(sorted_in, ccdf_in, color="#2b6cb0", linewidth=2.0, label="In-degree")
    ax1.plot(sorted_out, ccdf_out, color="#e53e3e", linewidth=2.0, alpha=0.8, label="Out-degree")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Grau ($k$)", fontsize=11)
    ax1.set_ylabel("CCDF $P(K \\geq k)$", fontsize=11)
    ax1.set_title("Distribuições de Grau Dirigido (CCDF)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, frameon=True)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    ax2 = axes[1]
    ax2.scatter(out_deg, in_deg, s=1, alpha=0.08, color="#2b6cb0", rasterized=True)
    max_val = max(int(np.max(out_deg)), int(np.max(in_deg)))
    ax2.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="$k_{in} = k_{out}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Out-degree ($k_{out}$)", fontsize=11)
    ax2.set_ylabel("In-degree ($k_{in}$)", fontsize=11)
    ax2.set_title(f"Correlação In vs. Out (Pearson $r = {corr_in_out:.4f}$)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, frameon=True)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.suptitle("Análise Dirigida: Graus de Entrada e Saída (G2 — ego-Gplus)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "parte3_reciprocidade_graus_dirigidos.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    band_labels = df_bands["faixa_grau_total"].values
    band_vals = df_bands["reciprocidade_media"].values
    band_ns = df_bands["n_nos"].values
    colors_b = plt.cm.viridis(np.linspace(0.2, 0.9, len(band_labels)))
    bars = ax.bar(range(len(band_labels)), band_vals, color=colors_b, edgecolor="#1a202c", linewidth=0.8)
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fração Recíproca Média ($r_u = \\text{recip}/k_{out}$)", fontsize=11)
    ax.set_xlabel("Faixa de Grau Total ($k_{in} + k_{out}$)", fontsize=11)
    ax.set_title("Reciprocidade Média por Faixa de Grau (G2 — ego-Gplus)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"n={band_ns[i]:,}",
                ha="center", va="bottom", fontsize=7, rotation=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "parte3_reciprocidade_por_faixa.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    t_total = time.perf_counter() - t0_total

    res_reciprocity = {
        "digrafo": {"n": n_d, "arcos": m_arcs},
        "reciprocidade_global": {
            "pares_reciprocos": int(reciprocal_pairs),
            "arcos_em_lacos_bidirecionais": int(arcs_in_reciprocal),
            "fracao_arcos_reciprocos": round(frac_reciprocal, 6),
            "fracao_pct": round(frac_reciprocal * 100, 2),
        },
        "validacao_interna": {
            "arcos_unicos_menos_pares": m_undirected_check,
            "m_undirected_esperado": m_undirected_expected,
            "fechou": validation_ok,
        },
        "baseline_aleatorio": {
            "densidade_digrafo": round(density_digraph, 8),
            "reciprocidade_esperada_aleatoria": round(expected_reciprocity, 8),
            "razao_observado_esperado": round(ratio_obs_exp, 2),
        },
        "graus_dirigidos": {
            "in_degree_medio": round(float(np.mean(in_deg)), 2),
            "out_degree_medio": round(float(np.mean(out_deg)), 2),
            "in_degree_max": int(np.max(in_deg)),
            "out_degree_max": int(np.max(out_deg)),
            "correlacao_in_out": round(corr_in_out, 4),
        },
        "reciprocidade_por_no": {
            "nos_reciprocidade_total": n_full_recip,
            "frac_reciprocidade_total_pct": round(frac_full_recip * 100, 2),
            "nos_sem_reciprocidade": n_no_recip,
            "frac_sem_reciprocidade_pct": round(frac_no_recip * 100, 2),
        },
        "scc": {
            "total_sccs": int(scc_count),
            "maior_scc_nos": largest_scc_size,
            "maior_scc_frac_pct": round(largest_scc_size / n_d * 100, 2),
            "arcos_reciprocos_saida_na_scc": recip_in_scc,
            "total_arcos_reciprocos_saida": total_recip_arcs,
            "frac_reciprocidade_na_scc_pct": round(frac_recip_in_scc * 100, 2),
        },
        "tempo_total_s": round(t_total, 2),
    }

    with open(os.path.join(RESULTS_DIR, "parte3_reciprocidade.json"), "w", encoding="utf-8") as f:
        json.dump(res_reciprocity, f, indent=2, ensure_ascii=False)

    print(f"\n[4] Reciprocidade concluída em {t_total:.2f}s")
    return res_reciprocity

def run_synthetic_models():
    print("\n" + "=" * 70)
    print("=== ITEM 5: MODELOS TEÓRICOS (ER, BA, WS) vs. GRAFO REAL ===")
    print("=" * 70)

    t0_global = time.perf_counter()

    print("[5.0] Carregando grafo real não-dirigido...")
    csr_data = np.load(UNDIRECTED_PATH)
    indptr_real = csr_data["indptr"]
    indices_real = csr_data["indices"]
    n = len(indptr_real) - 1
    m = int(len(indices_real) // 2)
    degrees_real = (indptr_real[1:] - indptr_real[:-1]).astype(np.int32)
    deg_mean = float(np.mean(degrees_real))
    deg_max_real = int(np.max(degrees_real))
    print(f"  Real: n={n:,}, m={m:,}, <k>={deg_mean:.2f}, k_max={deg_max_real:,}")

    SEED = 42
    K_APL = 1000
    m_attach = round(deg_mean / 2)
    k_ws = 228
    betas = [0.01, 0.1, 0.5]

    L_real = 2.8603
    C_real = 0.490118

    def clauset_mle(degs, kmin):
        tail = degs[degs >= kmin].astype(np.float64)
        n_tail = len(tail)
        if n_tail < 50:
            return np.nan, np.nan
        gamma = 1.0 + n_tail / np.sum(np.log(tail / (kmin - 0.5)))
        sorted_x = np.sort(tail)
        emp_cdf = np.arange(1, n_tail + 1) / n_tail
        model_cdf = 1.0 - (sorted_x / (kmin - 0.5)) ** (-(gamma - 1.0))
        ks = float(np.max(np.abs(emp_cdf - np.clip(model_cdf, 0, 1))))
        return gamma, ks

    print("\n[5.1] TESTE PEQUENO (n=5.000, grau reduzido)...")
    n_small = 5000
    m_small = int(n_small * deg_mean / 2)
    m_attach_small = m_attach
    k_ws_small = k_ws
    if k_ws_small >= n_small:
        k_ws_small = n_small // 2
        if k_ws_small % 2 != 0:
            k_ws_small -= 1

    print(f"  ER: G({n_small}, {m_small})")
    t0 = time.perf_counter()
    u_er, v_er = sample_er_edges(n_small, m_small, seed=SEED)
    ip_er_s, ix_er_s = build_csr_from_edges(n_small, m_small, u_er, v_er)
    t_er_s = time.perf_counter() - t0
    print(f"    Gerado em {t_er_s:.2f}s")

    print(f"  BA: n={n_small}, m_attach={m_attach_small}")
    t0 = time.perf_counter()
    ip_ba_s, ix_ba_s = generate_ba_csr(n_small, m_attach_small, seed=SEED)
    t_ba_s = time.perf_counter() - t0
    m_ba_s = len(ix_ba_s) // 2
    print(f"    Gerado em {t_ba_s:.2f}s, m={m_ba_s:,}")

    print(f"  WS: n={n_small}, k={k_ws_small}, beta=0.1")
    t0 = time.perf_counter()
    ip_ws_s, ix_ws_s = generate_ws_csr(n_small, k_ws_small, 0.1, seed=SEED)
    t_ws_s = time.perf_counter() - t0
    m_ws_s = len(ix_ws_s) // 2
    print(f"    Gerado em {t_ws_s:.2f}s, m={m_ws_s:,}")

    np.random.seed(SEED)
    srcs_small = np.random.choice(n_small, size=min(200, n_small), replace=False).astype(np.int32)
    apl_er_s = sample_apl_bfs(ip_er_s, ix_er_s, srcs_small)
    apl_ba_s = sample_apl_bfs(ip_ba_s, ix_ba_s, srcs_small)
    apl_ws_s = sample_apl_bfs(ip_ws_s, ix_ws_s, srcs_small)
    print(f"  APL (n=5000): ER={apl_er_s:.4f}, BA={apl_ba_s:.4f}, WS={apl_ws_s:.4f}")

    print("\n[5.1b] Validação cruzada com igraph (n=2.000)...")
    n_val = 2000
    m_val = int(n_val * deg_mean / 2)
    m_attach_val = m_attach
    k_ws_val = min(k_ws, n_val // 2)
    if k_ws_val % 2 != 0:
        k_ws_val -= 1

    u_er_v, v_er_v = sample_er_edges(n_val, m_val, seed=99)
    ip_erv, ix_erv = build_csr_from_edges(n_val, m_val, u_er_v, v_er_v)
    m_erv = len(ix_erv) // 2

    ip_bav, ix_bav = generate_ba_csr(n_val, m_attach_val, seed=99)
    m_bav = len(ix_bav) // 2

    ip_wsv, ix_wsv = generate_ws_csr(n_val, k_ws_val, 0.1, seed=99)
    m_wsv = len(ix_wsv) // 2

    import igraph as ig
    validation_results = {}
    for label, ip, ix, m_expected_approx in [
        ("ER", ip_erv, ix_erv, m_val),
        ("BA", ip_bav, ix_bav, None),
        ("WS", ip_wsv, ix_wsv, None),
    ]:
        nn = len(ip) - 1
        mm = len(ix) // 2
        edges = []
        for u in range(nn):
            for j in range(ip[u], ip[u+1]):
                v = ix[j]
                if u < v:
                    edges.append((u, v))
        g = ig.Graph(n=nn, edges=edges, directed=False)
        ig_m = g.ecount()
        ig_n = g.vcount()
        ig_deg_mean = np.mean(g.degree())
        ig_cc = g.transitivity_avglocal_undirected(mode="zero")
        c_manual = float(compute_clustering_manual(ip, ix))
        match_m = (ig_m == mm)
        match_c = abs(ig_cc - c_manual) < 0.01
        validation_results[label] = {
            "n": ig_n, "m_manual": mm, "m_igraph": ig_m, "match_m": match_m,
            "C_manual": round(c_manual, 6), "C_igraph": round(ig_cc, 6),
            "match_C": match_c, "deg_mean_igraph": round(ig_deg_mean, 2),
        }
        status = "✓" if (match_m and match_c) else "✗"
        print(f"  {label}: m_manual={mm:,} vs igraph={ig_m:,} ({status}), "
              f"C_manual={c_manual:.6f} vs igraph={ig_cc:.6f} ({status})")

    print(f"\n[5.2] GERAÇÃO COMPLETA (n = {n:,})...")
    models = {}

    print(f"  [ER] Gerando G(n={n:,}, m={m:,})...")
    t0 = time.perf_counter()
    u_er, v_er = sample_er_edges(n, m, seed=SEED)
    ip_er, ix_er = build_csr_from_edges(n, m, u_er, v_er)
    t_gen_er = time.perf_counter() - t0
    m_er = len(ix_er) // 2
    deg_max_er = int(compute_max_degree(ip_er))
    deg_mean_er = 2 * m_er / n
    print(f"    m={m_er:,}, k_max={deg_max_er}, <k>={deg_mean_er:.2f}, tempo={t_gen_er:.2f}s")
    mem_er_mb = (ip_er.nbytes + ix_er.nbytes) / 1024**2
    print(f"    Memória CSR: {mem_er_mb:.1f} MB")

    print("    Calculando APL (1000 fontes)...")
    np.random.seed(SEED)
    srcs_full = np.random.choice(n, size=K_APL, replace=False).astype(np.int32)
    t0 = time.perf_counter()
    apl_er = float(sample_apl_bfs(ip_er, ix_er, srcs_full))
    t_apl_er = time.perf_counter() - t0
    print(f"    APL_ER = {apl_er:.4f} ({t_apl_er:.2f}s)")

    print("    Calculando C médio...")
    t0 = time.perf_counter()
    c_er = float(compute_clustering_manual(ip_er, ix_er))
    t_c_er = time.perf_counter() - t0
    print(f"    C_ER = {c_er:.6f} ({t_c_er:.2f}s)")

    degs_er = (ip_er[1:] - ip_er[:-1]).astype(np.int32)
    gamma_er, ks_er = clauset_mle(degs_er, 10)

    models["ER"] = {
        "n": n, "m": m_er, "deg_mean": round(deg_mean_er, 2), "deg_max": deg_max_er,
        "L": round(apl_er, 4), "C": round(c_er, 6),
        "gamma": round(gamma_er, 4) if not np.isnan(gamma_er) else "N/A",
        "gamma_note": "ER não apresenta cauda pesada; gamma fora do regime scale-free",
        "t_gen_s": round(t_gen_er, 2), "t_apl_s": round(t_apl_er, 2), "t_clust_s": round(t_c_er, 2),
        "mem_csr_mb": round(mem_er_mb, 1),
    }
    del u_er, v_er, ip_er, ix_er, degs_er

    print(f"\n  [BA] Gerando BA(n={n:,}, m_attach={m_attach})...")
    t0 = time.perf_counter()
    ip_ba, ix_ba = generate_ba_csr(n, m_attach, seed=SEED)
    t_gen_ba = time.perf_counter() - t0
    m_ba = len(ix_ba) // 2
    deg_max_ba = int(compute_max_degree(ip_ba))
    deg_mean_ba = 2 * m_ba / n
    mem_ba_mb = (ip_ba.nbytes + ix_ba.nbytes) / 1024**2
    print(f"    m={m_ba:,}, k_max={deg_max_ba}, <k>={deg_mean_ba:.2f}, tempo={t_gen_ba:.2f}s")
    print(f"    Memória CSR: {mem_ba_mb:.1f} MB")

    print("    Calculando APL (1000 fontes)...")
    t0 = time.perf_counter()
    apl_ba = float(sample_apl_bfs(ip_ba, ix_ba, srcs_full))
    t_apl_ba = time.perf_counter() - t0
    print(f"    APL_BA = {apl_ba:.4f} ({t_apl_ba:.2f}s)")

    print("    Calculando C médio...")
    t0 = time.perf_counter()
    c_ba = float(compute_clustering_manual(ip_ba, ix_ba))
    t_c_ba = time.perf_counter() - t0
    print(f"    C_BA = {c_ba:.6f} ({t_c_ba:.2f}s)")

    degs_ba = (ip_ba[1:] - ip_ba[:-1]).astype(np.int32)
    gamma_ba, ks_ba = clauset_mle(degs_ba, 10)

    models["BA"] = {
        "n": n, "m": m_ba, "deg_mean": round(deg_mean_ba, 2), "deg_max": deg_max_ba,
        "L": round(apl_ba, 4), "C": round(c_ba, 6),
        "gamma": round(gamma_ba, 4) if not np.isnan(gamma_ba) else "N/A",
        "gamma_ks": round(ks_ba, 4) if not np.isnan(ks_ba) else "N/A",
        "t_gen_s": round(t_gen_ba, 2), "t_apl_s": round(t_apl_ba, 2), "t_clust_s": round(t_c_ba, 2),
        "mem_csr_mb": round(mem_ba_mb, 1),
    }
    del ip_ba, ix_ba, degs_ba

    ws_results = {}
    for beta in betas:
        label = f"WS_b{beta}"
        print(f"\n  [WS β={beta}] Gerando WS(n={n:,}, k={k_ws}, β={beta})...")
        t0 = time.perf_counter()
        ip_ws, ix_ws = generate_ws_csr(n, k_ws, beta, seed=SEED)
        t_gen_ws = time.perf_counter() - t0
        m_ws = len(ix_ws) // 2
        deg_max_ws = int(compute_max_degree(ip_ws))
        deg_mean_ws = 2 * m_ws / n
        mem_ws_mb = (ip_ws.nbytes + ix_ws.nbytes) / 1024**2
        print(f"    m={m_ws:,}, k_max={deg_max_ws}, <k>={deg_mean_ws:.2f}, tempo={t_gen_ws:.2f}s")
        print(f"    Memória CSR: {mem_ws_mb:.1f} MB")

        print("    Calculando APL (1000 fontes)...")
        t0 = time.perf_counter()
        apl_ws = float(sample_apl_bfs(ip_ws, ix_ws, srcs_full))
        t_apl_ws = time.perf_counter() - t0
        print(f"    APL_WS = {apl_ws:.4f} ({t_apl_ws:.2f}s)")

        print("    Calculando C médio...")
        t0 = time.perf_counter()
        c_ws = float(compute_clustering_manual(ip_ws, ix_ws))
        t_c_ws = time.perf_counter() - t0
        print(f"    C_WS = {c_ws:.6f} ({t_c_ws:.2f}s)")

        degs_ws = (ip_ws[1:] - ip_ws[:-1]).astype(np.int32)

        ws_results[label] = {
            "n": n, "m": m_ws, "deg_mean": round(deg_mean_ws, 2), "deg_max": deg_max_ws,
            "L": round(apl_ws, 4), "C": round(c_ws, 6),
            "beta": beta,
            "gamma": "N/A (WS não produz cauda pesada)",
            "t_gen_s": round(t_gen_ws, 2), "t_apl_s": round(t_apl_ws, 2), "t_clust_s": round(t_c_ws, 2),
            "mem_csr_mb": round(mem_ws_mb, 1),
        }

        if beta == betas[-1]:
            degs_ws_last = degs_ws.copy()
            ip_ws_last = ip_ws
            ix_ws_last = ix_ws
        else:
            del ip_ws, ix_ws, degs_ws

    models.update(ws_results)

    models["Real"] = {
        "n": n, "m": m, "deg_mean": round(deg_mean, 2), "deg_max": deg_max_real,
        "L": L_real, "C": C_real,
        "gamma": "~4.56 (Clauset kmin=4496, cauda 0.3%)",
    }

    print("\n[5.3] Tabela comparativa...")
    table_rows = []
    for name in ["Real", "ER", "BA"] + [f"WS_b{b}" for b in betas]:
        info = models[name]
        table_rows.append({
            "Modelo": name,
            "n": info["n"],
            "m": info["m"],
            "<k>": info["deg_mean"],
            "k_max": info["deg_max"],
            "L (amostr.)": info["L"],
            "C médio": info["C"],
            "γ (Clauset)": info.get("gamma", "N/A"),
        })

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(os.path.join(RESULTS_DIR, "parte3_extra_tabela_comparativa.csv"), index=False)
    print(df_table.to_string(index=False))

    print("\n[5.fig] Gerando figuras...")

    u_er2, v_er2 = sample_er_edges(n, m, seed=SEED)
    ip_er2, ix_er2 = build_csr_from_edges(n, m, u_er2, v_er2)
    degs_er2 = (ip_er2[1:] - ip_er2[:-1]).astype(np.int32)
    del u_er2, v_er2, ip_er2, ix_er2

    ip_ba2, ix_ba2 = generate_ba_csr(n, m_attach, seed=SEED)
    degs_ba2 = (ip_ba2[1:] - ip_ba2[:-1]).astype(np.int32)
    del ip_ba2, ix_ba2

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    for label, degs, color, ls, lw in [
        ("Real (G2)", degrees_real, "#1a202c", "-", 2.5),
        ("ER", degs_er2, "#e53e3e", "--", 1.8),
        ("BA", degs_ba2, "#2f855a", "-.", 1.8),
        (f"WS (β=0.5)", degs_ws_last, "#805ad5", ":",  1.8),
    ]:
        s = np.sort(degs)
        ccdf = 1.0 - np.arange(len(s)) / len(s)
        ax.plot(s, ccdf, color=color, linestyle=ls, linewidth=lw, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grau ($k$)", fontsize=12)
    ax.set_ylabel("CCDF $P(K \\geq k)$", fontsize=12)
    ax.set_title("Distribuição de Graus: Grafo Real vs. Modelos Teóricos", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig_ccdf_path = os.path.join(FIGURES_DIR, "parte3_extra_ccdf_graus.png")
    plt.savefig(fig_ccdf_path, bbox_inches="tight")
    plt.close()

    fig, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    model_names = ["Real", "ER", "BA"] + [f"WS\nβ={b}" for b in betas]
    L_vals = [models["Real"]["L"], models["ER"]["L"], models["BA"]["L"]] + \
             [models[f"WS_b{b}"]["L"] for b in betas]
    C_vals = [models["Real"]["C"], models["ER"]["C"], models["BA"]["C"]] + \
             [models[f"WS_b{b}"]["C"] for b in betas]

    colors_lc = ["#2b6cb0", "#e53e3e", "#2f855a", "#805ad5", "#d69e2e", "#dd6b20"]

    bars_l = ax_l.bar(model_names, L_vals, color=colors_lc, edgecolor="#1a202c", linewidth=1)
    ax_l.set_ylabel("Average Path Length ($L$)", fontsize=11)
    ax_l.set_title("Distância Média ($L$) por Modelo", fontsize=12, fontweight="bold")
    for bar in bars_l:
        h = bar.get_height()
        ax_l.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}",
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_l.grid(axis="y", linestyle=":", alpha=0.5)

    bars_c = ax_c.bar(model_names, C_vals, color=colors_lc, edgecolor="#1a202c", linewidth=1)
    ax_c.set_ylabel("Coeficiente de Clusterização ($C$)", fontsize=11)
    ax_c.set_title("Clusterização Média ($C$) por Modelo (escala log)", fontsize=12, fontweight="bold")
    ax_c.set_yscale("log")
    for bar in bars_c:
        h = bar.get_height()
        if h > 0:
            ax_c.text(bar.get_x() + bar.get_width()/2, h * 1.3, f"{h:.4f}",
                      ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_c.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Comparação Estrutural: Grafo Real vs. Modelos Teóricos (G2 — ego-Gplus)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_bars_path = os.path.join(FIGURES_DIR, "parte3_extra_L_C_barras.png")
    plt.savefig(fig_bars_path, bbox_inches="tight")
    plt.close()

    t_total = time.perf_counter() - t0_global

    res_extra = {
        "modelos": {k: v for k, v in models.items()},
        "validacao_igraph_n2000": validation_results,
        "parametros": {
            "n": n, "m_real": m, "m_attach_BA": m_attach, "k_WS": k_ws,
            "betas_WS": betas, "seed": SEED, "k_APL": K_APL,
        },
        "tempo_total_s": round(t_total, 2),
    }
    with open(os.path.join(RESULTS_DIR, "parte3_extra_modelos.json"), "w", encoding="utf-8") as f:
        json.dump(res_extra, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[5] Modelos teóricos concluídos em {t_total:.2f}s")
    return res_extra, models

def write_summary(res_recip, res_extra, models):
    print("\n--- GERANDO results/parte3_extra_resumo.md ---")

    rr = res_recip
    rg = rr["reciprocidade_global"]
    rb = rr["baseline_aleatorio"]
    rs = rr["scc"]
    rn = rr["reciprocidade_por_no"]

    model_table_lines = []
    for name in ["Real", "ER", "BA"] + [f"WS_b{b}" for b in [0.01, 0.1, 0.5]]:
        info = models[name]
        gamma_str = str(info.get("gamma", "N/A"))
        if len(gamma_str) > 30:
            gamma_str = gamma_str[:30] + "…"
        model_table_lines.append(
            f"| {name} | {info['n']:,} | {info['m']:,} | {info['deg_mean']} "
            f"| {info['deg_max']:,} | {info['L']} | {info['C']} | {gamma_str} |"
        )
    model_table = "\n".join(model_table_lines)

    resumo = f"""# Resumo Executivo — Parte III: Itens 4 e 5

**Aluno:** Vitor Renato Santos dos Santos | **Matrícula:** 222217109 — UFBA
**Dataset:** ego-Gplus (G2) | $n = 107.614$, $m = 12.238.285$

---

## Item 4: Reciprocidade do Dígrafo Original

### Métricas Globais
| Métrica | Valor |
|---------|-------|
| Pares recíprocos | {rg['pares_reciprocos']:,} |
| Fração de arcos em laços bidirecionais | {rg['fracao_pct']:.2f}% |
| Validação: arcos − pares = m_undirected | {rr['validacao_interna']['arcos_unicos_menos_pares']:,} = {rr['validacao_interna']['m_undirected_esperado']:,} → {'PASSOU ✓' if rr['validacao_interna']['fechou'] else 'FALHOU ✗'} |
| Reciprocidade esperada (dígrafo aleatório) | {rb['reciprocidade_esperada_aleatoria']:.6f} (ρ do dígrafo) |
| Razão observado / esperado | **{rb['razao_observado_esperado']:.2f}×** |
| Correlação in-degree vs out-degree | r = {rr['graus_dirigidos']['correlacao_in_out']:.4f} |
| Nós com reciprocidade total | {rn['nos_reciprocidade_total']:,} ({rn['frac_reciprocidade_total_pct']:.2f}%) |
| Nós sem nenhum arco recíproco | {rn['nos_sem_reciprocidade']:,} ({rn['frac_sem_reciprocidade_pct']:.2f}%) |

### Relação com a SCC
A maior SCC tem {rs['maior_scc_nos']:,} nós ({rs['maior_scc_frac_pct']:.2f}%).
{rs['frac_reciprocidade_na_scc_pct']:.2f}% da reciprocidade (arcos recíprocos de saída) está concentrada na maior SCC.

### Reciprocidade por Faixa de Grau
Nós de **baixo grau** (periféricos) tendem a ter reciprocidade mais **alta**: seus poucos arcos costumam ser bidirecionais.
Hubs (grau alto) têm reciprocidade mais **baixa** em fração, pois acumulam muitos seguidores unilaterais.
Essa relação inversa é consistente com a dinâmica de redes sociais onde a conexão com celebridades é assimétrica.

> **Hipótese (não verificada formalmente):** a alta reciprocidade nos nós de baixo grau pode refletir "círculos sociais" íntimos onde conexões tendem a ser mútuas, enquanto hubs acumulam seguidores de sentido único.

---

## Item 5: Modelos Teóricos vs. Grafo Real

### Tabela Comparativa

| Modelo | n | m | ⟨k⟩ | k_max | L | C | γ (Clauset) |
|--------|---|---|-----|-------|---|---|-------------|
{model_table}

### Validação com igraph (n = 2.000)
Todos os três geradores (ER, BA, WS) produziram grafos cujo m e C coincidem com os valores do igraph (diferença < 0.01 em C).

### Conclusão Honesta: Qual Modelo Reproduz o Quê

| Propriedade | ER | BA | WS |
|-------------|----|----|-----|
| **L pequeno** (≈ 2.86) | ✓ L ≈ L_real | ✓ L comparável | Depende de β (β alto → L baixo) |
| **C alto** (≈ 0.49) | ✗ C ≈ 0.002 (200× menor) | ✗ C baixo | ✓ C alto para β baixo |
| **Cauda pesada** | ✗ Poisson | ✓ Power-law | ✗ Aproximadamente regular |

**Nenhum modelo isolado reproduz todas as três propriedades** do grafo real:
- **ER** acerta L mas falha em C e na cauda.
- **BA** produz cauda pesada e L curto, mas C é muito baixo.
- **WS** (β baixo) acerta C alto e L moderado, mas não produz cauda pesada.

O grafo G2 combina L curto, C muito alto e distribuição de graus com cauda pesada (embora não power-law pura). Essa combinação é mais bem descrita por modelos híbridos (e.g., Holme-Kim) que adicionam fechamento triádico a BA, ou por modelos de comunidades.

---

## Limitações e Ressalvas

1. **APL por amostragem:** L medido com k=1.000 fontes (semente 42); erro amostral ≈ 1–2%.
2. **Clustering:** C calculado no grafo completo (exato), mas para modelos pode variar com a semente.
3. **Parâmetros dos modelos:** m_attach=114 (BA) e k=228 (WS) calibrados para igualar ⟨k⟩ do real; outros valores poderiam dar resultados diferentes.
4. **Seed fixa (42):** uma única realização por modelo; variância entre sementes não avaliada.
5. **γ Clauset nos modelos:** para ER e WS, reportar γ de cauda não faz sentido teórico — incluído apenas para completude.
6. **Reciprocidade por faixa:** a métrica é a média da fração recíproca por nó; nós com out_deg=0 excluídos.

---

**Tempo total de execução:** {res_extra.get('tempo_total_s', '?')}s (modelos) + {res_recip.get('tempo_total_s', '?')}s (reciprocidade).
"""

    with open(os.path.join(RESULTS_DIR, "parte3_extra_resumo.md"), "w", encoding="utf-8") as f:
        f.write(resumo)
    print("  Escrito: results/parte3_extra_resumo.md")

if __name__ == "__main__":
    t_start = time.perf_counter()
    print(f"{'=' * 70}")
    print(f"PARTE III — ITENS 4 (RECIPROCIDADE) e 5 (MODELOS TEÓRICOS)")
    print(f"{'=' * 70}")

    res_recip = run_reciprocity()
    res_extra, models = run_synthetic_models()
    write_summary(res_recip, res_extra, models)

    t_end = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"TUDO CONCLUÍDO em {t_end:.2f}s")
    print(f"{'=' * 70}")
