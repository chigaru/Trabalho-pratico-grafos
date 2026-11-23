import os
import sys
import json
import time
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from scipy.sparse import csr_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("scripts"))
from src.graph_loader import load_csr_graph
from src.algorithms_numba import (
    generate_symmetric_weights_numba,
    bfs_distances_csr,
    dfs_iterative_csr,
    check_eulerian_numba,
    dijkstra_csr,
    bellman_ford_csr,
    floyd_warshall_csr,
    tarjan_scc_iterative,
    kruskal_mst,
)
from run_parte2_benchmarks import get_bfs_order, extract_subgraph_csr, fit_complexity_curve

SIZES = [500, 1000, 2000, 3000, 5000]
BF_SIZES = [500, 1000, 2000, 3000]
SEED = 42

def ci95(times_ms):
    x = np.asarray(times_ms, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    if n >= 30:
        crit, kind = float(stats.norm.ppf(0.975)), "Z"
    else:
        crit, kind = float(stats.t.ppf(0.975, n - 1)), "t-Student"
    margin = crit * std / np.sqrt(n)
    return {"mean": mean, "std": std, "margin": float(margin), "n": n, "stat": kind}

def time_sources(fn, sources):
    fn(int(sources[0]))
    ts = []
    for s in sources:
        t0 = time.perf_counter()
        fn(int(s))
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts

def time_reps(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts

def row(alg, n, m, ci, theory):
    return {
        "Algoritmo": alg,
        "Tamanho (n)": int(n),
        "Arestas (m)": int(m),
        "Repetições": ci["n"],
        "Estatística": ci["stat"],
        "Tempo Médio (ms)": ci["mean"],
        "Desvio Padrão (ms)": ci["std"],
        "Margem IC 95% (ms)": ci["margin"],
        "Tempo (Média ± IC 95%)": f"{ci['mean']:.3f} ± {ci['margin']:.3f} ms",
        "Complexidade Teórica": theory,
    }

@njit
def bellman_ford_rodadas_fixas(indptr, indices, weights, source):
    N = len(indptr) - 1
    INF = np.int64(10 ** 15)
    dist = np.full(N, INF, dtype=np.int64)
    dist[source] = 0
    for r in range(N - 1):
        for u in range(N):
            d_u = dist[u]
            if d_u == INF:
                continue
            for i in range(indptr[u], indptr[u + 1]):
                v = indices[i]
                w = weights[i]
                if d_u + w < dist[v]:
                    dist[v] = d_u + w
    return dist

def main():
    t_start = time.perf_counter()
    print("=== EXTRAS DA PARTE II ===")

    csr_undir, _ = load_csr_graph("data/processed/gplus_undirected_gcc.npz")
    indptr, indices = csr_undir.indptr, csr_undir.indices
    N = len(indptr) - 1
    M = len(indices) // 2
    d = np.load("data/processed/gplus_directed.npz")
    d_indptr, d_indices = d["indptr"], d["indices"]
    weights = generate_symmetric_weights_numba(indptr, indices, seed=SEED)
    print(f"Grafo: n={N:,}, m={M:,} | digrafo: {len(d_indices):,} arcos")

    bfs_order = get_bfs_order(indptr, indices, 0)
    A_dir = csr_matrix((np.ones(len(d_indices), dtype=np.int8), d_indices, d_indptr), shape=(N, N))

    subs = {}
    for n in SIZES:
        s_ip, s_ix, s_w, s_m = extract_subgraph_csr(bfs_order, n, indptr, indices, weights)
        nodes = np.asarray(bfs_order[:n])
        dsub = A_dir[nodes][:, nodes]
        subs[n] = {
            "indptr": s_ip, "indices": s_ix, "weights": s_w, "m": s_m,
            "d_indptr": np.ascontiguousarray(dsub.indptr, dtype=np.int32),
            "d_indices": np.ascontiguousarray(dsub.indices, dtype=np.int32),
            "d_m": int(dsub.nnz),
        }
        print(f"  subgrafo n={n}: m={s_m:,} | arcos={dsub.nnz:,}")

    rng = np.random.default_rng(SEED)
    rows = []
    extras = {}

    print("\n[1] Bellman-Ford com e sem parada antecipada")
    bf = []
    for n in BF_SIZES:
        if n not in subs:
            continue
        sub = subs[n]
        reps = 30 if n < 2000 else 10
        srcs = rng.integers(0, n, size=reps)
        d_es, rounds = bellman_ford_csr(sub["indptr"], sub["indices"], sub["weights"], int(srcs[0]))
        d_full = bellman_ford_rodadas_fixas(sub["indptr"], sub["indices"], sub["weights"], int(srcs[0]))
        assert np.array_equal(d_es, d_full), "Bellman-Ford: resultados diferentes!"
        t_es = time_sources(lambda s: bellman_ford_csr(sub["indptr"], sub["indices"], sub["weights"], s), srcs)
        rounds_list = [bellman_ford_csr(sub["indptr"], sub["indices"], sub["weights"], int(s))[1] for s in srcs]
        t_full = time_sources(lambda s: bellman_ford_rodadas_fixas(sub["indptr"], sub["indices"], sub["weights"], s), srcs)
        ci_es, ci_full = ci95(t_es), ci95(t_full)
        bf.append({"n": n, "m": sub["m"], "reps": reps, "rounds_medio": float(np.mean(rounds_list)),
                   "early": ci_es, "full": ci_full})
        print(f"  n={n}: parada antecipada {ci_es['mean']:.2f} ± {ci_es['margin']:.2f} ms "
              f"(rodadas={np.mean(rounds_list):.1f}) | n-1 rodadas {ci_full['mean']:.1f} ± {ci_full['margin']:.1f} ms "
              f"[{ci_full['stat']}, {reps} reps]")
        rows.append(row("Bellman-Ford (parada antecipada)", n, sub["m"], ci_es, "O(rodadas · m) ≤ O(n · m)"))
        rows.append(row("Bellman-Ford (n-1 rodadas)", n, sub["m"], ci_full, "O(n · m)"))
    extras["bellman_ford"] = bf

    print("\n[2] Curvas de escala (subgrafos + grafo completo)")
    scale = {k: [] for k in ["BFS", "DFS", "Dijkstra", "Tarjan", "Eulerianidade"]}
    configs = [(n, subs[n]["indptr"], subs[n]["indices"], subs[n]["weights"], subs[n]["m"],
                subs[n]["d_indptr"], subs[n]["d_indices"], subs[n]["d_m"], "Subgrafo") for n in SIZES]
    configs.append((N, indptr, indices, weights, M, d_indptr, d_indices, len(d_indices), "Grafo Completo"))

    for n, ip, ix, w, m, dip, dix, dm, tag in configs:
        srcs = rng.integers(0, n, size=30)
        c = {
            "BFS": ci95(time_sources(lambda s: bfs_distances_csr(ip, ix, s), srcs)),
            "DFS": ci95(time_sources(lambda s: dfs_iterative_csr(ip, ix, s), srcs)),
            "Dijkstra": ci95(time_sources(lambda s: dijkstra_csr(ip, ix, w, s), srcs)),
            "Tarjan": ci95(time_reps(lambda: tarjan_scc_iterative(dip, dix), 30)),
            "Eulerianidade": ci95(time_reps(lambda: check_eulerian_numba(ip, ix), 30)),
        }
        theory = {"BFS": "O(n + m)", "DFS": "O(n + m)", "Dijkstra": "O(m log n)",
                  "Tarjan": "O(n + m)", "Eulerianidade": "O(n + m)"}
        for k, ci in c.items():
            mm = dm if k == "Tarjan" else m
            scale[k].append({"n": n, "m": mm, "ci": ci, "tag": tag})
            rows.append(row(f"{k} ({tag})", n, mm, ci, theory[k]))
        print(f"  n={n:>7,} ({tag}): BFS {c['BFS']['mean']:.3f} | DFS {c['DFS']['mean']:.3f} | "
              f"Dijkstra {c['Dijkstra']['mean']:.3f} | Tarjan {c['Tarjan']['mean']:.3f} | "
              f"Euler {c['Eulerianidade']['mean']:.3f}  (ms)")
    extras["escala"] = {k: [{"n": e["n"], "m": e["m"], **e["ci"]} for e in v] for k, v in scale.items()}

    print("\n[3] Repeticoes ampliadas (Floyd-Warshall e Kruskal completo)")
    fw_rows = []
    for n in [600, 800]:
        s_ip, s_ix, s_w, s_m = extract_subgraph_csr(bfs_order, n, indptr, indices, weights)
        ci = ci95(time_reps(lambda: floyd_warshall_csr(s_ip, s_ix, s_w), 30))
        rows.append(row("Floyd-Warshall", n, s_m, ci, "O(n³)"))
        fw_rows.append({"n": n, "m": s_m, **ci})
        print(f"  Floyd-Warshall n={n}: {ci['mean']:.2f} ± {ci['margin']:.2f} ms [{ci['stat']}, 30 reps]")
    extras["floyd_warshall_30reps"] = fw_rows

    src_full = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
    mask = src_full < indices
    full_u = src_full[mask].astype(np.int32)
    full_v = indices[mask].astype(np.int32)
    full_w = weights[mask].astype(np.int32)
    del src_full, mask
    w_tot, n_edges = kruskal_mst(N, full_u, full_v, full_w)
    ci = ci95(time_reps(lambda: kruskal_mst(N, full_u, full_v, full_w), 30))
    rows.append(row("Kruskal (Grafo Completo)", N, M, ci, "O(m · α(n)) (counting sort, pesos 1..10)"))
    extras["kruskal_completo_30reps"] = {"peso_mst": int(w_tot), "arestas_mst": int(n_edges), **ci}
    print(f"  Kruskal completo: peso={int(w_tot):,}, arestas={int(n_edges):,}, "
          f"{ci['mean']:.2f} ± {ci['margin']:.2f} ms [{ci['stat']}, 30 reps]")

    df_new = pd.DataFrame(rows)
    df_new.to_csv("results/parte2_extras_tabela.csv", index=False, encoding="utf-8-sig")

    final = df_new
    old_path = "results/parte2_tempos_tabela.csv"
    if os.path.exists(old_path):
        old = pd.read_csv(old_path)
        def std_from_margin(r):
            reps = int(r["Repetições"])
            crit = stats.norm.ppf(0.975) if r["Estatística"] == "Z" else stats.t.ppf(0.975, reps - 1)
            return r["Margem IC 95% (ms)"] * np.sqrt(reps) / crit
        old["Desvio Padrão (ms)"] = old.apply(std_from_margin, axis=1)
        final = pd.concat([old, df_new], ignore_index=True)
        final = final.drop_duplicates(subset=["Algoritmo", "Tamanho (n)"], keep="last")
    final = final.sort_values(["Algoritmo", "Tamanho (n)"]).reset_index(drop=True)
    final.to_csv("results/parte2_tempos_tabela_final.csv", index=False, encoding="utf-8-sig")

    with open("results/parte2_extras.json", "w", encoding="utf-8") as f:
        json.dump(extras, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=150)
    axes = axes.ravel()
    spec = [("BFS", "O(n + m)", lambda n, m: n + m),
            ("DFS", "O(n + m)", lambda n, m: n + m),
            ("Dijkstra", "O(m log n)", lambda n, m: m * np.log2(n)),
            ("Tarjan", "O(n + m)", lambda n, m: n + m),
            ("Eulerianidade", "O(n + m)", lambda n, m: n + m)]
    slopes = {}
    for ax, (name, label, fx) in zip(axes, spec):
        pts = scale[name]
        x = np.array([fx(e["n"], e["m"]) for e in pts], dtype=float)
        y = np.array([e["ci"]["mean"] for e in pts])
        err = np.array([e["ci"]["margin"] for e in pts])
        c, r2, _ = fit_complexity_curve(x, y, lambda v: v)
        slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
        slopes[name] = slope
        ax.errorbar(x, y, yerr=err, fmt="o", capsize=4, color="#2b6cb0", ecolor="#e53e3e",
                    label="Observado (média ± IC 95%)")
        xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        ax.plot(xs, c * xs, "--", color="#2c7a7b", label=f"c·f(n,m), c={c:.2e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{name}: {label}\ninclinação log-log = {slope:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("f(n, m)")
        ax.set_ylabel("Tempo (ms)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig("figures/parte2_escala_algoritmos.png", bbox_inches="tight")
    plt.close()
    extras["inclinacoes_loglog"] = slopes
    print("\n  Salvo: figures/parte2_escala_algoritmos.png | inclinações:", {k: round(v, 2) for k, v in slopes.items()})

    if bf:
        ns = [e["n"] for e in bf]
        ms = [e["m"] for e in bf]
        y_es = [e["early"]["mean"] for e in bf]
        e_es = [e["early"]["margin"] for e in bf]
        y_f = [e["full"]["mean"] for e in bf]
        e_f = [e["full"]["margin"] for e in bf]
        nm = np.array([n * m for n, m in zip(ns, ms)], dtype=float)
        c_f, r2_f, _ = fit_complexity_curve(nm, y_f, lambda v: v)
        rm = np.array([e["rounds_medio"] * m for e, m in zip(bf, ms)], dtype=float)
        c_e, r2_e, _ = fit_complexity_curve(rm, y_es, lambda v: v)
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=170)
        ax.errorbar(ns, y_f, yerr=e_f, fmt="s", capsize=4, color="#c53030", label="n−1 rodadas (observado)")
        ax.plot(ns, c_f * nm, "--", color="#c53030", alpha=0.7, label=f"c·n·m (R²={r2_f:.3f})")
        ax.errorbar(ns, y_es, yerr=e_es, fmt="o", capsize=4, color="#2b6cb0", label="parada antecipada (observado)")
        ax.plot(ns, c_e * rm, "--", color="#2b6cb0", alpha=0.7, label=f"c·rodadas·m (R²={r2_e:.3f})")
        ax.set_yscale("log")
        ax.set_xlabel("Ordem do subgrafo (n)")
        ax.set_ylabel("Tempo (ms, escala log)")
        ax.set_title("Bellman-Ford: pior caso O(n·m) × parada antecipada", fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig("figures/parte2_bellman_ford_comparacao.png", bbox_inches="tight")
        plt.close()
        print("  Salvo: figures/parte2_bellman_ford_comparacao.png")

    sel = final.copy()
    sel["rotulo"] = sel.apply(lambda r: f"{r['Algoritmo']} (n={int(r['Tamanho (n)']):,})", axis=1)
    sel = sel.sort_values("Tempo Médio (ms)")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(sel))), dpi=150)
    ax.barh(sel["rotulo"], sel["Tempo Médio (ms)"], xerr=sel["Margem IC 95% (ms)"],
            color="#4a90d9", edgecolor="#2d3748", capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Tempo médio (ms, escala log) ± IC 95%")
    ax.set_title("Tempos de execução de todos os algoritmos (tamanhos de entrada diferentes)", fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/parte2_resumo_tempos_log.png", bbox_inches="tight")
    plt.close()
    print("  Salvo: figures/parte2_resumo_tempos_log.png")

    with open("results/parte2_extras.json", "w", encoding="utf-8") as f:
        json.dump(extras, f, indent=2, ensure_ascii=False)
    print(f"\n=== EXTRAS FINALIZADOS EM {time.perf_counter() - t_start:.1f}s ===")

if __name__ == "__main__":
    main()
