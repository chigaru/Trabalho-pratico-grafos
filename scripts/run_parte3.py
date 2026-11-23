import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numba import njit, prange

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
DATA_PATH = "data/processed/gplus_undirected_gcc.npz"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

@njit
def sample_er_edges(n: int, m: int, seed: int = 42):
    np.random.seed(seed)
    table_size = 33554432
    mask = table_size - 1
    table = np.zeros(table_size, dtype=np.int64)
    u_edges = np.empty(m, dtype=np.int32)
    v_edges = np.empty(m, dtype=np.int32)

    count = 0
    while count < m:
        u = np.random.randint(0, n)
        v = np.random.randint(0, n)
        if u == v:
            continue
        if u > v:
            tmp = u; u = v; v = tmp

        key = (np.int64(u) << 32) | np.int64(v)

        h = (key ^ (key >> 30)) * np.int64(0xbf58476d1ce4e5b9)
        h = (h ^ (h >> 27)) * np.int64(0x94d049bb133111eb)
        h = h ^ (h >> 31)
        slot = int(h & mask)

        while True:
            k = table[slot]
            if k == 0:
                table[slot] = key
                u_edges[count] = u
                v_edges[count] = v
                count += 1
                break
            elif k == key:
                break
            slot = (slot + 1) & mask

    return u_edges, v_edges

@njit
def build_csr_from_edges(n: int, m: int, u_edges: np.ndarray, v_edges: np.ndarray):
    deg = np.zeros(n, dtype=np.int32)
    for i in range(m):
        deg[u_edges[i]] += 1
        deg[v_edges[i]] += 1

    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]

    indices = np.empty(2 * m, dtype=np.int32)
    pos = np.copy(indptr[:-1])

    for i in range(m):
        u = u_edges[i]
        v = v_edges[i]
        indices[pos[u]] = v
        pos[u] += 1
        indices[pos[v]] = u
        pos[v] += 1

    for u in range(n):
        start = indptr[u]
        end = indptr[u + 1]
        if end - start > 1:
            indices[start:end].sort()

    return indptr, indices

@njit(parallel=True)
def compute_clustering_manual(indptr: np.ndarray, indices: np.ndarray):
    n = len(indptr) - 1
    c_per_node = np.zeros(n, dtype=np.float64)
    for u in prange(n):
        u_start = indptr[u]
        u_end = indptr[u + 1]
        deg_u = u_end - u_start
        if deg_u < 2:
            continue
        common = 0
        for i in range(u_start, u_end):
            v = indices[i]
            v_start = indptr[v]
            v_end = indptr[v + 1]
            p1 = u_start
            p2 = v_start
            while p1 < u_end and p2 < v_end:
                w1 = indices[p1]
                w2 = indices[p2]
                if w1 == w2:
                    common += 1
                    p1 += 1
                    p2 += 1
                elif w1 < w2:
                    p1 += 1
                else:
                    p2 += 1
        tri_u = common // 2
        c_per_node[u] = (2.0 * tri_u) / (deg_u * (deg_u - 1))
    return np.mean(c_per_node)

@njit
def sample_apl_bfs(indptr: np.ndarray, indices: np.ndarray, sources: np.ndarray):
    n = len(indptr) - 1
    k = len(sources)
    dist = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    apl_sum = 0.0
    valid_sources = 0

    for s_idx in range(k):
        s = sources[s_idx]
        dist.fill(-1)
        dist[s] = 0
        queue[0] = s
        head = 0
        tail = 1
        sum_d = np.int64(0)

        while head < tail:
            u = queue[head]
            head += 1
            d_next = dist[u] + 1
            start = indptr[u]
            end = indptr[u + 1]
            for i in range(start, end):
                v = indices[i]
                if dist[v] == -1:
                    dist[v] = d_next
                    queue[tail] = v
                    tail += 1
                    sum_d += d_next
        if tail > 1:
            apl_sum += sum_d / (tail - 1)
            valid_sources += 1

    return apl_sum / valid_sources if valid_sources > 0 else 0.0

@njit
def compute_components_masked(indptr: np.ndarray, indices: np.ndarray, removed: np.ndarray):
    n = len(indptr) - 1
    visited = np.zeros(n, dtype=np.bool_)
    gcc_mask = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)

    max_comp_size = 0
    max_comp_start = -1
    isolated_count = 0
    num_components = 0

    for u in range(n):
        if removed[u] or visited[u]:
            continue

        num_components += 1
        visited[u] = True
        queue[0] = u
        head = 0
        tail = 1

        while head < tail:
            curr = queue[head]
            head += 1
            start = indptr[curr]
            end = indptr[curr + 1]
            for i in range(start, end):
                v = indices[i]
                if not removed[v] and not visited[v]:
                    visited[v] = True
                    queue[tail] = v
                    tail += 1

        c_size = tail
        if c_size == 1:
            isolated_count += 1

        if c_size > max_comp_size:
            max_comp_size = c_size
            max_comp_start = u

    if max_comp_start != -1:
        visited_gcc = np.zeros(n, dtype=np.bool_)
        visited_gcc[max_comp_start] = True
        gcc_mask[max_comp_start] = True
        queue[0] = max_comp_start
        head = 0
        tail = 1
        while head < tail:
            curr = queue[head]
            head += 1
            start = indptr[curr]
            end = indptr[curr + 1]
            for i in range(start, end):
                v = indices[i]
                if not removed[v] and not visited_gcc[v]:
                    visited_gcc[v] = True
                    gcc_mask[v] = True
                    queue[tail] = v
                    tail += 1

    return num_components, max_comp_size, gcc_mask, isolated_count

@njit
def bfs_single_masked(indptr: np.ndarray, indices: np.ndarray, removed: np.ndarray, source: int):
    n = len(indptr) - 1
    dist = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)

    dist[source] = 0
    queue[0] = source
    head = 0
    tail = 1

    max_d = 0
    farthest = source
    sum_d = np.int64(0)

    while head < tail:
        u = queue[head]
        head += 1
        d_next = dist[u] + 1
        start = indptr[u]
        end = indptr[u + 1]
        for i in range(start, end):
            v = indices[i]
            if not removed[v] and dist[v] == -1:
                dist[v] = d_next
                queue[tail] = v
                tail += 1
                if d_next > max_d:
                    max_d = d_next
                    farthest = v
                sum_d += d_next

    return max_d, farthest, sum_d, tail

@njit
def double_sweep_masked(indptr: np.ndarray, indices: np.ndarray, removed: np.ndarray, start_node: int, sweeps: int = 5):
    curr = start_node
    max_diam = 0
    for _ in range(sweeps):
        d, farthest, _, _ = bfs_single_masked(indptr, indices, removed, curr)
        if d > max_diam:
            max_diam = d
        if farthest == curr:
            break
        curr = farthest
    return max_diam

@njit
def sample_apl_masked(indptr: np.ndarray, indices: np.ndarray, removed: np.ndarray, sources: np.ndarray):
    k = len(sources)
    apl_sum = 0.0
    valid = 0
    for s_idx in range(k):
        s = sources[s_idx]
        _, _, sum_d, count = bfs_single_masked(indptr, indices, removed, s)
        if count > 1:
            apl_sum += sum_d / (count - 1)
            valid += 1
    return apl_sum / valid if valid > 0 else 0.0

@njit
def brandes_approx_betweenness(
    indptr: np.ndarray,
    indices: np.ndarray,
    pivots: np.ndarray
) -> np.ndarray:
    n = len(indptr) - 1
    num_pivots = len(pivots)
    cb = np.zeros(n, dtype=np.float64)

    d = np.full(n, -1, dtype=np.int32)
    sigma = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    q = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)

    for p_idx in range(num_pivots):
        s = pivots[p_idx]
        d.fill(-1)
        sigma.fill(0.0)
        delta.fill(0.0)

        d[s] = 0
        sigma[s] = 1.0
        q[0] = s
        head = 0
        tail = 1
        stack_top = 0

        while head < tail:
            u = q[head]
            head += 1
            stack[stack_top] = u
            stack_top += 1

            d_u = d[u]
            sigma_u = sigma[u]
            start = indptr[u]
            end = indptr[u + 1]

            for i in range(start, end):
                v = indices[i]
                if d[v] == -1:
                    d[v] = d_u + 1
                    q[tail] = v
                    tail += 1
                if d[v] == d_u + 1:
                    sigma[v] += sigma_u

        while stack_top > 0:
            stack_top -= 1
            w = stack[stack_top]
            d_w = d[w]
            delta_w = delta[w]
            coeff = (1.0 + delta_w) / sigma[w]

            start = indptr[w]
            end = indptr[w + 1]
            for i in range(start, end):
                u = indices[i]
                if d[u] == d_w - 1:
                    delta[u] += sigma[u] * coeff

            if w != s:
                cb[w] += delta_w

    return cb

def run_all_parte3(force: bool = False):
    t_global_start = time.perf_counter()
    print("=====================================================================")
    print("=== INICIANDO EXECUÇÃO DA PARTE III — ANÁLISE ESTRUTURAL AVANÇADA ===")
    print("=====================================================================")

    print(f"\n[0/3] Carregando representação CSR do grafo: {DATA_PATH}...")
    t0 = time.perf_counter()
    csr_data = np.load(DATA_PATH)
    indptr = csr_data["indptr"]
    indices = csr_data["indices"]
    n = len(indptr) - 1
    m = int(len(indices) // 2)
    degrees = indptr[1:] - indptr[:-1]
    deg_mean = float(np.mean(degrees))
    print(f"  Grafo carregado em {time.perf_counter()-t0:.2f}s: n={n:,}, m={m:,}, <k>={deg_mean:.2f}")

    l_g = 2.8603
    c_g = 0.490118
    r = int(np.ceil(0.05 * n))

    sw_file = os.path.join(RESULTS_DIR, "parte3_small_world.json")
    if os.path.exists(sw_file) and not force:
        print(f"\n[1/3] Carregando resultados existentes de Small-World de {sw_file}...")
        with open(sw_file, "r", encoding="utf-8") as f:
            res_small_world = json.load(f)
        l_rand_theo = res_small_world["teorico"]["L_rand"]
        c_rand_theo = res_small_world["teorico"]["C_rand"]
        ratio_l_theo = res_small_world["teorico"]["ratio_L"]
        ratio_c_theo = res_small_world["teorico"]["ratio_C"]
        sigma_theo = res_small_world["teorico"]["sigma"]
        l_rand_emp = res_small_world["empirico"]["L_rand"]
        c_rand_emp = res_small_world["empirico"]["C_rand"]
        ratio_l_emp = res_small_world["empirico"]["ratio_L"]
        ratio_c_emp = res_small_world["empirico"]["ratio_C"]
        sigma_emp = res_small_world["empirico"]["sigma"]
    else:
        print("\n--- ITEM 1: ANÁLISE SMALL-WORLD (WATTS-STROGATZ) ---")
        l_rand_theo = np.log(n) / np.log(deg_mean)
        c_rand_theo = deg_mean / n
        ratio_l_theo = l_g / l_rand_theo
        ratio_c_theo = c_g / c_rand_theo
        sigma_theo = ratio_c_theo / ratio_l_theo

        print(f"  [Teórico] L_rand = {l_rand_theo:.4f} | C_rand = {c_rand_theo:.6f}")
        print(f"  [Teórico] L/L_rand = {ratio_l_theo:.4f} | C/C_rand = {ratio_c_theo:.2f} | Sigma = {sigma_theo:.2f}")

        print(f"  [Empírico] Gerando rede aleatória Erdős–Rényi G(n={n:,}, m={m:,}) em Numba...")
        t_er0 = time.perf_counter()
        u_er, v_er = sample_er_edges(n, m, seed=42)
        indptr_er, indices_er = build_csr_from_edges(n, m, u_er, v_er)
        t_er_gen = time.perf_counter() - t_er0
        print(f"  [Empírico] Grafo ER montado em {t_er_gen:.2f}s.")

        print("  [Empírico] Calculando clusterização média de G(n, m)...")
        t0 = time.perf_counter()
        c_rand_emp = float(compute_clustering_manual(indptr_er, indices_er))
        t_c_er = time.perf_counter() - t0
        print(f"  [Empírico] C_rand_emp = {c_rand_emp:.6f} (em {t_c_er:.2f}s)")

        print("  [Empírico] Calculando APL amostrado em G(n, m) (200 fontes, seed 42)...")
        np.random.seed(42)
        sources_er = np.random.choice(n, size=200, replace=False).astype(np.int32)
        t0 = time.perf_counter()
        l_rand_emp = float(sample_apl_bfs(indptr_er, indices_er, sources_er))
        t_apl_er = time.perf_counter() - t0
        print(f"  [Empírico] L_rand_emp = {l_rand_emp:.4f} (em {t_apl_er:.2f}s)")

        ratio_l_emp = l_g / l_rand_emp
        ratio_c_emp = c_g / c_rand_emp
        sigma_emp = ratio_c_emp / ratio_l_emp
        is_small_world = bool(ratio_l_emp < 1.5 and ratio_c_emp > 100 and sigma_emp > 50)

        res_small_world = {
            "grafo_real": {"n": n, "m": m, "grau_medio": deg_mean, "L_G": l_g, "C_G": c_g},
            "teorico": {
                "L_rand": round(float(l_rand_theo), 4),
                "C_rand": round(float(c_rand_theo), 6),
                "ratio_L": round(float(ratio_l_theo), 4),
                "ratio_C": round(float(ratio_c_theo), 2),
                "sigma": round(float(sigma_theo), 2)
            },
            "empirico": {
                "L_rand": round(float(l_rand_emp), 4),
                "C_rand": round(float(c_rand_emp), 6),
                "ratio_L": round(float(ratio_l_emp), 4),
                "ratio_C": round(float(ratio_c_emp), 2),
                "sigma": round(float(sigma_emp), 2),
                "tempo_geracao_er_s": round(t_er_gen, 2),
                "tempo_clustering_s": round(t_c_er, 2),
                "tempo_apl_s": round(t_apl_er, 2)
            },
            "conclusao_small_world": is_small_world,
            "implicacao_pratica": "Alta coesão comunitária local (círculos sociais densos) aliada a caminhos mínimos globais ultra-curtos (baixo diâmetro e baixo APL)."
        }
        with open(sw_file, "w", encoding="utf-8") as f:
            json.dump(res_small_world, f, indent=2, ensure_ascii=False)

    pl_file = os.path.join(RESULTS_DIR, "parte3_power_law.json")
    def fit_clauset_candidate(x_tail, k_cand):
        n_tail = len(x_tail)
        if n_tail < 50:
            return np.nan, 1.0
        gamma_hat = 1.0 + n_tail / np.sum(np.log(x_tail / (k_cand - 0.5)))
        sorted_x = np.sort(x_tail)
        emp_cdf = np.arange(1, n_tail + 1) / n_tail
        emp_cdf_prev = np.arange(0, n_tail) / n_tail
        model_cdf = 1.0 - (sorted_x / (k_cand - 0.5)) ** (-(gamma_hat - 1.0))
        model_cdf = np.clip(model_cdf, 0.0, 1.0)
        ks_val = np.max(np.maximum(np.abs(emp_cdf - model_cdf), np.abs(emp_cdf_prev - model_cdf)))
        return gamma_hat, ks_val

    candidate_kmins = np.unique(np.round(np.logspace(np.log10(10), np.log10(6000), 400)).astype(int))
    scan_results = []
    best_ks = 1.0
    best_kmin = 1
    best_gamma = 0.0
    best_ntail = 0
    for k_cand in candidate_kmins:
        x_tail = degrees[degrees >= k_cand]
        if len(x_tail) < 50:
            continue
        g_cand, ks_cand = fit_clauset_candidate(x_tail, k_cand)
        scan_results.append({
            "k_min": int(k_cand),
            "gamma": float(g_cand),
            "ks": float(ks_cand),
            "n_tail": int(len(x_tail)),
            "fraction_tail_pct": float(len(x_tail) / n * 100)
        })
        if ks_cand < best_ks:
            best_ks = ks_cand
            best_kmin = int(k_cand)
            best_gamma = float(g_cand)
            best_ntail = int(len(x_tail))

    if os.path.exists(pl_file) and not force:
        print(f"\n[2/3] Carregando resultados existentes de Lei de Potência de {pl_file}...")
        with open(pl_file, "r", encoding="utf-8") as f:
            res_power_law = json.load(f)
        eval_global = res_power_law["otimo_clauset"]
        eval_1500 = res_power_law["referencia_1500"]
        eval_500 = res_power_law["referencia_500"]
    else:
        print("\n--- ITEM 2: ANÁLISE DE LEI DE POTÊNCIA E AJUSTE DE MODELOS DE CAUDA ---")
        def evaluate_tail_models(k_threshold):
            x_tail = degrees[degrees >= k_threshold].astype(float)
            n_tail = len(x_tail)
            gamma_hat = 1.0 + n_tail / np.sum(np.log(x_tail / (k_threshold - 0.5)))
            ll_pl = float(n_tail * np.log((gamma_hat - 1.0) / (k_threshold - 0.5)) - gamma_hat * np.sum(np.log(x_tail / (k_threshold - 0.5))))
            aic_pl = float(2 * 1 - 2 * ll_pl)
            sorted_x = np.sort(x_tail)
            emp_cdf = np.arange(1, n_tail + 1) / n_tail
            cdf_pl = 1.0 - (sorted_x / (k_threshold - 0.5)) ** (-(gamma_hat - 1.0))
            ks_pl = float(np.max(np.abs(emp_cdf - np.clip(cdf_pl, 0, 1))))

            loc_exp = float(k_threshold)
            scale_exp = float(np.mean(x_tail - k_threshold))
            ll_exp = float(np.sum(stats.expon.logpdf(x_tail, loc=loc_exp, scale=scale_exp)))
            aic_exp = float(2 * 1 - 2 * ll_exp)
            cdf_exp = stats.expon.cdf(sorted_x, loc=loc_exp, scale=scale_exp)
            ks_exp = float(np.max(np.abs(emp_cdf - cdf_exp)))

            def neg_ll_ln(params):
                mu, s = params
                if s <= 0.01:
                    return 1e12
                surv = stats.norm.sf((np.log(k_threshold) - mu) / s)
                if surv <= 1e-12:
                    return 1e12
                log_pdf = stats.lognorm.logpdf(x_tail, s=s, scale=np.exp(mu))
                return -(np.sum(log_pdf) - n_tail * np.log(surv))

            init_mu = float(np.mean(np.log(x_tail)))
            init_s = float(np.std(np.log(x_tail)))
            res_ln = opt.minimize(neg_ll_ln, [init_mu, init_s], method="Nelder-Mead")
            mu_ln, s_ln = float(res_ln.x[0]), float(res_ln.x[1])
            surv_opt = stats.norm.sf((np.log(k_threshold) - mu_ln) / s_ln)
            ll_ln = float(np.sum(stats.lognorm.logpdf(x_tail, s=s_ln, scale=np.exp(mu_ln))) - n_tail * np.log(surv_opt))
            aic_ln = float(2 * 2 - 2 * ll_ln)
            cdf_ln = (stats.lognorm.cdf(sorted_x, s=s_ln, scale=np.exp(mu_ln)) - (1.0 - surv_opt)) / surv_opt
            ks_ln = float(np.max(np.abs(emp_cdf - np.clip(cdf_ln, 0, 1))))

            return {
                "k_min": int(k_threshold),
                "n_tail": int(n_tail),
                "fraction_tail_pct": round(float(n_tail / n * 100), 2),
                "power_law": {
                    "gamma": round(float(gamma_hat), 4),
                    "ks": round(float(ks_pl), 4),
                    "log_likelihood": round(ll_pl, 2),
                    "aic": round(aic_pl, 2)
                },
                "exponencial": {
                    "scale": round(scale_exp, 2),
                    "ks": round(float(ks_exp), 4),
                    "log_likelihood": round(ll_exp, 2),
                    "aic": round(aic_exp, 2),
                    "llr_vs_pl": round(ll_pl - ll_exp, 2)
                },
                "log_normal": {
                    "mu": round(mu_ln, 4),
                    "sigma": round(s_ln, 4),
                    "ks": round(float(ks_ln), 4),
                    "log_likelihood": round(ll_ln, 2),
                    "aic": round(aic_ln, 2),
                    "llr_vs_pl": round(ll_pl - ll_ln, 2)
                }
            }

        eval_global = evaluate_tail_models(best_kmin)
        eval_1500 = evaluate_tail_models(1500)
        eval_500 = evaluate_tail_models(500)

        res_power_law = {
            "otimo_clauset": eval_global,
            "referencia_1500": eval_1500,
            "referencia_500": eval_500,
            "conclusao_diagnostica": (
                "A rede G2 (ego-Gplus) NÃO é uma lei de potência pura no sentido estrito. "
                "A distribuição de graus apresenta curvatura acentuada em escala log-log com queda pronunciada para k > 10^3. "
                "Na varredura de Clauset, o mínimo KS ocorre em k_min = 4.496 onde gamma = 4.56, valor atipicamente elevado (fora do intervalo 2 < gamma < 3). "
                "Para regimes de cauda intermediários (k_min = 500 e 1500), a distribuição Log-Normal supera a Lei de Potência por ampla margem "
                "(menor KS, maior verossimilhança e menor AIC), confirmando que a rede social é melhor descrita por um corpo log-normal com corte acentuado."
            )
        }
        with open(pl_file, "w", encoding="utf-8") as f:
            json.dump(res_power_law, f, indent=2, ensure_ascii=False)

    rob_file = os.path.join(RESULTS_DIR, "parte3_robustez.json")
    rep_csv_file = os.path.join(RESULTS_DIR, "parte3_robustez_repeticoes.csv")

    if os.path.exists(rob_file) and os.path.exists(rep_csv_file) and not force:
        print(f"\n[3/3] Carregando resultados existentes de Robustez de {rob_file}...")
        with open(rob_file, "r", encoding="utf-8") as f:
            res_robustez = json.load(f)
        base_metrics = res_robustez["base"]
        stats_rand = res_robustez["aleatorio_30_reps"]
        deg_metrics = res_robustez["ataque_grau"]
        bet_metrics = res_robustez["ataque_betweenness"]
        df_rand = pd.read_csv(rep_csv_file)
    else:
        print("\n--- ITEM 3: ROBUSTEZ ECOLÓGICA E ATAQUES DIRECIONADOS ---")
        removed_none = np.zeros(n, dtype=np.bool_)
        n_comp_base, gcc_sz_base, gcc_mask_base, n_iso_base = compute_components_masked(indptr, indices, removed_none)
        np.random.seed(42)
        srcs_base = np.random.choice(n, size=100, replace=False).astype(np.int32)
        apl_base = float(sample_apl_masked(indptr, indices, removed_none, srcs_base))
        diam_base = int(double_sweep_masked(indptr, indices, removed_none, 0, sweeps=5))

        base_metrics = {
            "fracao_gcc": float(gcc_sz_base / n),
            "num_componentes": int(n_comp_base),
            "apl_gcc": float(apl_base),
            "diametro_gcc": int(diam_base),
            "fracao_isolados": float(n_iso_base / n),
            "gcc_nos": int(gcc_sz_base),
            "nos_isolados": int(n_iso_base)
        }

        print("  [Aleatório] Executando T = 30 repetições de remoção aleatória de 5%...")
        np.random.seed(42)
        random_reps = []
        for rep in range(30):
            rand_nodes = np.random.choice(n, size=r, replace=False)
            rem_mask = np.zeros(n, dtype=np.bool_)
            rem_mask[rand_nodes] = True

            n_comp_r, gcc_sz_r, gcc_mask_r, n_iso_r = compute_components_masked(indptr, indices, rem_mask)
            gcc_nodes_idx = np.where(gcc_mask_r)[0].astype(np.int32)
            srcs_r = np.random.choice(gcc_nodes_idx, size=100, replace=False)
            apl_r = float(sample_apl_masked(indptr, indices, rem_mask, srcs_r))
            diam_r = int(double_sweep_masked(indptr, indices, rem_mask, gcc_nodes_idx[0], sweeps=5))

            rep_data = {
                "repeticao": rep + 1,
                "fracao_gcc": float(gcc_sz_r / n),
                "fracao_gcc_restante": float(gcc_sz_r / (n - r)),
                "num_componentes": int(n_comp_r),
                "apl_gcc": float(apl_r),
                "diametro_gcc": int(diam_r),
                "fracao_isolados": float(n_iso_r / (n - r)),
                "fracao_isolados_total": float(n_iso_r / n),
                "gcc_nos": int(gcc_sz_r),
                "nos_isolados": int(n_iso_r)
            }
            random_reps.append(rep_data)

        df_rand = pd.DataFrame(random_reps)
        df_rand.to_csv(rep_csv_file, index=False)

        stats_rand = {
            "fracao_gcc": {
                "mean": float(df_rand["fracao_gcc"].mean()),
                "std": float(df_rand["fracao_gcc"].std()),
                "min": float(df_rand["fracao_gcc"].min()),
                "max": float(df_rand["fracao_gcc"].max())
            },
            "fracao_gcc_restante": {
                "mean": float(df_rand["fracao_gcc_restante"].mean()),
                "std": float(df_rand["fracao_gcc_restante"].std())
            },
            "num_componentes": {
                "mean": float(df_rand["num_componentes"].mean()),
                "std": float(df_rand["num_componentes"].std()),
                "min": int(df_rand["num_componentes"].min()),
                "max": int(df_rand["num_componentes"].max())
            },
            "apl_gcc": {
                "mean": float(df_rand["apl_gcc"].mean()),
                "std": float(df_rand["apl_gcc"].std())
            },
            "diametro_gcc": {
                "mean": float(df_rand["diametro_gcc"].mean()),
                "std": float(df_rand["diametro_gcc"].std()),
                "min": int(df_rand["diametro_gcc"].min()),
                "max": int(df_rand["diametro_gcc"].max())
            },
            "fracao_isolados": {
                "mean": float(df_rand["fracao_isolados"].mean()),
                "std": float(df_rand["fracao_isolados"].std())
            }
        }

        node_ids = np.arange(n, dtype=np.int32)
        order_deg = np.lexsort((node_ids, -degrees))
        target_nodes_deg = order_deg[:r]
        rem_mask_deg = np.zeros(n, dtype=np.bool_)
        rem_mask_deg[target_nodes_deg] = True

        n_comp_deg, gcc_sz_deg, gcc_mask_deg, n_iso_deg = compute_components_masked(indptr, indices, rem_mask_deg)
        gcc_nodes_deg = np.where(gcc_mask_deg)[0].astype(np.int32)
        np.random.seed(42)
        srcs_deg = np.random.choice(gcc_nodes_deg, size=100, replace=False)
        apl_deg = float(sample_apl_masked(indptr, indices, rem_mask_deg, srcs_deg))
        diam_deg = int(double_sweep_masked(indptr, indices, rem_mask_deg, gcc_nodes_deg[0], sweeps=5))

        deg_metrics = {
            "fracao_gcc": float(gcc_sz_deg / n),
            "fracao_gcc_restante": float(gcc_sz_deg / (n - r)),
            "num_componentes": int(n_comp_deg),
            "apl_gcc": float(apl_deg),
            "diametro_gcc": int(diam_deg),
            "fracao_isolados": float(n_iso_deg / (n - r)),
            "fracao_isolados_total": float(n_iso_deg / n),
            "gcc_nos": int(gcc_sz_deg),
            "nos_isolados": int(n_iso_deg)
        }

        np.random.seed(42)
        pivots_200 = np.random.choice(n, size=200, replace=False).astype(np.int32)
        cb_approx = brandes_approx_betweenness(indptr, indices, pivots_200)
        order_bet = np.lexsort((node_ids, -cb_approx))
        target_nodes_bet = order_bet[:r]
        rem_mask_bet = np.zeros(n, dtype=np.bool_)
        rem_mask_bet[target_nodes_bet] = True

        n_comp_bet, gcc_sz_bet, gcc_mask_bet, n_iso_bet = compute_components_masked(indptr, indices, rem_mask_bet)
        gcc_nodes_bet = np.where(gcc_mask_bet)[0].astype(np.int32)
        np.random.seed(42)
        srcs_bet = np.random.choice(gcc_nodes_bet, size=100, replace=False)
        apl_bet = float(sample_apl_masked(indptr, indices, rem_mask_bet, srcs_bet))
        diam_bet = int(double_sweep_masked(indptr, indices, rem_mask_bet, gcc_nodes_bet[0], sweeps=5))

        bet_metrics = {
            "fracao_gcc": float(gcc_sz_bet / n),
            "fracao_gcc_restante": float(gcc_sz_bet / (n - r)),
            "num_componentes": int(n_comp_bet),
            "apl_gcc": float(apl_bet),
            "diametro_gcc": int(diam_bet),
            "fracao_isolados": float(n_iso_bet / (n - r)),
            "fracao_isolados_total": float(n_iso_bet / n),
            "gcc_nos": int(gcc_sz_bet),
            "nos_isolados": int(n_iso_bet)
        }

        res_robustez = {
            "n_original": n,
            "r_removidos": r,
            "fracao_removida_pct": 5.0,
            "base": base_metrics,
            "aleatorio_30_reps": stats_rand,
            "ataque_grau": deg_metrics,
            "ataque_betweenness": bet_metrics
        }
        with open(rob_file, "w", encoding="utf-8") as f:
            json.dump(res_robustez, f, indent=2, ensure_ascii=False)

    tabela_comp = pd.DataFrame([
        {
            "Cenário Experimental": "Grafo Base (Intacto)",
            "Fração GCC (A)": f"{base_metrics['fracao_gcc']*100:.2f}%",
            "Nº Componentes (B)": f"{base_metrics['num_componentes']:,}",
            "APL GCC (C)": f"{base_metrics['apl_gcc']:.4f}",
            "Diâmetro GCC (C)": f"{base_metrics['diametro_gcc']}",
            "Fração Isolados (D)": f"{base_metrics['fracao_isolados']*100:.2f}%",
            "Nós GCC": f"{base_metrics['gcc_nos']:,}",
            "Nós Isolados": f"{base_metrics['nos_isolados']:,}"
        },
        {
            "Cenário Experimental": "Falha Aleatória (Média ± DP, T=30)",
            "Fração GCC (A)": f"{stats_rand['fracao_gcc']['mean']*100:.2f}% ± {stats_rand['fracao_gcc']['std']*100:.2f}%",
            "Nº Componentes (B)": f"{stats_rand['num_componentes']['mean']:.1f} ± {stats_rand['num_componentes']['std']:.1f}",
            "APL GCC (C)": f"{stats_rand['apl_gcc']['mean']:.4f} ± {stats_rand['apl_gcc']['std']:.4f}",
            "Diâmetro GCC (C)": f"{stats_rand['diametro_gcc']['mean']:.1f} ± {stats_rand['diametro_gcc']['std']:.1f}",
            "Fração Isolados (D)": f"{stats_rand['fracao_isolados']['mean']*100:.2f}% ± {stats_rand['fracao_isolados']['std']*100:.2f}%",
            "Nós GCC": f"{int(stats_rand['fracao_gcc']['mean']*n):,}",
            "Nós Isolados": f"{int(stats_rand['fracao_isolados']['mean']*(n-r)):,}"
        },
        {
            "Cenário Experimental": "Ataque Direcionado: Grau",
            "Fração GCC (A)": f"{deg_metrics['fracao_gcc']*100:.2f}%",
            "Nº Componentes (B)": f"{deg_metrics['num_componentes']:,}",
            "APL GCC (C)": f"{deg_metrics['apl_gcc']:.4f}",
            "Diâmetro GCC (C)": f"{deg_metrics['diametro_gcc']}",
            "Fração Isolados (D)": f"{deg_metrics['fracao_isolados']*100:.2f}%",
            "Nós GCC": f"{deg_metrics['gcc_nos']:,}",
            "Nós Isolados": f"{deg_metrics['nos_isolados']:,}"
        },
        {
            "Cenário Experimental": "Ataque Direcionado: Betweenness",
            "Fração GCC (A)": f"{bet_metrics['fracao_gcc']*100:.2f}%",
            "Nº Componentes (B)": f"{bet_metrics['num_componentes']:,}",
            "APL GCC (C)": f"{bet_metrics['apl_gcc']:.4f}",
            "Diâmetro GCC (C)": f"{bet_metrics['diametro_gcc']}",
            "Fração Isolados (D)": f"{bet_metrics['fracao_isolados']*100:.2f}%",
            "Nós GCC": f"{bet_metrics['gcc_nos']:,}",
            "Nós Isolados": f"{bet_metrics['nos_isolados']:,}"
        }
    ])
    tabela_comp.to_csv(os.path.join(RESULTS_DIR, "parte3_tabela_comparativa.csv"), index=False)

    print("\n--- GERANDO FIGURAS CIENTÍFICAS DA PARTE III ---")

    print("  Gerando figures/parte3_distribuicao_graus_ajuste.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    ax1 = axes[0]
    deg_min, deg_max = int(degrees.min()), int(degrees.max())
    bins = np.logspace(np.log10(max(1, deg_min)), np.log10(deg_max), 40)
    counts, edges = np.histogram(degrees, bins=bins, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    valid = counts > 0
    ax1.scatter(bin_centers[valid], counts[valid], color="#2b6cb0", alpha=0.8, s=35, edgecolors="none", label="Empírico $P(k)$", zorder=3)

    k_grid = np.logspace(np.log10(1500), np.log10(deg_max), 200)
    frac_tail_1500 = len(degrees[degrees >= 1500]) / n
    g_1500 = eval_1500["power_law"]["gamma"]
    pdf_pl = frac_tail_1500 * (g_1500 - 1.0) / 1500.0 * (k_grid / 1500.0) ** (-g_1500)
    ax1.plot(k_grid, pdf_pl, color="#e53e3e", linestyle="--", linewidth=2.0, label=f"Power Law ($\\gamma = {g_1500:.2f}$)", zorder=4)

    mu_1500 = eval_1500["log_normal"]["mu"]
    s_1500 = eval_1500["log_normal"]["sigma"]
    surv_1500 = stats.norm.sf((np.log(1500) - mu_1500) / s_1500)
    pdf_ln = frac_tail_1500 * (stats.lognorm.pdf(k_grid, s=s_1500, scale=np.exp(mu_1500)) / surv_1500)
    ax1.plot(k_grid, pdf_ln, color="#2f855a", linestyle="-", linewidth=2.2, label=f"Log-Normal ($\\mu={mu_1500:.1f}, \\sigma={s_1500:.2f}$)", zorder=5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Grau do Vértice ($k$)", fontsize=11)
    ax1.set_ylabel("Densidade de Probabilidade $P(k)$", fontsize=11)
    ax1.set_title("PDF de Graus e Ajustes de Cauda (Log-Log)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True, fontsize=9.5)

    ax2 = axes[1]
    sorted_deg = np.sort(degrees)
    ccdf_emp = 1.0 - (np.arange(n) / n)
    ax2.plot(sorted_deg, ccdf_emp, color="#2b6cb0", linewidth=2.2, label=r"CCDF Empírica $P(K \geq k)$", zorder=3)

    ccdf_pl = frac_tail_1500 * (k_grid / 1500.0) ** (-(g_1500 - 1.0))
    ax2.plot(k_grid, ccdf_pl, color="#e53e3e", linestyle="--", linewidth=2.0, label=f"Power Law ($k_{{\\min}}=1500$)", zorder=4)

    ccdf_ln = frac_tail_1500 * (stats.norm.sf((np.log(k_grid) - mu_1500) / s_1500) / surv_1500)
    ax2.plot(k_grid, ccdf_ln, color="#2f855a", linestyle="-", linewidth=2.2, label=f"Log-Normal (Ajuste Superior)", zorder=5)

    ax2.axvline(1500, color="#718096", linestyle=":", linewidth=1.5, label="$k_{\\min} = 1.500$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Grau do Vértice ($k$)", fontsize=11)
    ax2.set_ylabel(r"Probabilidade Cumulativa $P(K \geq k)$", fontsize=11)
    ax2.set_title("CCDF Empírica vs. Modelos Teóricos", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, fontsize=9.5)

    fig.suptitle(
        f"Análise de Lei de Potência e Diagnóstico de Cauda (G2 — ego-Gplus)\n"
        f"AIC: Log-Normal ({eval_1500['log_normal']['aic']:.0f}) << Power Law ({eval_1500['power_law']['aic']:.0f})",
        fontsize=13, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "parte3_distribuicao_graus_ajuste.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()

    print("  Gerando figures/parte3_power_law_ks_scan.png...")
    plt.figure(figsize=(8, 4.5), dpi=200)
    df_scan = pd.DataFrame(scan_results)
    plt.plot(df_scan["k_min"], df_scan["ks"], color="#3182ce", linewidth=1.8, label="Distância KS ($D$)")
    plt.scatter([best_kmin], [best_ks], color="#e53e3e", s=90, zorder=5, label=f"Mínimo Global: $k_{{\\min}}={best_kmin}$ ($D={best_ks:.4f}, \\gamma={best_gamma:.2f}$)")
    plt.axvline(1500, color="#2f855a", linestyle="--", linewidth=1.5, label=f"Regime Típico: $k=1500$ ($D={eval_1500['power_law']['ks']:.4f}, \\gamma={g_1500:.2f}$)")
    plt.xscale("log")
    plt.xlabel("Limiar da Cauda ($k_{\\min}$)", fontsize=11)
    plt.ylabel("Distância de Kolmogorov-Smirnov ($D$)", fontsize=11)
    plt.title("Método de Clauset: Otimização de $k_{\\min}$ via Minimização de KS", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "parte3_power_law_ks_scan.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    print("  Gerando figures/parte3_robustez_boxplots.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

    axA = axes[0, 0]
    box_A = axA.boxplot([df_rand["fracao_gcc"] * 100], positions=[1], widths=0.45, patch_artist=True,
                        boxprops=dict(facecolor="#bee3f8", color="#2b6cb0", linewidth=1.5),
                        medianprops=dict(color="#1a365d", linewidth=2.0))
    axA.axhline(base_metrics["fracao_gcc"] * 100, color="#718096", linestyle="--", linewidth=1.8, label=f"Grafo Base: {base_metrics['fracao_gcc']*100:.1f}%")
    axA.scatter([1], [deg_metrics["fracao_gcc"] * 100], color="#e53e3e", s=110, zorder=6, marker="D", label=f"Ataque Grau: {deg_metrics['fracao_gcc']*100:.2f}%")
    axA.scatter([1], [bet_metrics["fracao_gcc"] * 100], color="#805ad5", s=110, zorder=6, marker="s", label=f"Ataque Betweenness: {bet_metrics['fracao_gcc']*100:.2f}%")
    mean_A = stats_rand["fracao_gcc"]["mean"] * 100
    std_A = stats_rand["fracao_gcc"]["std"] * 100
    axA.text(1.28, mean_A, f"Média Aleat:\n{mean_A:.2f}% ± {std_A:.2f}%", fontsize=10, fontweight="bold", color="#2b6cb0", va="center")
    axA.set_xticks([1])
    axA.set_xticklabels(["Falha Aleatória (T=30)"], fontsize=11)
    axA.set_ylabel("Fração do Maior Componente (%)", fontsize=11)
    axA.set_title("(A) Tamanho da Componente Gigante ($S$)", fontsize=12, fontweight="bold")
    axA.grid(True, linestyle=":", alpha=0.5)
    axA.legend(loc="lower left", frameon=True, fontsize=10)

    axB = axes[0, 1]
    box_B = axB.boxplot([df_rand["num_componentes"]], positions=[1], widths=0.45, patch_artist=True,
                        boxprops=dict(facecolor="#feebc8", color="#c05621", linewidth=1.5),
                        medianprops=dict(color="#7b341e", linewidth=2.0))
    axB.axhline(base_metrics["num_componentes"], color="#718096", linestyle="--", linewidth=1.8, label=f"Grafo Base: {base_metrics['num_componentes']}")
    axB.scatter([1], [deg_metrics["num_componentes"]], color="#e53e3e", s=110, zorder=6, marker="D", label=f"Ataque Grau: {deg_metrics['num_componentes']:,}")
    axB.scatter([1], [bet_metrics["num_componentes"]], color="#805ad5", s=110, zorder=6, marker="s", label=f"Ataque Betweenness: {bet_metrics['num_componentes']:,}")
    mean_B = stats_rand["num_componentes"]["mean"]
    std_B = stats_rand["num_componentes"]["std"]
    axB.text(1.28, mean_B, f"Média Aleat:\n{mean_B:.1f} ± {std_B:.1f}", fontsize=10, fontweight="bold", color="#c05621", va="center")
    axB.set_xticks([1])
    axB.set_xticklabels(["Falha Aleatória (T=30)"], fontsize=11)
    axB.set_ylabel("Número de Componentes Conexas", fontsize=11)
    axB.set_title("(B) Fragmentação da Rede (Total de Componentes)", fontsize=12, fontweight="bold")
    axB.grid(True, linestyle=":", alpha=0.5)
    axB.legend(loc="upper left", frameon=True, fontsize=10)

    axC = axes[1, 0]
    box_C = axC.boxplot([df_rand["apl_gcc"]], positions=[1], widths=0.45, patch_artist=True,
                        boxprops=dict(facecolor="#c6f6d5", color="#276749", linewidth=1.5),
                        medianprops=dict(color="#1c4532", linewidth=2.0))
    axC.axhline(base_metrics["apl_gcc"], color="#718096", linestyle="--", linewidth=1.8, label=f"Grafo Base: {base_metrics['apl_gcc']:.4f}")
    axC.scatter([1], [deg_metrics["apl_gcc"]], color="#e53e3e", s=110, zorder=6, marker="D", label=f"Ataque Grau: {deg_metrics['apl_gcc']:.4f}")
    axC.scatter([1], [bet_metrics["apl_gcc"]], color="#805ad5", s=110, zorder=6, marker="s", label=f"Ataque Betweenness: {bet_metrics['apl_gcc']:.4f}")
    mean_C = stats_rand["apl_gcc"]["mean"]
    std_C = stats_rand["apl_gcc"]["std"]
    axC.text(1.28, mean_C, f"Média Aleat:\n{mean_C:.4f} ± {std_C:.4f}", fontsize=10, fontweight="bold", color="#276749", va="center")
    axC.set_xticks([1])
    axC.set_xticklabels(["Falha Aleatória (T=30)"], fontsize=11)
    axC.set_ylabel("APL Amostrado na GCC", fontsize=11)
    axC.set_title("(C) Eficiência de Comunicação (APL da GCC)", fontsize=12, fontweight="bold")
    axC.grid(True, linestyle=":", alpha=0.5)
    axC.legend(loc="upper left", frameon=True, fontsize=10)

    axD = axes[1, 1]
    box_D = axD.boxplot([df_rand["fracao_isolados"] * 100], positions=[1], widths=0.45, patch_artist=True,
                        boxprops=dict(facecolor="#fed7d7", color="#9b2c2c", linewidth=1.5),
                        medianprops=dict(color="#63171b", linewidth=2.0))
    axD.axhline(base_metrics["fracao_isolados"] * 100, color="#718096", linestyle="--", linewidth=1.8, label=f"Grafo Base: {base_metrics['fracao_isolados']*100:.2f}%")
    axD.scatter([1], [deg_metrics["fracao_isolados"] * 100], color="#e53e3e", s=110, zorder=6, marker="D", label=f"Ataque Grau: {deg_metrics['fracao_isolados']*100:.2f}% ({deg_metrics['nos_isolados']:,})")
    axD.scatter([1], [bet_metrics["fracao_isolados"] * 100], color="#805ad5", s=110, zorder=6, marker="s", label=f"Ataque Betweenness: {bet_metrics['fracao_isolados']*100:.2f}% ({bet_metrics['nos_isolados']:,})")
    mean_D = stats_rand["fracao_isolados"]["mean"] * 100
    std_D = stats_rand["fracao_isolados"]["std"] * 100
    axD.text(1.28, mean_D, f"Média Aleat:\n{mean_D:.2f}% ± {std_D:.2f}%", fontsize=10, fontweight="bold", color="#9b2c2c", va="center")
    axD.set_xticks([1])
    axD.set_xticklabels(["Falha Aleatória (T=30)"], fontsize=11)
    axD.set_ylabel("Fração de Nós Isolados (%)", fontsize=11)
    axD.set_title("(D) Desconexão Terminal (Fração de Isolados)", fontsize=12, fontweight="bold")
    axD.grid(True, linestyle=":", alpha=0.5)
    axD.legend(loc="upper left", frameon=True, fontsize=10)

    fig.suptitle(
        f"Análise de Robustez Topológica: Falha Aleatória vs. Ataques Direcionados (G2 — ego-Gplus)\n"
        f"Remoção de 5% dos Vértices ($r = {r:,}$ nós)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "parte3_robustez_boxplots.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()

    print("  Gerando figures/parte3_small_world_comparacao.png...")
    fig, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)

    labels = ["Grafo Real $G$", "ER Empírico", "ER Teórico"]
    c_vals = [c_g, c_rand_emp, c_rand_theo]
    colors_c = ["#2b6cb0", "#48bb78", "#ed8936"]
    bars_c = ax_c.bar(labels, c_vals, color=colors_c, width=0.55, edgecolor="#1a202c", linewidth=1.2)
    ax_c.set_ylabel("Coeficiente de Clusterização Médio ($C$)", fontsize=11)
    ax_c.set_title("Clusterização: $C(G) \\gg C_{\\text{rand}}$ (~232x)", fontsize=12, fontweight="bold")
    ax_c.set_ylim(0, 0.58)
    for bar in bars_c:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.6f}" if h < 0.01 else f"{h:.4f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_c.grid(axis="y", linestyle=":", alpha=0.5)

    l_vals = [l_g, l_rand_emp, l_rand_theo]
    colors_l = ["#2b6cb0", "#48bb78", "#ed8936"]
    bars_l = ax_l.bar(labels, l_vals, color=colors_l, width=0.55, edgecolor="#1a202c", linewidth=1.2)
    ax_l.set_ylabel("Average Path Length ($L$)", fontsize=11)
    ax_l.set_title("Distância Média: $L(G) \\approx L_{\\text{rand}}$ (~1.09x)", fontsize=12, fontweight="bold")
    ax_l.set_ylim(0, 3.5)
    for bar in bars_l:
        h = bar.get_height()
        ax_l.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.4f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_l.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Propriedade de Mundo Pequeno (Watts–Strogatz)\n"
        f"$\\sigma_{{\\text{{emp}}}} = \\frac{{C/C_{{\\text{{rand}}}}}}{{L/L_{{\\text{{rand}}}}}} = {sigma_emp:.2f} \\gg 1$",
        fontsize=13, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "parte3_small_world_comparacao.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()

    print("\n--- GERANDO results/parte3_resumo.md ---")
    resumo_md = f"""# Resumo Executivo da Parte III: Análise Estrutural Avançada

**Aluno:** Vitor Renato Santos dos Santos | **Matrícula:** 222217109 — UFBA  
**Dataset:** ego-Gplus (Google+ social circles) — SNAP | **Grafo:** G2 ($n = {n:,}$, $m = {m:,}$)

---

## 1. Small-World (Apêndice A — Watts & Strogatz)

### Tabela de Métricas Comparativas
| Métrica | Grafo Real $G$ | Grafo Aleatório Teórico $G(n, m)$ | Grafo Aleatório Empírico $G(n, m)$ |
| :--- | :---: | :---: | :---: |
| **Comprimento Médio ($L$)** | **{l_g:.4f}** | {l_rand_theo:.4f} | **{l_rand_emp:.4f}** |
| **Clusterização Média ($C$)** | **{c_g:.6f}** | {c_rand_theo:.6f} | **{c_rand_emp:.6f}** |
| **Razão $L / L_{{\\text{{rand}}}}$** | — | {ratio_l_theo:.4f} | **{ratio_l_emp:.4f}** |
| **Razão $C / C_{{\\text{{rand}}}}$** | — | {ratio_c_theo:.2f} | **{ratio_c_emp:.2f}** |
| **Índice Small-World ($\\sigma$)** | — | **{sigma_theo:.2f}** | **{sigma_emp:.2f}** |

### Rascunho da Resposta (Pergunta 1 da Parte III)
A rede ego-Gplus satisfaz rigorosamente a definição de rede de mundo pequeno (*small-world*) proposta por Watts & Strogatz. O comprimento médio dos caminhos ($L = {l_g:.4f}$) é da mesma ordem de grandeza da rede aleatória equivalente ($L / L_{{\\text{{rand}}}} = {ratio_l_emp:.4f} \\approx 1$), enquanto o coeficiente de clusterização local ($C = {c_g:.6f}$) é extraordinariamente superior ao esperado puramente ao acaso ($C / C_{{\\text{{rand}}}} = {ratio_c_emp:.2f} \\gg 1$), resultando em um coeficiente de mundo pequeno $\\sigma = {sigma_emp:.2f} \\gg 1$. Na prática social, essa topologia reflete a convivência entre círculos sociais altamente coesos e fechados (onde amigos em comum tendem fortemente a ser amigos entre si) e a existência de conexões de atalho (*weak ties*) que encurtam radicalmente a distância geodésica entre quaisquer dois indivíduos no Google+.

---

## 2. Lei de Potência (Apêndice B — Clauset, Shalizi & Newman)

### Tabela de Comparação de Modelos na Cauda
| Cenário / Limiar | Modelo Ajustado | Parâmetros Estimados | Distância KS ($D$) | Log-Likelihood | AIC | $\\Delta$ AIC vs PL |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Mínimo Global Clauset**<br>($k_{{\\min}} = {best_kmin}$, $N_{{\\text{{cauda}}}} = {best_ntail:,}$, {best_ntail/n*100:.2f}%) | **Power Law** | $\\gamma = {best_gamma:.4f}$ | {best_ks:.4f} | {eval_global['power_law']['log_likelihood']:.2f} | {eval_global['power_law']['aic']:.2f} | — |
| | **Log-Normal** | $\\mu = {eval_global['log_normal']['mu']:.2f}, \\sigma = {eval_global['log_normal']['sigma']:.2f}$ | {eval_global['log_normal']['ks']:.4f} | {eval_global['log_normal']['log_likelihood']:.2f} | {eval_global['log_normal']['aic']:.2f} | **{eval_global['power_law']['aic'] - eval_global['log_normal']['aic']:.2f}** |
| | **Exponencial** | $\\lambda^{{-1}} = {eval_global['exponencial']['scale']:.2f}$ | {eval_global['exponencial']['ks']:.4f} | {eval_global['exponencial']['log_likelihood']:.2f} | {eval_global['exponencial']['aic']:.2f} | {eval_global['power_law']['aic'] - eval_global['exponencial']['aic']:.2f} |
| **Regime Representativo**<br>($k_{{\\min}} = 1.500$, $N_{{\\text{{cauda}}}} = {eval_1500['n_tail']:,}$, {eval_1500['fraction_tail_pct']:.2f}%) | **Power Law** | $\\gamma = {eval_1500['power_law']['gamma']:.4f}$ | {eval_1500['power_law']['ks']:.4f} | {eval_1500['power_law']['log_likelihood']:.2f} | {eval_1500['power_law']['aic']:.2f} | — |
| | **Log-Normal** | $\\mu = {eval_1500['log_normal']['mu']:.2f}, \\sigma = {eval_1500['log_normal']['sigma']:.2f}$ | **{eval_1500['log_normal']['ks']:.4f}** | **{eval_1500['log_normal']['log_likelihood']:.2f}** | **{eval_1500['log_normal']['aic']:.2f}** | **{eval_1500['power_law']['aic'] - eval_1500['log_normal']['aic']:.2f}** |
| | **Exponencial** | $\\lambda^{{-1}} = {eval_1500['exponencial']['scale']:.2f}$ | {eval_1500['exponencial']['ks']:.4f} | {eval_1500['exponencial']['log_likelihood']:.2f} | {eval_1500['exponencial']['aic']:.2f} | {eval_1500['power_law']['aic'] - eval_1500['exponencial']['aic']:.2f} |

### Rascunho da Resposta (Pergunta 2 da Parte III)
A distribuição de conectividade do ego-Gplus não segue uma lei de potência pura em sua cauda, apresentando nítida curvatura convexa na CCDF com declínio acelerado acima de $k \\approx 10^3$. A varredura sistemática de Clauset minimiza o teste KS somente em $k_{{\\min}} = {best_kmin}$, resultando em um expoente $\\gamma = {best_gamma:.2f}$ anipicamente íngreme (fora da faixa canônica de redes livres de escala $2 < \\gamma < 3$) e restringindo a cauda a apenas {best_ntail/n*100:.2f}% dos vértices. Ao confrontar os modelos teóricos concorrentes para $k \\geq 1.500$, a distribuição Log-Normal supera a Lei de Potência de forma decisiva (KS de {eval_1500['log_normal']['ks']:.4f} contra {eval_1500['power_law']['ks']:.4f}; $\\Delta\\text{{AIC}} = {eval_1500['power_law']['aic'] - eval_1500['log_normal']['aic']:.1f}$ a favor da Log-Normal), confirmando que a rede é governada por múltiplos processos multiplicativos e saturação cognitiva de conexões em vez de anexo preferencial irrestrito.

---

## 3. Robustez Topológica e Ataques (Apêndice C)

### Tabela de Comparação Estrutural (Remoção de $r = {r:,}$ vértices — 5,0%)
| Cenário Experimental | Fração GCC ($S$) | Nº Componentes ($B$) | APL na GCC ($C$) | Diâmetro GCC ($C$) | Fração Isolados ($D$) | Nós na GCC | Nós Isolados |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grafo Base (Intacto)** | **100,00%** | **1** | **{base_metrics['apl_gcc']:.4f}** | **{base_metrics['diametro_gcc']}** | **0,00%** | **{base_metrics['gcc_nos']:,}** | **0** |
| **Falha Aleatória ($T=30$)** | **{stats_rand['fracao_gcc']['mean']*100:.2f}% ± {stats_rand['fracao_gcc']['std']*100:.2f}%** | **{stats_rand['num_componentes']['mean']:.1f} ± {stats_rand['num_componentes']['std']:.1f}** | **{stats_rand['apl_gcc']['mean']:.4f} ± {stats_rand['apl_gcc']['std']:.4f}** | **{stats_rand['diametro_gcc']['mean']:.1f} ± {stats_rand['diametro_gcc']['std']:.1f}** | **{stats_rand['fracao_isolados']['mean']*100:.2f}% ± {stats_rand['fracao_isolados']['std']*100:.2f}%** | {int(stats_rand['fracao_gcc']['mean']*n):,} | {int(stats_rand['fracao_isolados']['mean']*(n-r)):,} |
| **Ataque: Grau (Top 5%)** | **{deg_metrics['fracao_gcc']*100:.2f}%** | **{deg_metrics['num_componentes']:,}** | **{deg_metrics['apl_gcc']:.4f}** | **{deg_metrics['diametro_gcc']}** | **{deg_metrics['fracao_isolados']*100:.2f}%** | **{deg_metrics['gcc_nos']:,}** | **{deg_metrics['nos_isolados']:,}** |
| **Ataque: Betweenness (Top 5%)** | **{bet_metrics['fracao_gcc']*100:.2f}%** | **{bet_metrics['num_componentes']:,}** | **{bet_metrics['apl_gcc']:.4f}** | **{bet_metrics['diametro_gcc']}** | **{bet_metrics['fracao_isolados']*100:.2f}%** | **{bet_metrics['gcc_nos']:,}** | **{bet_metrics['nos_isolados']:,}** |

### Rascunho da Resposta (Pergunta 3 da Parte III)
A rede exibe o comportamento clássico de robustez assimétrica frente a perdas nodais: alta tolerância a falhas estocásticas, mas extrema fragilidade a ataques direcionados. Sob remoção aleatória de 5% dos nós, a componente gigante retém $S_{{\\text{{rand}}}} = {stats_rand['fracao_gcc']['mean']*100:.2f}\\%$ da rede ({stats_rand['fracao_gcc_restante']['mean']*100:.2f}% dos nós sobreviventes), o diâmetro varia discretamente ({stats_rand['diametro_gcc']['mean']:.1f}) e formam-se em média apenas {stats_rand['num_componentes']['mean']:.0f} fragmentos. Em contrapartida, a eliminação dos 5% nós mais conectados ($S_{{\\text{{deg}}}} = {deg_metrics['fracao_gcc']*100:.2f}\\%$) e de maior intermediação ($S_{{\\text{{bet}}}} = {bet_metrics['fracao_gcc']*100:.2f}\\%$) provoca fragmentação severa, originando {deg_metrics['num_componentes']:,} e {bet_metrics['num_componentes']:,} componentes desconexas e elevando o diâmetro para {deg_metrics['diametro_gcc']} e {bet_metrics['diametro_gcc']}. Esse colapso acentuado ($S_{{\\text{{rand}}}} \\gg S_{{\\text{{cent}}}}$) decorre da dependência topológica em relação a hubs centrais e pontes estruturais responsáveis por manter interligados os aglomerados de círculos egoicos.
"""
    with open(os.path.join(RESULTS_DIR, "parte3_resumo.md"), "w", encoding="utf-8") as f:
        f.write(resumo_md)

    t_global_total = time.perf_counter() - t_global_start
    print(f"\n=====================================================================")
    print(f"=== PARTE III CONCLUÍDA COM SUCESSO EM {t_global_total:.2f}s! ===")
    print(f"Arquivos gerados em:")
    print(f"  - {os.path.join(RESULTS_DIR, 'parte3_small_world.json')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'parte3_power_law.json')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'parte3_robustez.json')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'parte3_robustez_repeticoes.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'parte3_tabela_comparativa.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'parte3_resumo.md')}")
    print(f"  - {fig1_path}")
    print(f"  - {fig2_path}")
    print(f"  - {fig3_path}")
    print(f"  - {fig4_path}")
    print(f"=====================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Força recalcular tudo")
    args = parser.parse_args()
    run_all_parte3(force=args.force)
