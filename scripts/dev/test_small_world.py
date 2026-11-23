import time
import numpy as np
from numba import njit, prange

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
            tmp = u
            u = v
            v = tmp

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
def compute_clustering_er(indptr: np.ndarray, indices: np.ndarray):
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
def sample_apl_er(indptr: np.ndarray, indices: np.ndarray, sources: np.ndarray):
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

if __name__ == "__main__":
    n = 107614
    m = 12238285
    print(f"Testando gerador G(n, m) com n={n:,}, m={m:,}...")
    t0 = time.perf_counter()
    u_edges, v_edges = sample_er_edges(n, m, seed=42)
    t_gen = time.perf_counter() - t0
    print(f"Arestas geradas em {t_gen:.2f}s. Construindo CSR...")

    t0 = time.perf_counter()
    indptr, indices = build_csr_from_edges(n, m, u_edges, v_edges)
    t_csr = time.perf_counter() - t0
    print(f"CSR construído em {t_csr:.2f}s.")

    print("Calculando clusterização média no ER...")
    t0 = time.perf_counter()
    c_er = compute_clustering_er(indptr, indices)
    t_c = time.perf_counter() - t0
    print(f"C_ER = {c_er:.6f} (calculado em {t_c:.2f}s)")

    print("Calculando APL amostrado (200 fontes) no ER...")
    np.random.seed(42)
    sources = np.random.choice(n, size=200, replace=False).astype(np.int32)
    t0 = time.perf_counter()
    apl_er = sample_apl_er(indptr, indices, sources)
    t_apl = time.perf_counter() - t0
    print(f"APL_ER = {apl_er:.4f} (calculado em {t_apl:.2f}s)")

    c_g = 0.490118
    apl_g = 2.8603
    print(f"L/L_rand_emp = {apl_g / apl_er:.4f}")
    print(f"C/C_rand_emp = {c_g / c_er:.4f}")
    sigma_emp = (c_g / c_er) / (apl_g / apl_er)
    print(f"sigma_emp = {sigma_emp:.4f}")
