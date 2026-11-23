import time
import numpy as np
from numba import njit, prange

@njit
def compute_components_masked(indptr: np.ndarray, indices: np.ndarray, removed: np.ndarray):
    n = len(indptr) - 1
    visited = np.zeros(n, dtype=np.bool_)
    gcc_mask = np.zeros(n, dtype=np.bool_)

    queue = np.empty(n, dtype=np.int32)

    comp_sizes = []
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
        comp_sizes.append(c_size)
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

if __name__ == "__main__":
    npz_path = "data/processed/gplus_undirected_gcc.npz"
    data = np.load(npz_path)
    indptr = data["indptr"]
    indices = data["indices"]
    n = len(indptr) - 1
    r = int(np.ceil(0.05 * n))

    print(f"Grafo carregado: n={n:,}, arestas={len(indices)//2:,}, r={r:,} (5%)")

    np.random.seed(42)
    rand_nodes = np.random.choice(n, size=r, replace=False)
    removed_rand = np.zeros(n, dtype=np.bool_)
    removed_rand[rand_nodes] = True

    t0 = time.perf_counter()
    n_comp, gcc_sz, gcc_mask, n_iso = compute_components_masked(indptr, indices, removed_rand)
    t_comp = time.perf_counter() - t0
    print(f"Remoção aleatória 1 rep: componentes={n_comp:,}, GCC={gcc_sz:,} ({gcc_sz/n*100:.2f}%), isolados={n_iso:,} (em {t_comp:.3f}s)")

    gcc_indices = np.where(gcc_mask)[0].astype(np.int32)
    sample_srcs = np.random.choice(gcc_indices, size=100, replace=False)

    t0 = time.perf_counter()
    apl_val = sample_apl_masked(indptr, indices, removed_rand, sample_srcs)
    t_apl = time.perf_counter() - t0

    t0 = time.perf_counter()
    diam_val = double_sweep_masked(indptr, indices, removed_rand, gcc_indices[0], sweeps=5)
    t_diam = time.perf_counter() - t0
    print(f"  APL na GCC: {apl_val:.4f} (em {t_apl:.3f}s) | Diâmetro (sweep): {diam_val} (em {t_diam:.3f}s)")

    degrees = indptr[1:] - indptr[:-1]
    node_ids = np.arange(n, dtype=np.int32)
    order_deg = np.lexsort((node_ids, -degrees))
    target_nodes = order_deg[:r]

    removed_deg = np.zeros(n, dtype=np.bool_)
    removed_deg[target_nodes] = True

    t0 = time.perf_counter()
    n_comp_d, gcc_sz_d, gcc_mask_d, n_iso_d = compute_components_masked(indptr, indices, removed_deg)
    t_comp_d = time.perf_counter() - t0
    print(f"\nAtaque direcionado por grau:")
    print(f"  Componentes: {n_comp_d:,}")
    print(f"  GCC: {gcc_sz_d:,} nós ({gcc_sz_d/n*100:.2f}%)")
    print(f"  Isolados: {n_iso_d:,} ({n_iso_d/(n-r)*100:.2f}%)")

    gcc_indices_d = np.where(gcc_mask_d)[0].astype(np.int32)
    if len(gcc_indices_d) > 0:
        srcs_d = np.random.choice(gcc_indices_d, size=min(100, len(gcc_indices_d)), replace=False)
        apl_d = sample_apl_masked(indptr, indices, removed_deg, srcs_d)
        diam_d = double_sweep_masked(indptr, indices, removed_deg, gcc_indices_d[0], sweeps=5)
        print(f"  APL na GCC: {apl_d:.4f} | Diâmetro: {diam_d}")
