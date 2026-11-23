import time
import numpy as np
from numba import njit, prange

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

if __name__ == "__main__":
    npz_path = "data/processed/gplus_undirected_gcc.npz"
    data = np.load(npz_path)
    indptr = data["indptr"]
    indices = data["indices"]
    n = len(indptr) - 1

    print("Compilando e testando 1 pivô...")
    t0 = time.perf_counter()
    _ = brandes_approx_betweenness(indptr, indices, np.array([0], dtype=np.int32))
    t_warm = time.perf_counter() - t0
    print(f"Warmup em {t_warm:.2f}s")

    np.random.seed(42)
    pivots_20 = np.random.choice(n, size=20, replace=False).astype(np.int32)
    t0 = time.perf_counter()
    _ = brandes_approx_betweenness(indptr, indices, pivots_20)
    t_20 = time.perf_counter() - t0
    print(f"20 pivôs em {t_20:.2f}s (estimativa para 200 pivôs: {t_20*10:.1f}s)")
