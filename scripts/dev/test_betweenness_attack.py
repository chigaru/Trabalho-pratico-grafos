import time
import numpy as np
from numba import njit
from test_betweenness import brandes_approx_betweenness
from test_robustez import compute_components_masked, double_sweep_masked, sample_apl_masked

npz_path = "data/processed/gplus_undirected_gcc.npz"
data = np.load(npz_path)
indptr = data["indptr"]
indices = data["indices"]
n = len(indptr) - 1
r = int(np.ceil(0.05 * n))

print("Calculando betweenness aproximada com 200 pivôs...")
np.random.seed(42)
pivots_200 = np.random.choice(n, size=200, replace=False).astype(np.int32)
t0 = time.perf_counter()
cb = brandes_approx_betweenness(indptr, indices, pivots_200)
t_cb = time.perf_counter() - t0
print(f"Betweenness calculada em {t_cb:.2f}s!")

node_ids = np.arange(n, dtype=np.int32)
order_bet = np.lexsort((node_ids, -cb))
target_nodes_bet = order_bet[:r]

removed_bet = np.zeros(n, dtype=np.bool_)
removed_bet[target_nodes_bet] = True

t0 = time.perf_counter()
n_comp_b, gcc_sz_b, gcc_mask_b, n_iso_b = compute_components_masked(indptr, indices, removed_bet)
t_comp_b = time.perf_counter() - t0

print(f"\nAtaque direcionado por Betweenness (200 pivôs):")
print(f"  Componentes: {n_comp_b:,}")
print(f"  GCC: {gcc_sz_b:,} nós ({gcc_sz_b/n*100:.2f}%)")
print(f"  Isolados: {n_iso_b:,} ({n_iso_b/(n-r)*100:.2f}%)")

gcc_indices_b = np.where(gcc_mask_b)[0].astype(np.int32)
if len(gcc_indices_b) > 0:
    srcs_b = np.random.choice(gcc_indices_b, size=min(100, len(gcc_indices_b)), replace=False)
    apl_b = sample_apl_masked(indptr, indices, removed_bet, srcs_b)
    diam_b = double_sweep_masked(indptr, indices, removed_bet, gcc_indices_b[0], sweeps=5)
    print(f"  APL na GCC: {apl_b:.4f} | Diâmetro: {diam_b}")
