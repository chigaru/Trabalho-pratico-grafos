import time
import pickle
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import igraph as ig

@njit
def generate_symmetric_weights_numba(indptr, indices, seed=42):
    np.random.seed(seed)
    N = len(indptr) - 1
    M_directed = len(indices)
    weights = np.zeros(M_directed, dtype=np.int32)

    for u in range(N):
        u_start = indptr[u]
        u_end = indptr[u + 1]
        for i in range(u_start, u_end):
            v = indices[i]
            if u < v:
                w = np.random.randint(1, 11)
                weights[i] = w
                v_start = indptr[v]
                v_end = indptr[v + 1]
                low = v_start
                high = v_end - 1
                found = -1
                while low <= high:
                    mid = (low + high) // 2
                    mid_val = indices[mid]
                    if mid_val == u:
                        found = mid
                        break
                    elif mid_val < u:
                        low = mid + 1
                    else:
                        high = mid - 1
                if found != -1:
                    weights[found] = w
    return weights

@njit
def bfs_csr(indptr, indices, source):
    N = len(indptr) - 1
    dist = np.full(N, -1, dtype=np.int32)
    queue = np.empty(N, dtype=np.int32)
    dist[source] = 0
    queue[0] = source
    head = 0
    tail = 1

    while head < tail:
        u = queue[head]
        head += 1
        d_next = dist[u] + 1
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if dist[v] == -1:
                dist[v] = d_next
                queue[tail] = v
                tail += 1
    return dist

@njit
def dfs_iterative_csr(indptr, indices, source):
    N = len(indptr) - 1
    visited = np.zeros(N, dtype=np.bool_)
    order = np.empty(N, dtype=np.int32)
    stack = np.empty(N, dtype=np.int32)

    visited[source] = True
    stack[0] = source
    top = 1
    count = 0

    while top > 0:
        top -= 1
        u = stack[top]
        order[count] = u
        count += 1
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if not visited[v]:
                visited[v] = True
                stack[top] = v
                top += 1
    return visited, order[:count]

@njit
def check_eulerian_numba(indptr, indices):
    N = len(indptr) - 1
    odd_count = 0
    even_count = 0
    for u in range(N):
        deg = indptr[u + 1] - indptr[u]
        if deg % 2 != 0:
            odd_count += 1
        else:
            even_count += 1

    visited = np.zeros(N, dtype=np.bool_)
    start_node = 0
    visited[start_node] = True
    queue = np.empty(N, dtype=np.int32)
    queue[0] = start_node
    head = 0
    tail = 1
    while head < tail:
        u = queue[head]
        head += 1
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if not visited[v]:
                visited[v] = True
                queue[tail] = v
                tail += 1

    is_connected = (tail == N)
    is_eulerian = is_connected and (odd_count == 0)
    has_eulerian_path = is_connected and (odd_count == 2)
    return is_eulerian, has_eulerian_path, is_connected, odd_count, even_count

@njit
def dijkstra_csr(indptr, indices, weights, source):
    N = len(indptr) - 1
    INF = np.int64(10**15)
    dist = np.full(N, INF, dtype=np.int64)
    dist[source] = 0

    heap = np.empty(N, dtype=np.int32)
    pos = np.full(N, -1, dtype=np.int32)

    heap[0] = source
    pos[source] = 0
    heap_size = 1

    while heap_size > 0:
        u = heap[0]
        heap_size -= 1
        if heap_size > 0:
            last = heap[heap_size]
            heap[0] = last
            pos[last] = 0
            idx = 0
            while True:
                left = 2 * idx + 1
                right = 2 * idx + 2
                smallest = idx
                if left < heap_size and dist[heap[left]] < dist[heap[smallest]]:
                    smallest = left
                if right < heap_size and dist[heap[right]] < dist[heap[smallest]]:
                    smallest = right
                if smallest != idx:
                    n_idx = heap[idx]
                    n_sm = heap[smallest]
                    heap[idx] = n_sm
                    heap[smallest] = n_idx
                    pos[n_sm] = idx
                    pos[n_idx] = smallest
                    idx = smallest
                else:
                    break
        pos[u] = -2  

        d_u = dist[u]
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if pos[v] == -2:
                continue
            new_d = d_u + weights[i]
            if new_d < dist[v]:
                dist[v] = new_d
                if pos[v] >= 0:
                    idx = pos[v]
                    while idx > 0:
                        parent = (idx - 1) // 2
                        if dist[heap[idx]] < dist[heap[parent]]:
                            n_idx = heap[idx]
                            n_p = heap[parent]
                            heap[idx] = n_p
                            heap[parent] = n_idx
                            pos[n_p] = idx
                            pos[n_idx] = parent
                            idx = parent
                        else:
                            break
                else:
                    idx = heap_size
                    heap_size += 1
                    heap[idx] = v
                    pos[v] = idx
                    while idx > 0:
                        parent = (idx - 1) // 2
                        if dist[heap[idx]] < dist[heap[parent]]:
                            n_idx = heap[idx]
                            n_p = heap[parent]
                            heap[idx] = n_p
                            heap[parent] = n_idx
                            pos[n_p] = idx
                            pos[n_idx] = parent
                            idx = parent
                        else:
                            break
    return dist

@njit
def bellman_ford_csr(indptr, indices, weights, source):
    N = len(indptr) - 1
    INF = np.int64(10**15)
    dist = np.full(N, INF, dtype=np.int64)
    dist[source] = 0

    rounds = 0
    for r in range(N - 1):
        rounds += 1
        updated = False
        for u in range(N):
            d_u = dist[u]
            if d_u == INF:
                continue
            for i in range(indptr[u], indptr[u+1]):
                v = indices[i]
                w = weights[i]
                if d_u + w < dist[v]:
                    dist[v] = d_u + w
                    updated = True
        if not updated:
            break
    return dist, rounds

@njit
def floyd_warshall_csr(indptr, indices, weights):
    N = len(indptr) - 1
    INF = np.int64(10**15)
    D = np.full((N, N), INF, dtype=np.int64)
    for u in range(N):
        D[u, u] = 0
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            w = weights[i]
            if w < D[u, v]:
                D[u, v] = w

    for k in range(N):
        for i in range(N):
            d_ik = D[i, k]
            if d_ik == INF:
                continue
            for j in range(N):
                d_kj = D[k, j]
                if d_kj != INF and d_ik + d_kj < D[i, j]:
                    D[i, j] = d_ik + d_kj
    return D

@njit
def tarjan_scc_iterative(indptr, indices):
    N = len(indptr) - 1
    indices_arr = indices

    index = np.full(N, -1, dtype=np.int32)
    lowlink = np.full(N, -1, dtype=np.int32)
    on_stack = np.zeros(N, dtype=np.bool_)
    scc = np.full(N, -1, dtype=np.int32)

    call_stack_u = np.empty(N, dtype=np.int32)
    call_stack_edge = np.empty(N, dtype=np.int32)
    call_top = 0

    tarjan_stack = np.empty(N, dtype=np.int32)
    tarjan_top = 0

    timer = 0
    scc_count = 0

    for start_node in range(N):
        if index[start_node] != -1:
            continue

        call_stack_u[call_top] = start_node
        call_stack_edge[call_top] = indptr[start_node]
        call_top += 1

        index[start_node] = timer
        lowlink[start_node] = timer
        timer += 1
        tarjan_stack[tarjan_top] = start_node
        tarjan_top += 1
        on_stack[start_node] = True

        while call_top > 0:
            frame_idx = call_top - 1
            u = call_stack_u[frame_idx]
            edge_pos = call_stack_edge[frame_idx]
            edge_end = indptr[u + 1]

            pushed_child = False
            while edge_pos < edge_end:
                v = indices_arr[edge_pos]
                edge_pos += 1
                call_stack_edge[frame_idx] = edge_pos

                if index[v] == -1:
                    index[v] = timer
                    lowlink[v] = timer
                    timer += 1
                    tarjan_stack[tarjan_top] = v
                    tarjan_top += 1
                    on_stack[v] = True

                    call_stack_u[call_top] = v
                    call_stack_edge[call_top] = indptr[v]
                    call_top += 1
                    pushed_child = True
                    break
                elif on_stack[v]:
                    if index[v] < lowlink[u]:
                        lowlink[u] = index[v]

            if pushed_child:
                continue

            if lowlink[u] == index[u]:
                while tarjan_top > 0:
                    tarjan_top -= 1
                    w = tarjan_stack[tarjan_top]
                    on_stack[w] = False
                    scc[w] = scc_count
                    if w == u:
                        break
                scc_count += 1

            call_top -= 1
            if call_top > 0:
                parent_idx = call_top - 1
                p = call_stack_u[parent_idx]
                if lowlink[u] < lowlink[p]:
                    lowlink[p] = lowlink[u]

    return scc, scc_count

@njit
def prim_mst_csr(indptr, indices, weights):
    N = len(indptr) - 1
    INF = np.int64(10**15)
    min_edge = np.full(N, INF, dtype=np.int64)
    min_edge[0] = 0
    in_tree = np.zeros(N, dtype=np.bool_)

    heap = np.empty(N, dtype=np.int32)
    pos = np.full(N, -1, dtype=np.int32)

    heap[0] = 0
    pos[0] = 0
    heap_size = 1

    total_weight = np.int64(0)
    edges_count = 0

    while heap_size > 0:
        u = heap[0]
        heap_size -= 1
        if heap_size > 0:
            last = heap[heap_size]
            heap[0] = last
            pos[last] = 0
            idx = 0
            while True:
                left = 2 * idx + 1
                right = 2 * idx + 2
                smallest = idx
                if left < heap_size and min_edge[heap[left]] < min_edge[heap[smallest]]:
                    smallest = left
                if right < heap_size and min_edge[heap[right]] < min_edge[heap[smallest]]:
                    smallest = right
                if smallest != idx:
                    n_idx = heap[idx]
                    n_sm = heap[smallest]
                    heap[idx] = n_sm
                    heap[smallest] = n_idx
                    pos[n_sm] = idx
                    pos[n_idx] = smallest
                    idx = smallest
                else:
                    break
        pos[u] = -2
        in_tree[u] = True
        total_weight += min_edge[u]
        edges_count += 1

        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if in_tree[v]:
                continue
            w = weights[i]
            if w < min_edge[v]:
                min_edge[v] = w
                if pos[v] >= 0:
                    idx = pos[v]
                    while idx > 0:
                        parent = (idx - 1) // 2
                        if min_edge[heap[idx]] < min_edge[heap[parent]]:
                            n_idx = heap[idx]
                            n_p = heap[parent]
                            heap[idx] = n_p
                            heap[parent] = n_idx
                            pos[n_p] = idx
                            pos[n_idx] = parent
                            idx = parent
                        else:
                            break
                else:
                    idx = heap_size
                    heap_size += 1
                    heap[idx] = v
                    pos[v] = idx
                    while idx > 0:
                        parent = (idx - 1) // 2
                        if min_edge[heap[idx]] < min_edge[heap[parent]]:
                            n_idx = heap[idx]
                            n_p = heap[parent]
                            heap[idx] = n_p
                            heap[parent] = n_idx
                            pos[n_p] = idx
                            pos[n_idx] = parent
                            idx = parent
                        else:
                            break

    return total_weight, edges_count

@njit
def kruskal_mst(N, u_arr, v_arr, w_arr):
    M = len(w_arr)
    count = np.zeros(12, dtype=np.int32)
    for i in range(M):
        count[w_arr[i] + 1] += 1
    for i in range(1, 12):
        count[i] += count[i - 1]

    sorted_idx = np.empty(M, dtype=np.int32)
    for i in range(M):
        w = w_arr[i]
        sorted_idx[count[w]] = i
        count[w] += 1

    parent = np.empty(N, dtype=np.int32)
    rank = np.zeros(N, dtype=np.int8)
    for i in range(N):
        parent[i] = i

    total_weight = np.int64(0)
    mst_edges = 0

    for i in range(M):
        e_idx = sorted_idx[i]
        u = u_arr[e_idx]
        v = v_arr[e_idx]
        w = w_arr[e_idx]

        curr_u = u
        while parent[curr_u] != curr_u:
            curr_u = parent[curr_u]
        p_u = u
        while parent[p_u] != curr_u:
            nxt = parent[p_u]
            parent[p_u] = curr_u
            p_u = nxt

        curr_v = v
        while parent[curr_v] != curr_v:
            curr_v = parent[curr_v]
        p_v = v
        while parent[p_v] != curr_v:
            nxt = parent[p_v]
            parent[p_v] = curr_v
            p_v = nxt

        if curr_u != curr_v:
            if rank[curr_u] < rank[curr_v]:
                parent[curr_u] = curr_v
            elif rank[curr_u] > rank[curr_v]:
                parent[curr_v] = curr_u
            else:
                parent[curr_v] = curr_u
                rank[curr_u] += 1
            total_weight += w
            mst_edges += 1
            if mst_edges == N - 1:
                break

    return total_weight, mst_edges

def main():
    print("=== TESTANDO CORREÇÃO DOS ALGORITMOS (SCRIPTS/DEV) ===")

    data_undir = np.load("data/processed/gplus_undirected_gcc.npz")
    indptr = data_undir["indptr"]
    indices = data_undir["indices"]
    N = len(indptr) - 1
    print(f"Grafo não-dirigido carregado: {N:,} nós, {len(indices)//2:,} arestas.")

    t0 = time.perf_counter()
    weights = generate_symmetric_weights_numba(indptr, indices, seed=42)
    print(f"1. Pesos simétricos gerados em {time.perf_counter()-t0:.2f}s (min={weights.min()}, max={weights.max()})")

    t0 = time.perf_counter()
    dist_bfs = bfs_csr(indptr, indices, 0)
    t_bfs = time.perf_counter() - t0
    print(f"2a. BFS (fonte 0): máx dist={dist_bfs.max()}, alcançados={np.sum(dist_bfs >= 0):,} em {t_bfs*1000:.2f}ms")

    t0 = time.perf_counter()
    visited_dfs, order_dfs = dfs_iterative_csr(indptr, indices, 0)
    t_dfs = time.perf_counter() - t0
    print(f"2b. DFS iterativo: nós visitados={len(order_dfs):,} em {t_dfs*1000:.2f}ms")
    assert len(order_dfs) == N, "DFS deve alcançar todos os nós da GCC!"

    t0 = time.perf_counter()
    is_eul, has_path, is_conn, odd_c, even_c = check_eulerian_numba(indptr, indices)
    print(f"3. Eulerianidade: Conexo={is_conn}, Graus ímpares={odd_c:,}, Graus pares={even_c:,}, Euleriano={is_eul}")

    t0 = time.perf_counter()
    dist_dijk = dijkstra_csr(indptr, indices, weights, 0)
    t_dijk = time.perf_counter() - t0
    print(f"4a. Dijkstra (pesos 1..10, fonte 0): máx dist={dist_dijk.max()} em {t_dijk*1000:.2f}ms")

    weights_one = np.ones(len(indices), dtype=np.int32)
    dist_dijk_one = dijkstra_csr(indptr, indices, weights_one, 0)
    diff_bfs_dijk = np.max(np.abs(dist_bfs.astype(np.int64) - dist_dijk_one))
    print(f"4b. Validação Dijkstra (pesos 1) == BFS: divergência máxima = {diff_bfs_dijk}")
    assert diff_bfs_dijk == 0, "Dijkstra com pesos 1 deve coincidir exatamente com a BFS!"

    sub_n = 200
    sub_nodes = order_dfs[:sub_n]
    sub_set = set(sub_nodes)
    node_map = {orig: new for new, orig in enumerate(sub_nodes)}

    sub_u = []
    sub_v = []
    sub_w = []
    for u in sub_nodes:
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if v in sub_set:
                sub_u.append(node_map[u])
                sub_v.append(node_map[v])
                sub_w.append(weights[i])

    sub_mat = csr_matrix((sub_w, (sub_u, sub_v)), shape=(sub_n, sub_n))
    sub_indptr = sub_mat.indptr
    sub_indices = sub_mat.indices
    sub_weights = sub_mat.data.astype(np.int32)

    print(f"\nSubgrafo de teste n={sub_n}: {sub_mat.nnz//2} arestas.")

    t0 = time.perf_counter()
    D_fw = floyd_warshall_csr(sub_indptr, sub_indices, sub_weights)
    t_fw = time.perf_counter() - t0
    print(f"5. Floyd-Warshall no subgrafo n={sub_n} em {t_fw*1000:.2f}ms")

    fw_match = True
    for s in range(sub_n):
        d_s = dijkstra_csr(sub_indptr, sub_indices, sub_weights, s)
        if not np.array_equal(d_s, D_fw[s]):
            fw_match = False
            break
    print(f"   Validação Floyd-Warshall == All-Pairs Dijkstra: {fw_match}")
    assert fw_match, "Floyd-Warshall deve coincidir com Dijkstra!"

    t0 = time.perf_counter()
    d_bf, rounds = bellman_ford_csr(sub_indptr, sub_indices, sub_weights, 0)
    t_bf = time.perf_counter() - t0
    d_dijk_sub = dijkstra_csr(sub_indptr, sub_indices, sub_weights, 0)
    bf_match = np.array_equal(d_bf, d_dijk_sub)
    print(f"6. Bellman-Ford no subgrafo n={sub_n}: rounds={rounds}, tempo={t_bf*1000:.2f}ms, match={bf_match}")
    assert bf_match, "Bellman-Ford deve coincidir com Dijkstra!"

    t0 = time.perf_counter()
    w_prim, e_prim = prim_mst_csr(sub_indptr, sub_indices, sub_weights)
    t_prim = time.perf_counter() - t0

    u_list = []
    v_list = []
    w_list = []
    for u in range(sub_n):
        for i in range(sub_indptr[u], sub_indptr[u+1]):
            v = sub_indices[i]
            if u < v:
                u_list.append(u)
                v_list.append(v)
                w_list.append(sub_weights[i])
    u_arr = np.array(u_list, dtype=np.int32)
    v_arr = np.array(v_list, dtype=np.int32)
    w_arr = np.array(w_list, dtype=np.int32)

    t0 = time.perf_counter()
    w_kruskal, e_kruskal = kruskal_mst(sub_n, u_arr, v_arr, w_arr)
    t_kruskal = time.perf_counter() - t0

    sp_mst = minimum_spanning_tree(sub_mat)
    sp_weight = int(sp_mst.sum())

    print(f"7. MST no subgrafo n={sub_n}:")
    print(f"   Prim: peso={w_prim}, arestas={e_prim} ({t_prim*1000:.2f}ms)")
    print(f"   Kruskal: peso={w_kruskal}, arestas={e_kruskal} ({t_kruskal*1000:.2f}ms)")
    print(f"   SciPy MST: peso={sp_weight}")
    assert w_prim == w_kruskal == sp_weight, "Pesos da MST de Prim, Kruskal e SciPy devem ser rigorosamente iguais!"

    data_dir = np.load("data/processed/gplus_directed.npz")
    d_indptr = data_dir["indptr"]
    d_indices = data_dir["indices"]
    N_dir = len(d_indptr) - 1
    print(f"\n8. Tarjan iterativo no dígrafo ({N_dir:,} nós, {len(d_indices):,} arcos dirigidos)...")
    t0 = time.perf_counter()
    scc, scc_count = tarjan_scc_iterative(d_indptr, d_indices)
    t_tarjan = time.perf_counter() - t0

    scc_sizes = np.bincount(scc)
    max_scc = scc_sizes.max()
    print(f"   Tarjan concluído em {t_tarjan:.2f}s!")
    print(f"   Total de SCCs: {scc_count:,}")
    print(f"   Maior SCC: {max_scc:,} nós ({max_scc/N_dir*100:.2f}%)")
    print(f"   Referência oficial SNAP para maior SCC: 69,501 nós (64.6%)")
    assert max_scc == 69501, f"Maior SCC deve ser 69.501, obtido {max_scc}"
    print("   CONCORDÂNCIA PERFEITA COM O SNAP (69.501 nós) CONFIRMADA!")

    print("\n=== TODOS OS TESTES PASSARAM COM SUCESSO! ===")

if __name__ == "__main__":
    main()
