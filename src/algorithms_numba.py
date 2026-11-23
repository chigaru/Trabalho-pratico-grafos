import numpy as np
from numba import njit, prange

@njit
def bfs_single_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    source: int
) -> tuple:
    N = len(indptr) - 1
    dist = np.full(N, -1, dtype=np.int32)
    queue = np.empty(N, dtype=np.int32)

    dist[source] = 0
    queue[0] = source
    head = 0
    tail = 1

    max_dist = 0
    farthest_node = source
    sum_dist = np.int64(0)

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
                if d_next > max_dist:
                    max_dist = d_next
                    farthest_node = v
                sum_dist += d_next

    return max_dist, farthest_node, sum_dist, tail

@njit
def double_sweep_diameter(
    indptr: np.ndarray,
    indices: np.ndarray,
    num_sweeps: int = 10,
    initial_node: int = 0
) -> tuple:
    curr = initial_node
    max_diam = 0
    best_src = curr
    best_dst = curr
    history = np.zeros(num_sweeps, dtype=np.int32)

    for sweep in range(num_sweeps):
        d, farthest, _, _ = bfs_single_csr(indptr, indices, curr)
        history[sweep] = d
        if d > max_diam:
            max_diam = d
            best_src = curr
            best_dst = farthest
        if farthest == curr:
            break
        curr = farthest

    return max_diam, best_src, best_dst, history

@njit
def sample_apl_and_eccentricities(
    indptr: np.ndarray,
    indices: np.ndarray,
    sources: np.ndarray
) -> tuple:
    N = len(indptr) - 1
    K = len(sources)
    dist = np.full(N, -1, dtype=np.int32)
    queue = np.empty(N, dtype=np.int32)

    mean_distances = np.empty(K, dtype=np.float64)
    eccentricities = np.empty(K, dtype=np.int32)

    for k in range(K):
        s = sources[k]
        dist.fill(-1)
        dist[s] = 0
        queue[0] = s
        head = 0
        tail = 1

        max_d = 0
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
                    if d_next > max_d:
                        max_d = d_next
                    sum_d += d_next

        mean_distances[k] = sum_d / (tail - 1) if tail > 1 else 0.0
        eccentricities[k] = max_d

    return mean_distances, eccentricities

@njit(parallel=True)
def compute_triangles_and_clustering(
    indptr: np.ndarray,
    indices: np.ndarray
) -> tuple:
    N = len(indptr) - 1
    triangles_per_node = np.zeros(N, dtype=np.int64)
    clustering_per_node = np.zeros(N, dtype=np.float64)

    for u in prange(N):
        u_start = indptr[u]
        u_end = indptr[u + 1]
        deg_u = u_end - u_start

        if deg_u < 2:
            continue

        tri_u = 0
        for i in range(u_start, u_end):
            v = indices[i]
            v_start = indptr[v]
            v_end = indptr[v + 1]

            p1 = u_start
            p2 = v_start
            common = 0
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
            tri_u += common

        t = tri_u // 2
        triangles_per_node[u] = t
        clustering_per_node[u] = (2.0 * t) / (deg_u * (deg_u - 1))

    return triangles_per_node, clustering_per_node

@njit
def generate_symmetric_weights_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    seed: int = 42
) -> np.ndarray:
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
def bfs_distances_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    source: int
) -> np.ndarray:
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
def dfs_iterative_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    source: int
) -> tuple:
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
def check_eulerian_numba(
    indptr: np.ndarray,
    indices: np.ndarray
) -> tuple:
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
def dijkstra_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    source: int
) -> np.ndarray:
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
def bellman_ford_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    source: int
) -> tuple:
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
def floyd_warshall_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
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
def tarjan_scc_iterative(
    indptr: np.ndarray,
    indices: np.ndarray
) -> tuple:
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
def prim_mst_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray
) -> tuple:
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
def kruskal_mst(
    N: int,
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    w_arr: np.ndarray
) -> tuple:
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
