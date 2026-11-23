import gzip
import os
import json
import pickle
import time
from typing import Dict, Tuple, Optional, Any
import numpy as np
import scipy.sparse as sp
import igraph as ig

def load_and_process_gplus(
    filepath: str,
    max_lines: Optional[int] = None,
    verbose: bool = True
) -> Tuple[ig.Graph, sp.csr_matrix, Dict[str, Any], Dict[str, int]]:
    t0 = time.perf_counter()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    node_map: Dict[str, int] = {}
    directed_edges = set()
    reciprocal_pairs_count = 0
    self_loops_count = 0
    duplicate_lines_count = 0
    total_lines = 0

    if verbose:
        mode_str = f"primeiras {max_lines:,} linhas" if max_lines else "arquivo completo (30.5M linhas)"
        print(f"Iniciando carga em streaming ({mode_str}): {filepath}")

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if max_lines and total_lines > max_lines:
                total_lines -= 1
                break

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            src_str, dst_str = parts[0], parts[1]

            if src_str == dst_str:
                self_loops_count += 1
                continue

            u = node_map.setdefault(src_str, len(node_map))
            v = node_map.setdefault(dst_str, len(node_map))

            edge_key = (u << 32) | v
            if edge_key in directed_edges:
                duplicate_lines_count += 1
                continue

            directed_edges.add(edge_key)

            rev_key = (v << 32) | u
            if rev_key in directed_edges:
                reciprocal_pairs_count += 1

            if verbose and total_lines % 5_000_000 == 0:
                print(f"  Linhas lidas: {total_lines:,} | Nós únicos: {len(node_map):,} | Arcos únicos: {len(directed_edges):,}")

    t_read = time.perf_counter() - t0
    num_nodes_raw = len(node_map)
    unique_directed_arcs = len(directed_edges)
    reciprocity_rate = (2.0 * reciprocal_pairs_count / unique_directed_arcs) if unique_directed_arcs > 0 else 0.0

    if verbose:
        print(f"Leitura concluída em {t_read:.2f}s:")
        print(f"  Total de linhas processadas: {total_lines:,}")
        print(f"  Linhas duplicadas ignoradas: {duplicate_lines_count:,}")
        print(f"  Self-loops detectados: {self_loops_count:,}")
        print(f"  Nós únicos encontrados: {num_nodes_raw:,}")
        print(f"  Arcos dirigidos únicos: {unique_directed_arcs:,}")
        print(f"  Pares recíprocos: {reciprocal_pairs_count:,} (taxa de arcos recíprocos: {reciprocity_rate*100:.2f}%)")

    t1 = time.perf_counter()
    undirected_edges = []
    for k in directed_edges:
        u = k >> 32
        v = k & 0xFFFFFFFF
        if u < v:
            undirected_edges.append((u, v))
        elif ((v << 32) | u) not in directed_edges:
            undirected_edges.append((v, u))

    unique_undirected_edges = len(undirected_edges)
    del directed_edges

    if verbose:
        print(f"Conversão para não-dirigido concluída em {time.perf_counter()-t1:.2f}s:")
        print(f"  Arestas não-dirigidas simples: {unique_undirected_edges:,}")

    t2 = time.perf_counter()
    g = ig.Graph(n=num_nodes_raw, edges=undirected_edges, directed=False)
    del undirected_edges

    components = g.connected_components()
    num_components = len(components)
    gcc = components.giant()
    gcc_nodes = gcc.vcount()
    gcc_edges = gcc.ecount()
    gcc_fraction = (gcc_nodes / num_nodes_raw) * 100.0 if num_nodes_raw > 0 else 0.0

    if verbose:
        print(f"Construção do grafo e GCC concluída em {time.perf_counter()-t2:.2f}s:")
        print(f"  Componentes conexas: {num_components:,}")
        print(f"  Tamanho da GCC: {gcc_nodes:,} nós ({gcc_fraction:.2f}%) | {gcc_edges:,} arestas")

    csr_gcc = gcc.get_adjacency_sparse()

    degrees = np.array(gcc.degree())
    deg_min = int(np.min(degrees)) if len(degrees) > 0 else 0
    deg_max = int(np.max(degrees)) if len(degrees) > 0 else 0
    deg_mean = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
    density = (2.0 * gcc_edges) / (gcc_nodes * (gcc_nodes - 1)) if gcc_nodes > 1 else 0.0

    stats = {
        "antes": {
            "total_linhas_lidas": total_lines,
            "linhas_duplicadas": duplicate_lines_count,
            "self_loops": self_loops_count,
            "vertices_unicos": num_nodes_raw,
            "arcos_dirigidos_unicos": unique_directed_arcs,
            "pares_reciprocos": reciprocal_pairs_count,
            "taxa_reciprocidade_arcos_pct": round(reciprocity_rate * 100, 2),
        },
        "depois": {
            "vertices_gcc": gcc_nodes,
            "arestas_nao_dirigidas_gcc": gcc_edges,
            "fracao_gcc_pct": round(gcc_fraction, 2),
            "num_componentes_conexas": num_components,
            "densidade": density,
            "grau_minimo": deg_min,
            "grau_medio": round(deg_mean, 2),
            "grau_maximo": deg_max,
        },
        "meta": {
            "tempo_leitura_s": round(t_read, 2),
            "tempo_total_s": round(time.perf_counter() - t0, 2),
            "max_lines": max_lines,
        }
    }

    return gcc, csr_gcc, stats, node_map

def save_processed_data(
    gcc: ig.Graph,
    csr_gcc: sp.csr_matrix,
    stats: Dict[str, Any],
    node_map: Dict[str, int],
    output_dir: str = "data/processed",
    prefix: str = "gplus_undirected_gcc"
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    pickle_path = os.path.join(output_dir, f"{prefix}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(gcc, f, protocol=pickle.HIGHEST_PROTOCOL)
    saved_paths["pickle"] = pickle_path

    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    rev_nodes = np.empty(len(node_map), dtype="U21")
    for raw_id, int_id in node_map.items():
        if int_id < len(rev_nodes):
            rev_nodes[int_id] = raw_id

    np.savez_compressed(
        npz_path,
        indptr=csr_gcc.indptr,
        indices=csr_gcc.indices,
        node_ids=rev_nodes,
        shape=csr_gcc.shape
    )
    saved_paths["npz"] = npz_path

    stats_path = os.path.join(output_dir, "stats_antes_depois.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    saved_paths["stats_json"] = stats_path

    return saved_paths

def generate_directed_npz(
    raw_path: str,
    node_ids: np.ndarray,
    output_path: str = "data/processed/gplus_directed.npz",
    verbose: bool = True
) -> str:
    N = len(node_ids)
    id_to_int = {str(nid): i for i, nid in enumerate(node_ids)}

    if verbose:
        print(f"Lendo {raw_path} para extrair o dígrafo original...")

    directed_edges = set()
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0] == parts[1]:
                continue
            u = id_to_int[parts[0]]
            v = id_to_int[parts[1]]
            directed_edges.add((u << 32) | v)

    num_arcs = len(directed_edges)
    if verbose:
        print(f"  Arcos dirigidos únicos identificados: {num_arcs:,}")

    src_arr = np.empty(num_arcs, dtype=np.int32)
    dst_arr = np.empty(num_arcs, dtype=np.int32)
    for i, edge in enumerate(directed_edges):
        src_arr[i] = edge >> 32
        dst_arr[i] = edge & 0xFFFFFFFF
    del directed_edges

    csr_directed = sp.csr_matrix(
        (np.ones(num_arcs, dtype=np.bool_), (src_arr, dst_arr)),
        shape=(N, N)
    )
    csr_directed.sort_indices()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        indptr=csr_directed.indptr,
        indices=csr_directed.indices,
        node_ids=np.array(node_ids, dtype="U21"),
        shape=csr_directed.shape
    )
    if verbose:
        print(f"Dígrafo salvo em {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")
    return output_path

def load_csr_graph(npz_path: str) -> Tuple[sp.csr_matrix, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    node_ids = d["node_ids"] if "node_ids" in d else d["node_map"]
    shape = tuple(d["shape"])
    csr = sp.csr_matrix((np.ones(len(d["indices"]), dtype=np.bool_), d["indices"], d["indptr"]), shape=shape)
    return csr, node_ids
