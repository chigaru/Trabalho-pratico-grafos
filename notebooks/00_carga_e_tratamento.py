import os
import sys
import time
import json
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph as ig

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from src.graph_loader import load_and_process_gplus, save_processed_data, generate_directed_npz

RAW_DATA_PATH = "data/raw/gplus_combined.txt.gz"
if not os.path.exists(RAW_DATA_PATH):
    RAW_DATA_PATH = "gplus_combined.txt.gz"

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

TEST_MODE = False if "--full" in sys.argv else True
MAX_LINES = 1_000_000 if TEST_MODE else None

print(f"Arquivo de entrada: {RAW_DATA_PATH}")
print(f"Modo de execução: {'TESTE (1.000.000 linhas)' if TEST_MODE else 'COMPLETO (30.494.866 linhas)'}")

t_start = time.perf_counter()

gcc, csr_gcc, stats, node_map = load_and_process_gplus(
    filepath=RAW_DATA_PATH,
    max_lines=MAX_LINES,
    verbose=True
)

elapsed_total = time.perf_counter() - t_start
print(f"\nPipeline concluído com sucesso em {elapsed_total:.2f} segundos!")

antes = stats["antes"]
depois = stats["depois"]

tabela_dados = [
    ("Linhas brutas processadas", f"{antes['total_linhas_lidas']:,}", "—"),
    ("Linhas duplicadas (sobreposição)", f"{antes['linhas_duplicadas']:,}", "0 (removidas)"),
    ("Self-loops detectados", f"{antes['self_loops']:,}", "0 (removidos)"),
    ("Tipo do grafo", "Dígrafo (Dirigido)", "Grafo Simples Não-Dirigido"),
    ("Arcos dirigidos únicos", f"{antes['arcos_dirigidos_unicos']:,}", "—"),
    ("Pares recíprocos (a↔b)", f"{antes['pares_reciprocos']:,}", f"Colapsados em {depois['arestas_nao_dirigidas_gcc']:,} arestas"),
    ("Taxa de reciprocidade de arcos", f"{antes['taxa_reciprocidade_arcos_pct']:.2f}%", "Simétrico (100%)"),
    ("Vértices", f"{antes['vertices_unicos']:,}", f"{depois['vertices_gcc']:,} (GCC: {depois['fracao_gcc_pct']:.2f}%)"),
    ("Arestas não-dirigidas", "—", f"{depois['arestas_nao_dirigidas_gcc']:,}"),
    ("Componentes conexas", "—", f"{depois['num_componentes_conexas']:,}"),
    ("Densidade", "—", f"{depois['densidade']:.6e}"),
    ("Grau mín / médio / máx", "—", f"{depois['grau_minimo']} / {depois['grau_medio']:.1f} / {depois['grau_maximo']:,}"),
]

df_comparativo = pd.DataFrame(tabela_dados, columns=["Métrica / Característica", "Antes (Bruto / Dígrafo)", "Depois (Tratado / GCC)"])
print(df_comparativo.to_string(index=False))

is_simple = gcc.is_simple()
is_connected = gcc.is_connected()
csr_shape = csr_gcc.shape
csr_nnz = csr_gcc.nnz
expected_nnz = 2 * gcc.ecount()

print(f"O grafo tratado é simples? {is_simple}")
print(f"O grafo tratado é conexo? {is_connected}")
print(f"Dimensão da matriz CSR: {csr_shape}")
print(f"Não-nulos na CSR: {csr_nnz:,} (Esperado: {expected_nnz:,} = 2 × arestas)")

assert is_simple, "Erro: Grafo tratado contém laços ou multi-arestas!"
assert is_connected, "Erro: Grafo tratado não é estritamente conexo!"
assert csr_nnz == expected_nnz, "Erro: Inconsistência no número de elementos não-nulos da CSR!"
print("Todos os testes de integridade passaram com sucesso!")

prefix = "gplus_undirected_gcc_test" if TEST_MODE else "gplus_undirected_gcc"

saved_files = save_processed_data(
    gcc=gcc,
    csr_gcc=csr_gcc,
    stats=stats,
    node_map=node_map,
    output_dir=PROCESSED_DIR,
    prefix=prefix
)

print("Arquivos persistidos com sucesso:")
for k, path in saved_files.items():
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [{k}] {path} ({file_size_mb:.2f} MB)")

directed_npz_path = os.path.join(PROCESSED_DIR, "gplus_directed.npz")
if not os.path.exists(directed_npz_path):
    print("Gerando representação CSR do dígrafo original...")
    rev_nodes = np.empty(len(node_map), dtype="U21")
    for raw_id, int_id in node_map.items():
        if int_id < len(rev_nodes):
            rev_nodes[int_id] = raw_id
    generate_directed_npz(raw_path=RAW_DATA_PATH, node_ids=rev_nodes, output_path=directed_npz_path)
else:
    print(f"Dígrafo original já existe em: {directed_npz_path}")

d_dir = np.load(directed_npz_path)
d_undir = np.load(saved_files["npz"], allow_pickle=True)

print(f"Arcos no dígrafo: {len(d_dir['indices']):,} (Esperado: 13.673.453)")
print(f"Mapeamento de nós idêntico? {np.array_equal(d_dir['node_ids'], d_undir['node_ids'])}")
assert len(d_dir['indices']) == 13_673_453, "Erro: Contagem de arcos dirigidos incorreta!"
assert np.array_equal(d_dir['node_ids'], d_undir['node_ids']), "Erro: Mapeamento de nós difere do grafo não-dirigido!"
print("Dígrafo original validado com sucesso!")
