import os
import sys
import json
import time
import pickle
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sys.path.append(os.path.abspath("."))

from src.graph_loader import load_csr_graph
from src.metrics import compute_all_parte1_metrics
from src.utils import (
    plot_degree_distribution,
    plot_ego_network_hub,
    build_status_table
)

def main():
    t_start = time.perf_counter()
    npz_path = "data/processed/gplus_undirected_gcc.npz"
    pickle_path = "data/processed/gplus_undirected_gcc.pickle"

    print("=== EXECUTANDO PIPELINE DA PARTE I (ANÁLISE ESTRUTURAL) ===")
    print(f"Carregando CSR de {npz_path}...")
    csr, node_ids = load_csr_graph(npz_path)

    metrics = compute_all_parte1_metrics(
        csr=csr,
        node_ids=node_ids,
        k_apl=1000,
        seed=42,
        pickle_path=pickle_path,
        verbose=True
    )

    print("\nGerando gráfico da distribuição de graus (PDF e CCDF)...")
    degrees = csr.indptr[1:] - csr.indptr[:-1]
    fig_deg_path = plot_degree_distribution(degrees, "figures/distribuicao_graus.png")
    print(f"  Salvo: {fig_deg_path}")

    hub_id = metrics["ego_hub_id"]
    print(f"\nExtraindo ego-network do hub (ID interno {hub_id}, grau {metrics['ego_hub_degree']})...")
    neighbors = csr.indices[csr.indptr[hub_id]:csr.indptr[hub_id+1]]
    ego_nodes = np.concatenate(([hub_id], neighbors))

    with open(pickle_path, "rb") as f:
        g = pickle.load(f)
    subg = g.induced_subgraph(ego_nodes)
    metrics["ego_hub_edges"] = subg.ecount()

    print(f"  Subgrafo induzido: {subg.vcount():,} nós e {subg.ecount():,} arestas.")
    fig_ego_path = plot_ego_network_hub(
        subg=subg,
        hub_idx=0,
        hub_label=f"Hub #{hub_id}",
        out_path="figures/visualizacao_ego_hub.png"
    )
    print(f"  Salvo: {fig_ego_path}")

    print("\nGerando Tabela de Conformidade do Enunciado...")
    df_status = build_status_table(metrics)
    csv_status_path = "results/parte1_tabela_status.csv"
    json_status_path = "results/parte1_tabela_status.json"
    df_status.to_csv(csv_status_path, index=False, encoding="utf-8-sig")
    df_status.to_json(json_status_path, orient="records", indent=2, force_ascii=False)
    print(f"  Tabela salva em: {csv_status_path} e {json_status_path}")

    metrics_json_path = "results/parte1_metricas_estruturais.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Métricas salvas em: {metrics_json_path}")

    validacao_json_path = "results/parte1_validacao_igraph.json"
    with open(validacao_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics["validacao_igraph"], f, indent=2, ensure_ascii=False)
    print(f"  Validação igraph salva em: {validacao_json_path}")

    t_total = time.perf_counter() - t_start
    print(f"\n=== PIPELINE DA PARTE I CONCLUÍDO COM SUCESSO EM {t_total:.2f}s ===")

if __name__ == "__main__":
    main()
