# Trabalho Prático de Grafos — ego-Gplus (G2)

Repositório do Trabalho Prático da disciplina de Teoria dos Grafos.  
**Aluno:** Vitor Renato Santos dos Santos | **Matrícula:** 222217109 — UFBA  
**Dataset:** ego-Gplus (Google+ social circles) — SNAP ($n = 107.614$, $m = 12.238.285$).

---

## 1. Ambiente, Versão do Python e Instalação

Ambiente utilizado e validado nos experimentos:
- **Versão do Python (`python --version`):** `Python 3.14.7` (64-bit AMD64, Windows 11 Build 26300)
- **Hardware:** AMD Ryzen 5 5600GT (6 núcleos físicos / 12 threads), 16 GB RAM.
- **Observação importante de compatibilidade:** `numba` e `igraph` possuem vínculos nativos (LLVM / C) sensíveis à versão do interpretador Python; para reproduzir o ambiente sem conflitos de compilação, utilize `Python 3.14.7` com as versões exatas fixadas em `requirements.txt`.

Comandos para recriar e ativar o ambiente virtual no Windows (PowerShell):

```powershell
python --version
# Saída esperada: Python 3.14.7

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pacotes instalados com versões fixadas (`requirements.txt`): `numpy==2.5.5`, `scipy==1.18.1`, `pandas==3.1.1`, `numba==0.68.1`, `matplotlib==3.11.2`, `igraph==1.0.0`, `jupytext==1.19.7`.

---

## 2. Ordem de Execução e Reprodução dos Resultados

Os dados brutos processados encontram-se em `data/processed/` (`gplus_undirected_gcc.npz`, `gplus_directed.npz`).

A reprodução completa ou modular deve seguir a ordem abaixo:

### Etapa 0 — Carga e Pré-processamento
- **Notebook:** `notebooks/00_carga_e_tratamento.ipynb` (ou script correspondente)
- **Função:** Lê o arquivo original `gplus_combined.txt.gz`, extrai a maior componente conexa (GCC) e serializa os grafos dirigidos e não-dirigidos em formato CSR comprimido (`.npz`).
- **Tempo estimado:** ~2 min.

### Etapa 1 — Parte I: Análise Estrutural Básica
- **Script:** `python scripts/run_parte1_metrics.py`
- **Figuras auxiliares:** `python scripts/make_figs_parte1.py`
- **Resultados:** `results/parte1_metricas_estruturais.json`, `results/parte1_tabela_status.csv`, `figures/distribuicao_graus.png`.
- **Tempo estimado:** ~3 min.

### Etapa 2 — Parte II: Benchmarks de Algoritmos em Grafos
- **Script principal:** `python scripts/run_parte2_benchmarks.py`
- **Script de extras e repetições:** `python scripts/run_parte2_extras.py`
- **Figuras dedicadas:** `python scripts/make_fig_fw_mst.py`
- **Resultados:** `results/parte2_tempos_tabela_final.csv`, `figures/parte2_floyd_warshall.png`, `figures/parte2_mst_prim_kruskal.png`, `figures/parte2_escala_algoritmos.png`.
- **Tempo estimado:** ~6 min (compilação Numba + execuções repetidas com IC 95%).

### Etapa 3 — Parte III: Propriedades Avançadas (Itens 1 a 3)
- **Script principal:** `python scripts/run_parte3.py`
- **Função:** Teste de Watts-Strogatz (Small-World), ajuste de cauda pesada de Clauset (Power Law vs Log-Normal vs Exponencial) e simulação de robustez topológica (falhas aleatórias $T=30$ vs ataques dirigidos por grau e betweenness).
- **Resultados:** `results/parte3_small_world.json`, `results/parte3_power_law.json`, `results/parte3_robustez.json`, `results/parte3_resumo.md`.
- **Tempo estimado:** ~15 a 25 min (devido às BFSs amostrais e simulações de robustez).

### Etapa 4 — Parte III Extra: Reciprocidade e Modelos Clássicos (Itens 4 e 5)
- **Script principal:** `python scripts/run_parte3_extra.py`
- **Figura dedicada:** `python scripts/make_fig_extra_barras.py`
- **Resultados:** `results/parte3_reciprocidade.json`, `results/parte3_extra_modelos.json`, `results/parte3_extra_tabela_comparativa.csv`, `results/parte3_extra_resumo.md`, `figures/parte3_extra_L_C_barras.png`.
- **Tempo estimado:** ~4 a 6 min.

---

## 3. Estrutura do Repositório

- `data/processed/`: Grafos serializados em formato NumPy `.npz` (CSR).
- `figures/`: Figuras finais de alta resolução (200+ DPI).
- `figures/_obsoletas/`: Figuras históricas ou substituídas mantidas para rastreabilidade.
- `notebooks/`: Notebooks Jupyter sincronizados (`.ipynb` e `.py` via Jupytext).
- `results/`: Arquivos `.csv`, `.json` e `.md` contendo todos os dados numéricos brutos consolidados.
- `scripts/`: Códigos de execução e geração reprodutível de gráficos.
