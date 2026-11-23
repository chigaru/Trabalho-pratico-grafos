# Resumo Executivo — Parte III: Itens 4 e 5

**Aluno:** Vitor Renato Santos dos Santos | **Matrícula:** 222217109 — UFBA
**Dataset:** ego-Gplus (G2) | $n = 107.614$, $m = 12.238.285$

---

## Item 4: Reciprocidade do Dígrafo Original

### Métricas Globais
| Métrica | Valor |
|---------|-------|
| Pares recíprocos | 1,435,168 |
| Fração de arcos em laços bidirecionais | 20.99% |
| Validação: arcos − pares = m_undirected | 12,238,285 = 12,238,285 → PASSOU ✓ |
| Reciprocidade esperada (dígrafo aleatório) | 0.001181 (ρ do dígrafo) |
| Razão observado / esperado | **177.79×** |
| Correlação in-degree vs out-degree | r = 0.3207 |
| Nós com reciprocidade total | 4,391 (4.08%) |
| Nós sem nenhum arco recíproco | 6,943 (6.45%) |

### Relação com a SCC
A maior SCC tem 69,501 nós (64.58%).
99.93% da reciprocidade (arcos recíprocos de saída) está concentrada na maior SCC.

### Reciprocidade por Faixa de Grau
Nós de **baixo grau** (periféricos) tendem a ter reciprocidade mais **alta**: seus poucos arcos costumam ser bidirecionais.
Hubs (grau alto) têm reciprocidade mais **baixa** em fração, pois acumulam muitos seguidores unilaterais.
Essa relação inversa é consistente com a dinâmica de redes sociais onde a conexão com celebridades é assimétrica.

> **Hipótese (não verificada formalmente):** a alta reciprocidade nos nós de baixo grau pode refletir "círculos sociais" íntimos onde conexões tendem a ser mútuas, enquanto hubs acumulam seguidores de sentido único.

---

## Item 5: Modelos Teóricos vs. Grafo Real

### Tabela Comparativa

| Modelo | n | m | ⟨k⟩ | k_max | L | C | γ (Clauset) |
|--------|---|---|-----|-------|---|---|-------------|
| Real | 107,614 | 12,238,285 | 227.45 | 20,127 | 2.8603 | 0.490118 | ~4.56 (Clauset kmin=4496, caud… |
| ER | 107,614 | 12,238,285 | 227.45 | 317 | 2.6165 | 0.002112 | N/A (ajuste inválido: kmin=10 abaixo do suporte do modelo) |
| BA | 107,614 | 12,261,441 | 227.88 | 5,113 | 2.4738 | 0.011384 | N/A (ajuste inválido: kmin=10 abaixo do suporte do modelo; γ teórico = 3) |
| WS_b0.01 | 107,614 | 12,267,723 | 227.99 | 235 | 3.3069 | 0.72463 | N/A (WS não produz cauda pesad… |
| WS_b0.1 | 107,614 | 12,265,488 | 227.95 | 249 | 2.9065 | 0.544847 | N/A (WS não produz cauda pesad… |
| WS_b0.5 | 107,614 | 12,258,202 | 227.82 | 269 | 2.6917 | 0.095132 | N/A (WS não produz cauda pesad… |

### Validação com igraph (n = 2.000)
Todos os três geradores (ER, BA, WS) produziram grafos cujo m e C coincidem com os valores do igraph (diferença < 0.01 em C).

### Conclusão Honesta: Qual Modelo Reproduz o Quê

| Propriedade | ER | BA | WS |
|-------------|----|----|-----|
| **L pequeno** (≈ 2.86) | ✓ L ≈ L_real | ✓ L comparável | Depende de β (β alto → L baixo) |
| **C alto** (≈ 0.49) | ✗ C ≈ 0.002 (200× menor) | ✗ C baixo | ✓ C alto para β baixo |
| **Cauda pesada** | ✗ Poisson | ✓ Power-law | ✗ Aproximadamente regular |

**Nenhum modelo isolado reproduz todas as três propriedades** do grafo real:
- **ER** acerta L mas falha em C e na cauda.
- **BA** produz cauda pesada e L curto, mas C é muito baixo.
- **WS** (β baixo) acerta C alto e L moderado, mas não produz cauda pesada.

O grafo G2 combina L curto, C muito alto e distribuição de graus com cauda pesada (embora não power-law pura). Essa combinação é mais bem descrita por modelos híbridos (e.g., Holme-Kim) que adicionam fechamento triádico a BA, ou por modelos de comunidades.

---

## Limitações e Ressalvas

1. **APL por amostragem:** L medido com k=1.000 fontes (semente 42); erro amostral ≈ 1–2%.
2. **Clustering:** C calculado no grafo completo (exato), mas para modelos pode variar com a semente.
3. **Parâmetros dos modelos:** m_attach=114 (BA) e k=228 (WS) calibrados para igualar ⟨k⟩ do real; outros valores poderiam dar resultados diferentes.
4. **Seed fixa (42):** uma única realização por modelo; variância entre sementes não avaliada.
5. **γ Clauset nos modelos:** para ER e WS, reportar γ de cauda não faz sentido teórico — incluído apenas para completude.
6. **Reciprocidade por faixa:** a métrica é a média da fração recíproca por nó; nós com out_deg=0 excluídos.

---

**Tempo total de execução:** 285.26s (modelos) + 6.32s (reciprocidade).
