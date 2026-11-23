# Resumo Executivo da Parte III: Análise Estrutural Avançada

**Aluno:** Vitor Renato Santos dos Santos | **Matrícula:** 222217109 — UFBA  
**Dataset:** ego-Gplus (Google+ social circles) — SNAP | **Grafo:** G2 ($n = 107,614$, $m = 12,238,285$)

---

## 1. Small-World (Apêndice A — Watts & Strogatz)

### Tabela de Métricas Comparativas
| Métrica | Grafo Real $G$ | Grafo Aleatório Teórico $G(n, m)$ | Grafo Aleatório Empírico $G(n, m)$ |
| :--- | :---: | :---: | :---: |
| **Comprimento Médio ($L$)** | **2.8603** | 2.1350 | **2.6188** |
| **Clusterização Média ($C$)** | **0.490118** | 0.002114 | **0.002112** |
| **Razão $L / L_{\text{rand}}$** | — | 1.3397 | **1.0922** |
| **Razão $C / C_{\text{rand}}$** | — | 231.89 | **232.08** |
| **Índice Small-World ($\sigma$)** | — | **173.09** | **212.48** |

### Rascunho da Resposta (Pergunta 1 da Parte III)
A rede ego-Gplus satisfaz rigorosamente a definição de rede de mundo pequeno (*small-world*) proposta por Watts & Strogatz. O comprimento médio dos caminhos ($L = 2.8603$) é da mesma ordem de grandeza da rede aleatória equivalente ($L / L_{\text{rand}} = 1.0922 \approx 1$), enquanto o coeficiente de clusterização local ($C = 0.490118$) é extraordinariamente superior ao esperado puramente ao acaso ($C / C_{\text{rand}} = 232.08 \gg 1$), resultando em um coeficiente de mundo pequeno $\sigma = 212.48 \gg 1$. Na prática social, essa topologia reflete a convivência entre círculos sociais altamente coesos e fechados (onde amigos em comum tendem fortemente a ser amigos entre si) e a existência de conexões de atalho (*weak ties*) que encurtam radicalmente a distância geodésica entre quaisquer dois indivíduos no Google+.

---

## 2. Lei de Potência (Apêndice B — Clauset, Shalizi & Newman)

### Tabela de Comparação de Modelos na Cauda
| Cenário / Limiar | Modelo Ajustado | Parâmetros Estimados | Distância KS ($D$) | Log-Likelihood | AIC | $\Delta$ AIC vs PL |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Mínimo Global Clauset**<br>($k_{\min} = 4496$, $N_{\text{cauda}} = 328$, 0.30%) | **Power Law** | $\gamma = 4.5618$ | 0.0349 | -2762.19 | 5526.39 | — |
| | **Log-Normal** | $\mu = 5.92, \sigma = 0.92$ | 0.0307 | -2761.14 | 5526.28 | **0.11** |
| | **Exponencial** | $\lambda^{-1} = 1693.85$ | 0.0715 | -2766.60 | 5535.20 | -8.81 |
| **Regime Representativo**<br>($k_{\min} = 1.500$, $N_{\text{cauda}} = 3,483$, 3.24%) | **Power Law** | $\gamma = 2.9878$ | 0.0652 | -28312.96 | 56627.91 | — |
| | **Log-Normal** | $\mu = 7.01, \sigma = 0.75$ | **0.0099** | **-28205.08** | **56414.16** | **213.75** |
| | **Exponencial** | $\lambda^{-1} = 1224.59$ | 0.0472 | -28248.39 | 56498.78 | 129.13 |

### Rascunho da Resposta (Pergunta 2 da Parte III)
A distribuição de conectividade do ego-Gplus não segue uma lei de potência pura em sua cauda, apresentando nítida curvatura convexa na CCDF com declínio acelerado acima de $k \approx 10^3$. A varredura sistemática de Clauset minimiza o teste KS somente em $k_{\min} = 4496$, resultando em um expoente $\gamma = 4.56$ atipicamente íngreme (fora da faixa canônica de redes livres de escala $2 < \gamma < 3$) e restringindo a cauda a apenas 0.30% dos vértices. No limiar de Clauset (k_min = 4496, 328 nós) os três modelos ficam praticamente empatados (ΔAIC da log-normal = 0,11); a vantagem da log-normal vale para o regime k ≥ 1500, escolhido por nós. Ao confrontar os modelos teóricos concorrentes para $k \geq 1.500$, a distribuição Log-Normal supera a Lei de Potência de forma decisiva (KS de 0.0099 contra 0.0652; $\Delta\text{AIC} = 213.8$ a favor da Log-Normal), o que sugere, como hipótese não testada, que a rede é governada por múltiplos processos multiplicativos e saturação cognitiva de conexões em vez de anexo preferencial irrestrito.

---

## 3. Robustez Topológica e Ataques (Apêndice C)

### Tabela de Comparação Estrutural (Remoção de $r = 5,381$ vértices — 5,0%)
| Cenário Experimental | Fração GCC ($S$) | Nº Componentes ($B$) | APL na GCC ($C$) | Diâmetro GCC ($C$) | Fração Isolados (% dos sobreviventes) ($D$) | Nós na GCC | Nós Isolados |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grafo Base (Intacto)** | **100,00%** | **1** | **2.8612** | **6** | **0,00%** | **107,614** | **0** |
| **Falha Aleatória ($T=30$)** | **94.74% ± 0.23%** | **275.9 ± 241.3** | **2.8801 ± 0.0381** | **7.1 ± 0.9** | **0.27% ± 0.23%** | 101,954 | 272 |
| **Ataque: Grau (Top 5%)** | **88.51%** | **6,890** | **3.6185** | **11** | **6.67%** | **95,254** | **6,816** |
| **Ataque: Betweenness (Top 5%)** | **84.83%** | **10,582** | **3.7417** | **14** | **10.21%** | **91,291** | **10,438** |

### Rascunho da Resposta (Pergunta 3 da Parte III)
A rede exibe o comportamento clássico de robustez assimétrica frente a perdas nodais: alta tolerância a falhas estocásticas, mas extrema fragilidade a ataques direcionados. Sob remoção aleatória de 5% dos nós, a componente gigante retém $S_{\text{rand}} = 94.74\%$ da rede (99.73% dos nós sobreviventes), o diâmetro varia discretamente (7.1) e formam-se em média apenas 276 fragmentos. Em contrapartida, a eliminação dos 5% nós mais conectados ($S_{\text{deg}} = 88.51\%$) e de maior intermediação ($S_{\text{bet}} = 84.83\%$) provoca fragmentação severa, originando 6,890 e 10,582 componentes desconexas e elevando o diâmetro para 11 e 14. Esse colapso acentuado ($S_{\text{rand}} \gg S_{\text{cent}}$) decorre da dependência topológica em relação a hubs centrais e pontes estruturais responsáveis por manter interligados os aglomerados de círculos egoicos.
