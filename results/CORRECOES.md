# Registro de Correções Textuais e Metodológicas (Fechamento Técnico)

Data: 22/11/2026

## 1. Ajuste dos Coeficientes γ dos Modelos Teóricos ER e BA
- **Arquivos modificados:**
  - `results/parte3_extra_tabela_comparativa.csv`
  - `results/parte3_extra_modelos.json`
  - `results/parte3_extra_resumo.md`
- **Valores anteriores:**
  - ER: `γ = 1.3151`
  - BA: `γ = 1.3349` (e `gamma_ks = 0.5649` no JSON)
- **Motivação:**
  - Os valores anteriores foram gerados por uma chamada cega de ajuste com limiar fixo $k_{\min}=10$. Em ER, o grau médio é $\approx 227$ e o suporte não possui cauda pesada. Em BA, os nós gerados têm grau mínimo $m_{\text{attach}}=114$, de modo que $k_{\min}=10$ está estritamente fora/abaixo do suporte do modelo, invalidando a estimativa de máxima verossimilhança.
- **Novos valores:**
  - ER: `"N/A (ajuste inválido: kmin=10 abaixo do suporte do modelo)"`
  - BA: `"N/A (ajuste inválido: kmin=10 abaixo do suporte do modelo, γ teórico = 3)"`

## 2. Correções Textuais e Conceituais em `results/parte3_resumo.md`
- **Tipografia:**
  - Corrigido `"anipicamente"` para `"atipicamente"` na discussão sobre o expoente de Clauset ($\gamma = 4.56$).
- **Cabeçalho de Tabela:**
  - `"Fração Isolados ($D$)"` alterado para `"Fração Isolados (% dos sobreviventes) ($D$)"` para evitar ambiguidade entre fração do total de nós inicial vs. nós ativos sobreviventes ao ataque.
- **Comparação de Modelos de Cauda (Limiar de Clauset vs. Regime Escolhido):**
  - Adicionado esclarecimento crucial: *"No limiar de Clauset (k_min = 4496, 328 nós) os três modelos ficam praticamente empatados (ΔAIC da log-normal = 0,11); a vantagem da log-normal vale para o regime k ≥ 1500, escolhido por nós."*
- **Atenuação Epistemológica (Processos Multiplicativos e Saturação Cognitiva):**
  - Substituição da afirmação categórica *"confirmando que a rede é governada por múltiplos processos multiplicativos e saturação cognitiva..."* por formulação de hipótese: *"o que sugere, como hipótese não testada, que a rede é governada por múltiplos processos multiplicativos e saturação cognitiva de conexões em vez de anexo preferencial irrestrito."*
