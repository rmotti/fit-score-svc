# FitScoreMl.ipynb — Estrutura do Notebook

Dataset: `davidcariboo/player-scores` (Kagglehub, ~211 MB)
Resultado: 62.383 transferências limpas, 4.010 perfis de clube, AUC-ROC 0.863

---

## Seções e o que cada uma produz

| # | Título | Variáveis/Artefatos produzidos |
|---|--------|-------------------------------|
| 1 | Importação e Carregamento dos Dados | `transfers`, `players`, `clubs` |
| 2 | Engenharia de Features e Limpeza Inicial | `df` (157k linhas, com `age_at_transfer`, `origin_league`) |
| 3 | Exploração de Dados Adicionais e Cobertura | análise exploratória, sem novas variáveis |
| 4 | Mapeamento Refinado de Ligas Domésticas | `competitions`, `club_games` |
| 5 | Consolidação do Mapeamento de Ligas | `games`, `club_league_expanded` (club_id → liga doméstica mais frequente) |
| 6 | Limpeza e Preparação do Dataset Final | **`df_clean`** (62.395 linhas, origin_league + age_at_transfer preenchidos) |
| 7 | Análise Exploratória de Features Principais | `fee_type` adicionado ao `df_clean` |
| 8 | Refinamento de Features e Correlações | `fee_proxy`, `log_fee_proxy`, `position_group` mapeado de sub_position |
| 9 | Agrupamento de Posições e Nacionalidades | `position_group` (13 grupos), `nationality` (69 países + Other) |
| 10 | Preparação de Features para o Modelo | **`df_model`** (7 colunas: to_club_name, position_group + 5 features encodadas/normalizadas), `age_norm`, `fee_norm` |
| 11 | Cálculo de Distância Gower | `df_features` (5 features para Gower), `dist_matrix` (teste 500×500) |
| 12 | Função de Pontuação de Fit (Inicial) | `compute_fit_score()` v1 (sem objetivo) |
| 13 | Adição de Filtros por Objetivo | `OBJECTIVE_FILTERS` (rebuild/youth/title/balanced), `compute_fit_score()` v2 |
| 14 | Pré-computação de Perfis de Clubes e Índices de Posição | **`club_profiles.pkl`** (4.010 perfis buyer), **`position_index.pkl`** |
| **14b** | **[NOVO] Pré-computação de Perfis de Venda** | **`seller_profiles.pkl`** — mesmo processo, chave = (from_club_name, position_group) |
| 15 | Função de Pontuação de Fit Otimizada | `compute_fit_score()` v3 (usa pkl, sample_size=2000) |
| 16 | Teste com Múltiplos Candidatos e Clubes | tabela de scores por candidato × clube × objetivo |
| 17 | Salvando Artefatos Finais e Metadados | **`model_metadata.json`** |
| 18 | Casos de Uso e Validação do Modelo | — |
| 18.1 | CASO 1 — Validador de Rumor: AUC-ROC + Confusion Matrix | AUC = 0.863 |
| 18.1b | Otimização do Threshold | threshold ótimo = 0.29, F1 = 0.806 |
| 18.2 | CASO 2 — Query Reversa: Precision@K + MRR | P@1=0.08, P@10=0.32, MRR=0.154 |
| 18.4 | CASO 4 — Fee Regression R² | R²=0.281, MAE ~2.92× em escala log |
| **18.5** | **[NOVO] CASO 5 — Sucesso de Transferência** | carrega `appearances.csv`; cruza aparições pós-transfer_date com fit score; Spearman correlation |

---

## Features do modelo

5 features de entrada para Gower distance:

| Feature | Tipo | Encoding |
|---------|------|----------|
| nationality | categórica | raw (Gower trata nativamente) |
| origin_league | categórica | raw |
| age_norm | numérica | MinMaxScaler [0,1] |
| fee_norm | numérica | MinMaxScaler [0,1] |
| fee_type | categórica | raw (paid / free / undisclosed) |

---

## Artefatos salvos

| Arquivo | Conteúdo | Produzido em |
|---------|----------|--------------|
| `club_profiles.pkl` | dict: (to_club_name, position_group) → DataFrame de histórico de compras | Seção 14 |
| `seller_profiles.pkl` | dict: (from_club_name, position_group) → DataFrame de histórico de vendas | Seção 14b (novo) |
| `position_index.pkl` | dict: position_group → todos os jogadores naquela posição | Seção 14 |
| `model_metadata.json` | total_transfers, clubs, profiles, position_groups, origin_leagues, nationalities, objectives, date_range | Seção 17 |

---

## Objetivos disponíveis

| Objetivo | Filtro aplicado |
|----------|----------------|
| balanced | sem filtro |
| rebuild | age_norm ≤ 0.35 (~até 26 anos) |
| youth | age_norm ≤ 0.25 (~até 23 anos) |
| title | fee_norm ≥ 0.50 (contratações de alto valor) |

---

## O que precisa rodar para as novas seções?

- **14b** (seller profile): requer `df_clean` + `df_features` em memória → rodar seções 1–11 antes
- **18.5** (success validation): requer `appearances.csv` (carregado na própria seção) + `df_clean` + `compute_fit_score` → rodar seções 1–15 antes
- Se a sessão Colab estiver ativa e com tudo em memória: basta inserir e rodar só as novas células
