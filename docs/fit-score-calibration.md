# Calibração do Fit Score — percentil contra o DNA do clube

> Status: **implementado** (2026-06-22). Escala do `fit_score` mudou de 0–1 para **0–100**.
> Contexto: discussão sobre o teto artificial do fit score.

## TL;DR

O fit score atual é uma distância média absoluta cujo **teto é estrutural em ~0.5–0.68**,
não 1.0 — nem os jogadores que o próprio clube já contratou chegam perto de 1.0 quando
avaliados contra o próprio perfil. Isso comprime todos os candidatos numa faixa estreita e
o número não discrimina bons de medianos. A mudança recalibra o score para o **percentil do
candidato contra a dispersão interna do perfil**: passa a significar *"encaixa melhor que X%
das contratações reais do clube"*, usa a faixa 0–100 inteira e o topo são os arquétipos certos.

---

## Antes

Em [`app/scoring.py`](../app/scoring.py), o fit é a distância Gower média do candidato contra
**todos** os jogadores do perfil do clube:

```python
distances = dist_matrix[0, 1:]
fit_score = 1.0 - np.mean(distances)   # [0, 1]
```

### Por que o teto é estrutural

A Gower média de 5 features (peso 0.2 cada). Para uma feature **categórica**, a distância
média de um valor fixo `v` contra a nuvem = fração de jogadores do perfil que *não* são `v`
= `1 - p_v`. As features `nationality` e `origin_league` têm cardinalidade altíssima, então
mesmo o valor modal cobre uma fração pequena do perfil → distância ≈ 0.8 em cada uma,
**independente de quão bom seja o candidato**. O melhor candidato matematicamente possível:

```
mean_dist ≈ 0.2·(0.8 + 0.8 + 0.2 + 0.2 + 0.5) ≈ 0.50   →   fit_max ≈ 0.50
```

Comparar um ponto contra a *média de uma nuvem dispersa* nunca dá distância perto de 0.

### Medições reais (dados de prod, `club_profiles.pkl`)

Membros **reais** do clube avaliados contra o próprio perfil (leave-one-out) — esse é o teto
que um jogador que o clube de fato contratou consegue atingir:

| Clube / posição | melhor membro real | mediana dos membros |
|---|---|---|
| Inter — CF (65)    | **0.68** | 0.53 |
| Chelsea — CB (55)  | **0.59** | 0.51 |
| Genoa — CF (52)    | **0.67** | 0.52 |

Distribuição dos candidatos reais da posição — tudo comprimido, sem discriminação no topo:

| Clube | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| Inter CF    | 13.2 | 44.5 | 51.5 | 61.2 | 69.2 |
| Chelsea CB  | 15.5 | 44.5 | 50.5 | 58.0 | 60.0 |
| Genoa CF    | 16.2 | 41.6 | 47.4 | 58.0 | 68.7 |

(escala ×100). A massa de candidatos vive em 0.40–0.52: a diferença entre um encaixe
excelente e um medíocre são ~5 pontos.

---

## Depois

O score do candidato passa a ser o **percentil** dele contra a distribuição-baseline do perfil.

### Definição

1. **Baseline (por clube × posição × objetivo):** para cada membro real do perfil, calcular a
   distância Gower média dele aos *outros* membros (leave-one-out). Isso dá um vetor de N
   distâncias = "como um membro típico desse clube encaixa nesse clube".
2. **Score do candidato:** calcular a distância média do candidato ao perfil (igual a hoje) e
   devolver a fração de membros reais que encaixam **pior** (distância ≥) que ele, com
   interpolação linear na distribuição-baseline para evitar saturação no topo.

```
fit_relativo = % de contratações reais do clube que encaixam pior que o candidato
```

### O que isso passa a significar

- Um membro **mediano** do clube → ~50.
- Um candidato mais central que **qualquer** contratação real do clube → ~100.
- A escala 0–100 inteira é usada.

### Medições reais (mesmos clubes, opção A)

| Clube | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| Inter CF    | 0.0 | 20.0 | 41.5 | 69.2 | 100.0 |
| Chelsea CB  | 0.0 | 25.5 | 47.3 | 96.4 | 100.0 |
| Genoa CF    | 0.0 | 19.2 | 36.5 | 67.3 | 100.0 |

Sanidade — o topo são os arquétipos corretos (mesmos jogadores que o absoluto já apontava,
mas agora com número alto e separável):

```
Inter CF   → Pio Esposito, Camarda, Colombo, Raimondo  (atacantes italianos jovens)  REL 88–100
Chelsea CB → Levi Colwill + academia inglesa de zaga    (CBs ingleses jovens)         REL 100
Genoa CF   → Pinamonti, Esposito, Borrelli              (atacantes italianos)         REL 100
```

---

## Justificativa

1. **O número absoluto nunca foi interpretável.** "1.0" significaria ser idêntico a *todo* o
   histórico de contratações simultaneamente — impossível quando o histórico é diverso. O teto
   real depende da diversidade interna do clube, não da qualidade do candidato, então comparar
   scores entre clubes ou contra um limiar fixo era enganoso.
2. **A escala relativa é interpretável e acionável.** "Encaixa melhor que 90% das contratações
   reais do clube" é uma frase que o usuário entende e que responde a pergunta de produto.
3. **Mantém a semântica de centroide** (não recompensa nichos artificiais como um kNN faria) —
   só conserta a escala, ancorando-a na dispersão real de cada clube.
4. **Custo desprezível.** Computar os 7.441 baselines (4.010 perfis × 4 objetivos) leva ~2.8s e
   ocupa ~0.3 MB — feito uma vez no boot do serviço, sem novo artefato nem re-rodar o notebook.

## Onde foi implementado

Tudo dentro do `fit-score-svc` (sem re-rodar o notebook, sem novo artefato):

- [`app/scoring.py`](../app/scoring.py): helpers `_prepare_profile`, `compute_baseline` (leave-one-out),
  `calibrate` (0–100 via CDF interpolada) e `build_all_baselines`. `compute_fit_score` e
  `recommend_candidates` passam a receber `club_baselines` e devolver o score calibrado.
- [`app/profiles.py`](../app/profiles.py): `club_baselines` derivado no boot (`load_artifacts`), ~3s.
- [`app/main.py`](../app/main.py): fia `store.club_baselines` nas 3 rotas de score.
- [`app/schemas.py`](../app/schemas.py): `fit_score` documentado como 0–100.

## Impacto no consumidor (`career-hub-api`)

A escala 0→100 quebraria quem assumia 0–1. Ajustado junto:

- `scout-playbooks/scout-score.ts`: removido o `* 100` (o score já chega 0–100).
- `fc26-players.service.ts`: prefixo de cache bumpado para `fit-score:v2:` — invalida as
  entradas 0–1 antigas, que senão conviveriam com as 0–100 por até 6h (TTL.fitScore).
- **Ordem de deploy:** o API falha aberto (score `null` em erro) e o cache foi versionado, então
  o desalinhamento na janela de deploy é transitório e auto-corrige. O front (repo separado) que
  formata `fitScore` precisa parar de multiplicar por 100, se fizer.

## Features do fit — custo removido

O fit usa **3 features**: `nationality`, `origin_league`, `age_norm` (pesos iguais em
`FEATURE_WEIGHTS = [1, 1, 1]`). Responde "**é o tipo de jogador que o clube contrata**".

**Custo (`fee_norm` + `fee_type`) foi removido das features**, por dois motivos:

1. **Dado ~70% lixo.** No dataset, **69% de todas as transferências** vêm como `free`/`fee_norm=0`
   (fee ausente/não-divulgado registrado como grátis). Os maiores gastadores do mundo aparecem
   60–70% "de graça" (Chelsea 67%, Man City 63%, Juventus 71%, Man Utd 62%) — impossível na
   realidade. `fee_norm` e `fee_type` saem da mesma origem, então estavam contaminados juntos.
2. **Redundante.** Custo/orçamento já é tratado **fora** do fit, no componente `marketValue` do
   scout score (relativo ao budget do save, com dado confiável do FC26 — trabalho do B-003).

> Histórico: o custo já passou por anti-duplo-contagem (40% → 25% do peso) antes de ser removido
> de vez. As features de custo seguem nos perfis (`.pkl`) — o filtro de objetivo `title`
> (`fee_norm >= 0.50`) ainda as usa —, só não entram mais na distância Gower do fit.

## Ressalvas

- **Saturação no topo:** vários candidatos podem ser mais próximos que qualquer membro real e
  empatar em 100. Mitigado com interpolação linear na distribuição-baseline.
- **Perfis pequenos** (< 5) continuam retornando `fit_score: null` / `confidence` por
  `profile_size`, igual a hoje.
