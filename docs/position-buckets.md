# Buckets de posição — agregar GK/DEF/MID/ATT contra o dataset pequeno

> Status: **implementado** (2026-06-22). Paliativo até existir um dataset histórico maior.
> Contexto: o cohort por (clube × posição granular) é raso demais para calibrar bem.

## TL;DR

O histórico de transferências é fatiado por `(club_name, position_group)`, com 13 grupos
granulares (GK, CB, RB, LB, DM, CM, AM, RM, LM, CF, SS, RW, LW). Dividir cada clube em 13
posições deixa os cohorts minúsculos: **mediana ~7 transferências por perfil, 28% no piso de
5**, só 1,2% acima de 20. Com n tão baixo, cada transferência pesa demais e a calibração em
percentil fica grosseira. A solução atual agrega as 13 posições em **4 buckets amplos** —
**GK, DEF, MID, ATT** — adicionados **em memória no boot**, ao lado dos 13 grupos granulares
(que continuam existindo). Resultado: mediana ~9, perfis robustos (n≥10) sobem de 21% → 48%.
Sem regenerar o `.pkl`.

## O problema (números da v652, sem retornos de empréstimo)

| Métrica | Valor |
|---|---|
| Perfis granulares `(club, pos)` | 2785 |
| Mediana de transferências/perfil | 7 |
| p25 | 5 (o piso mínimo) |
| Perfis exatamente no piso (n=5) | 790 (28%) |
| Perfis robustos (n≥10) | 21% |
| Perfis com n>20 | 34 (1,2%) |

Exemplo Man Utd: CF=11, CB=10, GK=7, RB=7, CM=6, RW=6, AM=5, DM=5, LB=5 — quase tudo no fundo
do poço.

## O mapeamento

```
GK  → GK    (isolado: não há com o que agrupar — goleiro tem idade/liga/nacionalidade
             muito diferentes; juntar com a defesa poluiria o cohort)
CB, RB, LB             → DEF
DM, CM, AM, RM, LM     → MID
CF, SS, RW, LW         → ATT
```

Definido em [`app/profiles.py`](../app/profiles.py) (`POSITION_BUCKETS`). O cliente
(career-hub-api) tem o espelho desse mapa: posição FC26 (PT) → bucket, nos dois
`POSITION_GROUP` de `scouting.service.ts` e `fc26-players.service.ts`.

## Como funciona (sem regenerar o pkl)

No boot, `_add_position_buckets()` roda **depois** de carregar os `.pkl` e **antes** de
`build_all_baselines()`:

1. Concatena os DataFrames das posições granulares de cada bucket numa nova chave
   `(club, "MID")`, `(club, "DEF")`, `(club, "ATT")` em `club_profiles`.
2. Faz o mesmo no `position_index` (candidatos por posição).
3. Como os baselines de calibração são derivados de **tudo** que está em `club_profiles`,
   os baselines dos buckets saem de graça — nenhuma função de scoring precisa mudar.

As 13 chaves granulares são **preservadas** → totalmente backward-compatible. O schema
(`PositionGroup` em `app/schemas.py`) aceita os 13 + os 4 buckets.

### Efeito no `/health`

`profiles_loaded` sobe de 2785 → **4308** (granulares + buckets); `position_groups` de
13 → **16** (os 13 + DEF/MID/ATT; GK já contava).

## Ganho medido

| Métrica | 13 grupos | 4 buckets |
|---|---|---|
| Mediana transf./perfil | 7 | **9** |
| Perfis com n≥10 | 21% | **48%** |
| Man Utd MID | n=6, confidence `low` | **n=16, confidence `medium`** |
| Man Utd DEF | CB=10, RB=7, LB=5 | **DEF=22** |

## Trade-off

Perde-se granularidade tática: o "DNA de centroavante" se dilui no DNA de "atacante"
(CF+SS+RW+LW juntos). Para um clube que contrata pontas caros e CF baratos, o cohort vira a
média dos dois. Mas com n=5–6 a calibração já era grosseira — então na prática troca-se
*granularidade ilusória* por *robustez real*.

## Reverter

O serviço continua expondo os 13 grupos granulares. Para voltar à granularidade:

1. No cliente (career-hub-api): remapear os valores dos `POSITION_GROUP` de volta aos grupos
   granulares (`MC: 'CM'` etc.).
2. (Opcional) no serviço: remover a chamada `_add_position_buckets()` no boot e os buckets do
   `Literal PositionGroup`.

## Caminho definitivo

Buckets são um paliativo. A solução de fundo é **um dataset histórico maior** (mais temporadas
e/ou mais transferências por posição), que permitiria voltar à granularidade de 13 grupos com
cohorts robustos. Enquanto isso não existe, os buckets ficam.
