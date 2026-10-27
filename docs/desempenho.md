# Desempenho medido

Todas as medições usam `time.perf_counter()` (tempo) e `tracemalloc` (pico de memória alocada) por meio da função `medir()` de cada `modelo.py`. Ambiente das medições abaixo: Python 3.13, numpy 2.5, pandas 3.0, Linux x86-64.

Os números mudam de máquina para máquina; o que deve se manter é a **forma de crescimento** prevista em [`complexidade.md`](complexidade.md).

**Como reproduzir:**
- Q1: `python questao_01/main.py --modo completo` → `questao_01/saida/06_escalabilidade.csv`, `fig4_escalabilidade.png`.
- Q2: `python questao_02/main.py --modo completo` → `questao_02/saida/10_escalabilidade.csv`, `fig6_escalabilidade.png`.

---

## Questão 1 — entradas pequena, média e grande

**Brute Force** (subconjuntos de 4, 5 e 6 SKUs):

| Entrada | Π L_j | Vetorizado: tempo | Vetorizado: memória | Ingênuo: tempo | Ingênuo: memória |
|---|---|---|---|---|---|
| pequena (k=4) | 1.470 | 0,0012 s | 0,5 MiB | 0,097 s | 14 KiB |
| média (k=5) | 10.290 | 0,0042 s | 3,9 MiB | 0,79 s | 14 KiB |
| grande (k=6) | 51.450 | 0,026 s | 23,2 MiB | 4,84 s | 14 KiB |
| catálogo inteiro (extrapolado) | 1,5 × 10¹² | cerca de 8,9 dias | — | cerca de 4,6 anos | — |

Tempo e memória da versão vetorizada crescem proporcionalmente a Π L_j, porque o bloco B ainda não foi atingido.

**Algoritmo Genético** (configuração A, P = 40, seed 1028):

| Entrada | n | Gerações | Avaliações | Tempo | ms por geração | Memória |
|---|---|---|---|---|---|---|
| pequena (1/3 do catálogo) | 10 | 91 | 3.589 | 0,092 s | 1,01 | 141 KiB |
| média (catálogo) | 30 | 128 | 5.032 | 0,138 s | 1,08 | 234 KiB |
| grande (catálogo ×3) | 90 | 210 | 8.230 | 0,241 s | 1,15 | 541 KiB |

- A memória cresce linearmente em n, como prevê O(P·n).
- O custo por geração quase não muda, porque a vetorização faz o overhead fixo do numpy dominar.
- O tempo total cresce porque G cresce com n.

**Comparação das configurações do GA** (catálogo, 5 seeds):

| Config | Gerações | Avaliações | Tempo médio | Memória | Fitness |
|---|---|---|---|---|---|
| A (P = 40, torneio 2) | 134 | 5.282 | 0,15 s | 242 KiB | R$ 536.125,91 |
| B (P = 120, torneio 4) | 92 | 11.023 | 0,11 s | 498 KiB | R$ 536.125,91 |

**Greedy** (regra "verba carimbada", que aceita mais passos):

| Entrada | n | Pares (SKU, opção) | Heap: tempo | Heap: memória | Ingênuo: tempo |
|---|---|---|---|---|---|
| pequena | 10 | 28 | 0,012 s | 47 KiB | 0,008 s |
| média | 30 | 95 | 0,039 s | 92 KiB | 0,027 s |
| grande | 90 | 288 | 0,120 s | 219 KiB | 0,085 s |

Nessa faixa o número de passos S é limitado pelo orçamento, e não por n·K. Por isso as duas versões crescem de forma linear, e o overhead do heap em Python ainda pesa mais que o termo quadrático da versão ingênua.

**Otimização de código demonstrada (antes/depois):**

| Trecho | Antes | Depois | Speedup |
|---|---|---|---|
| BF, 51.450 combinações | `itertools.product` + Python puro: 4,63 s, 35 KiB | numpy em blocos: 0,038 s, 23 MiB | cerca de 122× (troca tempo por memória, limitada pelo bloco B) |
| Fitness de uma geração do GA (120 × 30) | laço por indivíduo: 4,47 ms | matriz numpy: 0,30 ms | cerca de 15× |

As duas versões são conferidas com `assert` no `main.py` e nos testes.

---

## Questão 2 — entradas pequena, média e grande

| Entrada | J | Estados da DP | MC construção | DP | Greedy | MC validação |
|---|---|---|---|---|---|---|
| pequena (8 SKUs) | 8 | 18.444 | 0,20 s · 13 MiB | 0,41 s · 6,0 MiB | 0,13 s · 0,15 MiB | 0,09 s · 24 MiB |
| média (catálogo) | 24 | 61.560 | 0,60 s · 40 MiB | 1,39 s · 6,3 MiB | 0,38 s · 0,28 MiB | 0,26 s · 68 MiB |
| grande (×3) | 72 | 184.680 | 1,85 s · 119 MiB | 2,98 s · 8,1 MiB | 1,24 s · 0,66 MiB | 0,80 s · 200 MiB |

**DP isolada, variando o estoque** (24 SKUs):

| Estoque | Estados | Tempo | Memória |
|---|---|---|---|
| ×1 | 61.560 | 1,07 s | 6,2 MiB |
| ×2 | 122.832 | 1,23 s | 12,3 MiB |
| ×4 | 245.376 | 1,63 s | 24,4 MiB |

**Reuso de subproblemas na DP** (SKU Q2-001):

| Caso | Chamadas | Subproblemas | Acertos de cache | Tempo |
|---|---|---|---|---|
| Top-Down com memo, 12 semanas | 74.122 | 610 | 66.088 | 0,05 s |
| Top-Down com memo, 3 semanas | 9.916 | 88 | 3.332 | 0,006 s |
| Top-Down sem memo, 3 semanas | 1.455.780 | — | 0 | 0,73 s |

**Leitura.**
- MC e Greedy crescem linearmente em J.
- A DP cresce linearmente em J, com memória quase constante, porque as matrizes temporárias são por SKU.
- A memória da DP dobra quando o estoque dobra, como prevê O(J·T·S). O tempo cresce menos que S·D porque o suporte D da demanda não depende do estoque.
- O maior consumidor de memória é o MC, por guardar as amostras para o Greedy.
