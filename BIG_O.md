# Big O e desempenho — Questões 1 e 2

As medições abaixo foram feitas com `time.perf_counter()` (tempo) e `tracemalloc` (pico de memória alocada pelo Python/numpy). Os números vêm das execuções oficiais:

- Questão 1: `questao_01/saida/06_escalabilidade.csv`, `01_*.csv`, `02_*.csv` e `05_*.csv`.
- Questão 2: `questao_02/saida/10_escalabilidade.csv` e `resultado.json`.

Ambiente: Python 3.13, numpy 2.5, pandas 3.0, Linux x86-64. Os valores mudam de máquina para máquina; o que deve se manter é a **forma de crescimento**. Para medir na máquina do grupo, rode `python questao_0X/main.py --modo completo`.

---

## Questão 1

### Variáveis

| Símbolo | Significado | Valor na instância G01 |
|---|---|---|
| n | SKUs no catálogo | 30 |
| k | SKUs no subconjunto do Brute Force | 6 |
| L_j | preços permitidos do SKU j (≤ 7 níveis do config) | 1 a 7 |
| Π L_j | combinações do Brute Force = produto dos L_j do subconjunto | 51.450 |
| C | categorias (para a média de canibalização) | 6 |
| B | linhas por bloco no BF vetorizado (`bloco=200_000`) | 200.000 |
| P | tamanho da população do GA | 40 (A) e 120 (B) |
| G | gerações executadas (até `geracoes_max` ou parada por paciência) | 92 a 134 |
| t | tamanho do torneio | 2 (A) e 4 (B) |
| K | opções de incentivo por SKU | 5 |
| S | passos de upgrade aceitos pelo Greedy (≤ n·K) | 1 (regra principal) a 30 |

### (A) Brute Force — `questao_01/brute_force.py`

- **Tempo: O(Π L_j · (k + C)).** Cada combinação é avaliada por `avaliar_vetorizado` em O(k + C): demanda e caixa por SKU, mais a média por categoria.
- **Memória.**
  - Versão ingênua: **O(k)**. O `itertools.product` gera uma combinação por vez.
  - Versão vetorizada: **O(min(Π L_j, B) · k)**. Cada bloco materializa uma matriz de até B combinações × k SKUs.
- **Trecho dominante.**
  - Ingênuo: o laço `for ms in itertools.product(*inst.permitidos)` (linha 15) chama `avaliar_ingenuo` uma vez por combinação.
  - Vetorizado: `np.unravel_index` + `avaliar_vetorizado(inst, M)` (linhas 30–32). Dentro dela, a linha `dem = ... M ** (-e) ...` em `modelo.py` (linha 220) é aplicada a todas as combinações do bloco.

**Medição** (subconjuntos de 4, 5 e 6 SKUs):

| Entrada | Π L_j | Vetorizado: tempo | Vetorizado: memória | Ingênuo: tempo | Ingênuo: memória |
|---|---|---|---|---|---|
| pequena (k=4) | 1.470 | 0,0012 s | 0,5 MiB | 0,097 s | 14 KiB |
| média (k=5) | 10.290 | 0,0042 s | 3,9 MiB | 0,79 s | 14 KiB |
| grande (k=6) | 51.450 | 0,026 s | 23,2 MiB | 4,84 s | 14 KiB |

O tempo e a memória da versão vetorizada crescem proporcionalmente a Π L_j, porque nenhum dos três casos chegou ao bloco B. A versão ingênua tem memória constante, mas é cerca de 120× mais lenta.

**Quando deixa de ser viável.** O catálogo completo tem Π L_j ≈ **1,5 × 10¹²** combinações. Extrapolando as taxas medidas, isso levaria cerca de **8,9 dias** na versão vetorizada e cerca de **4,6 anos** na ingênua. Na prática, o limite fica em torno de 10⁶ a 10⁷ combinações, ou seja, segundos a poucos minutos. O código recusa subconjuntos acima de 1.000.000 de combinações e escolhe automaticamente subconjuntos com até 200.000 (`LIMITE_COMBINACOES_BF`).

**Alternativa.** O **Algoritmo Genético** resolve o catálogo inteiro. O BF fica restrito ao subconjunto pequeno, onde serve de benchmark exato.

### (B) Algoritmo Genético — `questao_01/genetic_algorithm.py`

**Tempo: O(P·n + G · P · (t + n + C)).**

| Componente | Custo por geração |
|---|---|
| Seleção por torneio | O(P·t) |
| Crossover uniforme e mutação | O(P·n) |
| Avaliação da população | O(P·(n + C)) |
| Ordenação para o elitismo | O(P log P), desprezível perto de P·n |

**Memória: O(P · n + G).** Guarda a população e os filhos (matrizes P × n) e o histórico do melhor e da média por geração (G valores).

**Trecho dominante.** `fit = avaliar_vetorizado(inst, decod(pop))` (linha 90), executada uma vez por geração dentro do laço `for g in range(cfg.geracoes_max)` (linha 69).

**Medição** (configuração A, P = 40, seed 1028):

| Entrada | n | Gerações até parar | Avaliações | Tempo | ms por geração | Memória (pico) |
|---|---|---|---|---|---|---|
| pequena (1/3 do catálogo) | 10 | 91 | 3.589 | 0,092 s | 1,01 | 141 KiB |
| média (catálogo) | 30 | 128 | 5.032 | 0,138 s | 1,08 | 234 KiB |
| grande (catálogo ×3) | 90 | 210 | 8.230 | 0,241 s | 1,15 | 541 KiB |

- A **memória cresce linearmente em n**, como prevê O(P·n).
- O **custo por geração quase não cresce**, porque a avaliação é vetorizada e, nessa faixa, domina o overhead fixo do numpy.
- O tempo total cresce porque **G cresce com n**: um espaço maior exige mais gerações para estagnar.

**Comparação de configurações** (catálogo, 5 seeds):

| Config | Gerações (média) | Avaliações | Tempo médio | Memória |
|---|---|---|---|---|
| A (P = 40) | 134 | 5.282 | 0,15 s | 242 KiB |
| B (P = 120) | 92 | 11.023 | 0,11 s | 498 KiB |

Com P três vezes maior, a memória dobra, mas B para antes. As duas chegam ao mesmo fitness.

**Quando deixa de ser viável.** O GA escala bem em tempo e memória: com n = 10⁴ e P = 120, a matriz da população tem cerca de 10 MiB. O risco está na **qualidade**, não no custo: o número de gerações cresce com n, e o GA não dá garantia de ótimo. Com muitos SKUs, as alternativas são:
- **decompor por categoria**, porque a canibalização só acopla SKUs da mesma categoria, e rodar um GA (ou até BF) por categoria;
- usar **busca local** (subida coordenada SKU a SKU) a partir da solução do GA.

### (C) Greedy de incentivos — `questao_01/greedy.py`

- **Tempo.**
  - Gerar as opções (`opcoes_incentivo`): O(n·K).
  - Greedy com heap: **O(n·K + S·(K + log(n·K)))**. A cada upgrade aceito, `empilhar` recalcula as opções do SKU em O(K) e faz um `heappush` em O(log(n·K)). No pior caso S = n·K, o que dá O(n·K² + n·K·log(n·K)).
  - Versão ingênua (`greedy_ingenuo`): **O(S · n · K)**, porque reescaneia todos os pares (SKU, opção) a cada passo. No pior caso dá O(n²·K²).
- **Memória: O(n·K)** para as opções, o heap e a alocação por SKU.
- **Trecho dominante.** O laço `while heap:` (linha 99) com `empilhar(j)` e `heapq.heappush` (linhas 80–93). Na ingênua, o duplo `for` dentro de `while True:` (linha 139).

**Medição** (regra "verba carimbada", que aloca mais passos):

| Entrada | n | Pares (SKU, opção) | Heap: tempo | Heap: memória | Ingênuo: tempo |
|---|---|---|---|---|---|
| pequena | 10 | 28 | 0,012 s | 47 KiB | 0,008 s |
| média | 30 | 95 | 0,039 s | 92 KiB | 0,027 s |
| grande | 90 | 288 | 0,120 s | 219 KiB | 0,085 s |

Nessas entradas as duas versões crescem de forma linear. O motivo é que o número de passos S é limitado pelo **orçamento** (S ≤ B / custo mínimo), e não por n·K. Por isso o termo quadrático da versão ingênua ainda não aparece, e o overhead do heap em Python a deixa um pouco mais lenta. O heap passa a compensar quando S cresce junto com n, ou seja, com orçamentos grandes e catálogos de milhares de SKUs.

**Quando deixa de ser viável.** Em custo computacional, nunca, para catálogos realistas. O limite é de **qualidade**: o Greedy é uma heurística para a mochila de múltipla escolha e pode deixar orçamento ocioso por causa de custos indivisíveis. Se o orçamento ficar apertado e isso importar, a alternativa exata é uma **DP de mochila** O(n·K·B/Δ), com orçamento discretizado em passos Δ, ou programação inteira.

### Otimização de código demonstrada (antes/depois)

| Trecho | Antes | Depois | Speedup |
|---|---|---|---|
| BF, 51.450 combinações | 4,63 s, 35 KiB | 0,038 s, 23 MiB | cerca de 122× (troca tempo por memória, limitada pelo bloco B) |
| Fitness de uma geração do GA (120 × 30) | 4,47 ms | 0,30 ms | cerca de 15× |

---

## Questão 2

### Variáveis

| Símbolo | Significado | Valor na instância G01 |
|---|---|---|
| J | SKUs | 24 |
| T | semanas | 12 |
| M | níveis de preço | 6 |
| N_c | cenários de construção | 2.000 |
| N_v | cenários de validação | 5.000 |
| S | estoque inicial máximo de um SKU (estados da DP por semana = S + 1) | 520 |
| D | tamanho do suporte da PMF da demanda semanal (≤ S; na prática, pico de demanda observado) | até 358 (Q2-003, o mais vendido) |
| R | recursos promocionais | 3 |
| \|ρ\| | valores de ρ na calibração | 4 |

### (A) Monte Carlo — `questao_02/monte_carlo.py` e `modelo.gerar_cenarios`

- **Tempo.**
  - Construção: **O(N_c · T · J · M)**. Para cada nível de preço, `rng.poisson` sorteia a demanda de N_c × T × J (linha 28). Depois, os `np.bincount` montam as PMFs (linha 39), também em O(N_c·T·J·M).
  - Calibração de ρ: |ρ| simulações nos cenários de construção, O(|ρ| · N_c · T · J).
  - Validação: **O(N_v · T · J)** por política simulada. O laço `for t in range(T)` (linha 104) processa os N_v cenários × J SKUs de uma vez (linha 114).
- **Memória.**
  - Construção: **O(N_c · T · J · M)**, porque as amostras `lam` e `dem` ficam guardadas para o Greedy. Somam-se O(J · T · M · D) das PMFs.
  - Validação: **O(N_v · T · J)** do histórico de estoque e preço, usado nos gráficos de trajetória.
- **Trecho dominante.** `dem[..., k] = rng.poisson(lam[..., k])` na construção e `q = np.minimum(s, rng.poisson(lam))` na validação.

### (B) Programação Dinâmica — `questao_02/dynamic_programming.py`

- **Tempo** (Bottom-Up, `resolver_sku`): **O(J · T · M · S · D)**. Para cada SKU, semana e nível, a esperança da equação de Bellman é calculada para todos os s em uma matriz (S+1) × (D+1) (linha 51).
- **Memória: O(J · T · S)** para a tabela V e a política π, mais **O(S · D)** de matrizes temporárias por SKU (`q`, `prox` e `Vnext`, linha 48).
- **Trecho dominante.** `valores[k] = (pmf[t, k][None, :] * (r[t, k] * q + Vnext)).sum(axis=1)` (linha 51). Ali ocorre o reuso: `Vnext = V[t + 1][prox]` lê subproblemas já resolvidos.
- **Comparação com o Top-Down.** Com memo, é O(T · S · M · D), igual ao Bottom-Up (610 subproblemas e 66.088 acertos de cache no Q2-001). Sem memo, é **O((M · D)^T)**: 1.455.780 chamadas com apenas 3 semanas, contra 9.916 com memo.

### (C) Greedy semanal — `questao_02/greedy.py`

- **Tempo: O(T · J · R · N_c + T · J·R · log(J·R)).** Cada candidato (SKU, recurso) estima o ganho com as N_c amostras do MC (linha 64). A ordenação por razão (linha 75) custa O(J·R · log(J·R)) por semana.
- **Memória: O(J · R + N_c)**: a lista de candidatos da semana e um vetor de amostras por vez.
- **Trecho dominante.** `q1 = np.minimum(s, d0 + rng.poisson(lam * (fator - 1)))`, dentro de `for nome, rec in RECURSOS.items()`.

### Medição em três tamanhos de entrada

| Entrada | J | Estados da DP | MC construção | DP | Greedy | MC validação |
|---|---|---|---|---|---|---|
| pequena (8 SKUs) | 8 | 18.444 | 0,20 s · 13 MiB | 0,41 s · 6,0 MiB | 0,13 s · 0,15 MiB | 0,09 s · 24 MiB |
| média (catálogo) | 24 | 61.560 | 0,60 s · 40 MiB | 1,39 s · 6,3 MiB | 0,38 s · 0,28 MiB | 0,26 s · 68 MiB |
| grande (×3) | 72 | 184.680 | 1,85 s · 119 MiB | 2,98 s · 8,1 MiB | 1,24 s · 0,66 MiB | 0,80 s · 200 MiB |

**DP isolada variando o estoque** (24 SKUs):

| Estoque | Estados | Tempo | Memória |
|---|---|---|---|
| ×1 | 61.560 | 1,07 s | 6,2 MiB |
| ×2 | 122.832 | 1,23 s | 12,3 MiB |
| ×4 | 245.376 | 1,63 s | 24,4 MiB |

**Leitura.**
- **MC e Greedy** crescem linearmente em J, como previsto.
- **DP em função de J:** cresce linearmente, porque os SKUs são independentes, e a memória quase não muda, porque a matriz temporária é por SKU.
- **DP em função do estoque:** a memória dobra quando S dobra, como prevê O(J·T·S). O tempo cresce menos que S·D porque o suporte D da demanda não depende do estoque.

### Quando cada método deixa de ser viável e qual alternativa usamos

| Método | Limite | Alternativa |
|---|---|---|
| **MC** | A memória das amostras guardadas, O(N_c·T·J·M), passa de 1 GiB por volta de J ≈ 600 SKUs (cerca de 1,6 MiB por SKU com N_c = 2.000) | Gerar as PMFs SKU a SKU sem guardar as amostras (o Greedy reamostra) e usar números aleatórios comuns para reduzir variância e, com isso, o N necessário |
| **DP conjunta** | Estado (t, s₁..s_J, caixa) teria Π S_j ≈ 10⁵⁰ estados: inviável desde o início | Decomposição por SKU, com ρ como peso do caixa no tempo e o MC medindo o risco conjunto (já implementado) |
| **DP por SKU** | Pseudo-polinomial em S. Com estoques de 10⁵ unidades e D de centenas, seriam cerca de 10⁹ operações por SKU | Discretizar o estoque em lotes (s em blocos de 10 ou 100 unidades), reduzindo S e D na mesma proporção |
| **Top-Down sem memo** | Já é inviável com T = 4 | Memoization ou Bottom-Up |
| **Greedy** | Sempre viável; o custo é dominado por N_c | Calcular as esperanças pela PMF em vez das amostras quando N_c for grande |
