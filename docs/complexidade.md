# Complexidade assintótica (Big O)

Para cada algoritmo, este documento traz:
- complexidade de tempo e de memória;
- significado das variáveis;
- trecho do código que domina o custo;
- quando o método deixa de ser viável e qual alternativa usar.

As medições reais em entradas pequena, média e grande estão em [`desempenho.md`](desempenho.md). As linhas citadas referem-se à versão atual dos arquivos.

---

## Questão 1

### Variáveis

| Símbolo | Significado | Valor na instância G01 |
|---|---|---|
| n | SKUs no catálogo | 30 |
| k | SKUs no subconjunto do Brute Force | 6 |
| L_j | preços permitidos do SKU j (≤ 7 níveis do config) | 1 a 7 |
| Π L_j | combinações do Brute Force: produto dos L_j do subconjunto | 51.450 |
| C | categorias (média de canibalização) | 6 |
| B | linhas por bloco no BF vetorizado (`bloco=200_000`) | 200.000 |
| P | tamanho da população do GA | 40 (A) e 120 (B) |
| G | gerações executadas (até `geracoes_max` ou parada por paciência) | 92 a 134 |
| t | tamanho do torneio | 2 (A) e 4 (B) |
| K | opções de incentivo por SKU | 5 |
| S | passos de upgrade aceitos pelo Greedy (≤ n·K) | 1 (regra principal) a 30 |

### (A) Brute Force — `questao_01/brute_force.py`

- **Tempo: O(Π L_j · (k + C)).** Cada combinação é avaliada em O(k + C): demanda e caixa por SKU, mais a média por categoria.
- **Memória.**
  - Ingênuo: **O(k)**. O `itertools.product` gera uma combinação por vez.
  - Vetorizado: **O(min(Π L_j, B) · k)**. Cada bloco é uma matriz de até B combinações × k SKUs.
- **Trecho dominante.**
  - Ingênuo: o laço `for ms in itertools.product(*inst.permitidos)` (linha 15), que chama `avaliar_ingenuo` uma vez por combinação.
  - Vetorizado: `np.unravel_index` + `avaliar_vetorizado(inst, M)` (linhas 30–32). Dentro dela pesa a linha `dem = ... M ** (-e) ...` de `modelo.py`, aplicada a todo o bloco.
- **Quando deixa de ser viável.** O custo cresce exponencialmente com o número de SKUs. O catálogo completo tem Π L_j ≈ 1,5 × 10¹² combinações: cerca de 8,9 dias vetorizado ou 4,6 anos ingênuo, pelas taxas medidas. O limite prático fica entre 10⁶ e 10⁷ combinações.
  - O código escolhe automaticamente subconjuntos com até 200.000 combinações (`LIMITE_COMBINACOES_BF`).
  - Recusa subconjuntos acima de 1.000.000.
- **Alternativa.** O Algoritmo Genético resolve o catálogo; o BF fica como benchmark exato no subconjunto.

### (B) Algoritmo Genético — `questao_01/genetic_algorithm.py`

**Tempo: O(P·n + G · P · (t + n + C)).**

| Componente | Custo por geração |
|---|---|
| Seleção por torneio | O(P·t) |
| Crossover uniforme e mutação | O(P·n) |
| Avaliação da população | O(P·(n + C)) |
| Ordenação para o elitismo | O(P log P) |

- **Memória: O(P · n + G).** População e filhos (matrizes P × n), mais o histórico do melhor e da média por geração.
- **Trecho dominante.** `fit = avaliar_vetorizado(inst, decod(pop))` (linha 90), uma vez por geração dentro de `for g in range(cfg.geracoes_max)` (linha 69).
- **Quando deixa de ser viável.** O custo não é o problema: com n = 10⁴ e P = 120, a população ocupa cerca de 10 MiB. O limite é de **qualidade**: G cresce com n e o GA não garante o ótimo. As alternativas são:
  - decompor por categoria, já que a canibalização só acopla SKUs da mesma categoria;
  - aplicar busca local SKU a SKU a partir da solução do GA.

### (C) Greedy de incentivos — `questao_01/greedy.py`

- **Tempo.**
  - Gerar as opções (`opcoes_incentivo`): O(n·K).
  - Greedy com heap: **O(n·K + S·(K + log(n·K)))**. No pior caso, com S = n·K, dá O(n·K² + n·K·log(n·K)).
  - Ingênuo (`greedy_ingenuo`): **O(S · n · K)**, porque reescaneia todos os pares (SKU, opção) a cada passo. No pior caso, O(n²·K²).
- **Memória: O(n·K)** para as opções, o heap e a alocação por SKU.
- **Trecho dominante.**
  - Com heap: `while heap:` (linha 99), com `empilhar(j)` e `heapq.heappush` (linhas 80–93).
  - Ingênuo: o duplo `for` dentro de `while True:` (linha 139).
- **Quando deixa de ser viável.** Em custo, nunca para catálogos realistas. O limite é de **qualidade**: é uma heurística para a mochila de múltipla escolha. A alternativa exata é uma DP de mochila O(n·K·B/Δ), com orçamento discretizado em passos Δ, ou programação inteira.

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
| D | suporte da PMF da demanda semanal (≤ S; na prática, o pico de demanda observado) | até 358 |
| R | recursos promocionais | 3 |
| \|ρ\| | valores de ρ na calibração | 4 |

### (A) Monte Carlo — `questao_02/monte_carlo.py` e `modelo.gerar_cenarios`

- **Tempo.**
  - Construção: **O(N_c · T · J · M)**. Sorteio Poisson por nível (linha 28) e PMFs com `np.bincount` (linha 39).
  - Calibração de ρ: O(|ρ| · N_c · T · J).
  - Validação: **O(N_v · T · J)** por política. O laço `for t in range(T)` (linha 104) é vetorizado em cenários × SKUs (linha 114).
- **Memória.**
  - Construção: **O(N_c · T · J · M)**, porque as amostras ficam guardadas para o Greedy. Somam-se O(J · T · M · D) das PMFs.
  - Validação: **O(N_v · T · J)** de histórico.
- **Trecho dominante.** `dem[..., k] = rng.poisson(lam[..., k])` na construção e `q = np.minimum(s, rng.poisson(lam))` na validação.
- **Quando deixa de ser viável.** A memória das amostras passa de 1 GiB por volta de J ≈ 600 SKUs (cerca de 1,6 MiB por SKU com N_c = 2.000). A alternativa é gerar as PMFs SKU a SKU sem guardar as amostras e usar números aleatórios comuns para reduzir o N necessário.

### (B) Programação Dinâmica — `questao_02/dynamic_programming.py`

- **Tempo** (Bottom-Up, `resolver_sku`): **O(J · T · M · S · D)**.
- **Memória: O(J · T · S)** para V e π, mais **O(S · D)** de matrizes temporárias por SKU.
- **Trecho dominante.** `valores[k] = (pmf[t, k][None, :] * (r[t, k] * q + Vnext)).sum(axis=1)` (linha 51). O reuso de subproblemas está em `Vnext = V[t + 1][prox]` (linha 48).
- **Top-Down.** Com memo, O(T · S · M · D), igual ao Bottom-Up. Sem memo, **O((M · D)^T)**: exponencial, já inviável com T = 4.
- **Quando deixa de ser viável.**
  - A DP **conjunta** (t, s₁..s_J, caixa) teria cerca de 10⁵⁰ estados. Por isso decompomos por SKU, com o peso ρ e o MC medindo o risco conjunto.
  - A DP **por SKU** é pseudo-polinomial em S. Com estoques de 10⁵ unidades, a alternativa é agrupar o estoque em lotes.

### (C) Greedy semanal — `questao_02/greedy.py`

- **Tempo: O(T · J · R · N_c + T · J·R · log(J·R)).** Cada candidato estima o ganho com as amostras do MC (linha 64); a ordenação por razão fica na linha 75.
- **Memória: O(J · R + N_c).**
- **Trecho dominante.** `q1 = np.minimum(s, d0 + rng.poisson(lam * (fator - 1)))`.
- **Quando deixa de ser viável.** Sempre viável; o custo é dominado por N_c. Para N_c muito grande, calcular as esperanças pela PMF em vez das amostras.
