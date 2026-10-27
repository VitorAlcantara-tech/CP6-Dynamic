# Registro de decisões

Cada linha registra uma decisão de modelagem ou de implementação: a alternativa considerada, o motivo da escolha, onde ela está no código e o parâmetro que permite mudá-la.

A coluna **Origem** segue o [`IA_USO.md`](../IA_USO.md): "IA" indica decisão proposta pela IA generativa; "Grupo" indica decisão do grupo. A coluna **Validação do grupo** ✏️ deve ser preenchida pelo grupo (concorda, alterou, por quê).

## Transversais (Q1 e Q2)

| # | Decisão | Alternativa considerada | Motivo | Onde | Origem | Validação do grupo ✏️ |
|---|---|---|---|---|---|---|
| T1 | **Caixa operacional = contribuição:** `vendas · [p(1−taxa) − frete − custo]` | Receita líquida (custo do estoque tratado como afundado) | Na escala da receita (cerca de R$ 2 milhões), metas e dívidas seriam triviais; a meta da Q1 fica próxima da contribuição a preço atual. Também evita que liquidar abaixo do custo pareça bom | `modelo.avaliar_vetorizado` (Q1), `monte_carlo.simular` (Q2) | IA | |
| T2 | Seeds derivadas: `seed_final = seed + 1000`, mais deslocamentos fixos por componente | Uma única seed global | Cada componente fica reprodutível e independente; a validação da Q2 nunca reaproveita cenários da construção | `comum.resolver_seeds`, manifesto | IA | |
| T3 | Cenário-surpresa por fatores de linha de comando | Editar config ou código | O enunciado exige absorver alterações sem reescrever o projeto | `comum.parser_base`, `modelo.aplicar_cenario` | IA | |
| T4 | Validação de entrada com `ValueError` e mensagem clara | Deixar o erro estourar dentro dos algoritmos | O enunciado pede testes de entradas inválidas; erros cedo são mais fáceis de explicar | `modelo.validar_entrada` | IA | |
| T5 | Testes com `unittest` | pytest | Não exige instalação; o pytest também executa | `tests/` | IA | |
| T6 | Divisão de cada questão em um módulo por algoritmo + `main.py` | Um único script | Deixa visível a independência dos algoritmos e onde está a integração | estrutura das pastas | **Grupo** (divisão) e IA (`modelo.py` comum) | |

## Questão 1

| # | Decisão | Alternativa considerada | Motivo | Onde | Origem | Validação do grupo ✏️ |
|---|---|---|---|---|---|---|
| Q1.1 | Preços permitidos: sorteio de 4 a 7 níveis por SKU com `seed_base`, sempre incluindo 1,00 | Todos os 7 níveis para todos os SKUs | O enunciado diz que o conjunto é "gerado a partir da seed do grupo" | `modelo.gerar_precos_permitidos` | IA | |
| Q1.2 | Piso de margem rígido: níveis abaixo do piso são removidos; SKU sem nível viável fica **fora da campanha** | Penalidade no fitness; manter o SKU com margem abaixo do piso | O enunciado manda respeitar o piso; uma restrição rígida é mais fácil de explicar | idem | IA | |
| Q1.3 | Desconto profundo (m < 0,90) só para estoque com 180 dias ou mais | Nenhuma regra de idade | Dá uso a `idade_estoque_dias` e evita desconto indiscriminado | `DESCONTO_PROFUNDO`, `IDADE_MIN_DESC_PROFUNDO` | IA | |
| Q1.4 | No máximo 30% dos SKUs com desconto (penalidade) | Sem limite | "Evitar política de descontos indiscriminada"; também acopla os SKUs | `MAX_FRAC_COM_DESCONTO`, `PENALIDADE` | IA | |
| Q1.5 | Canibalização dentro da categoria, γ = 0,5 | SKUs independentes | Sem acoplamento, o ótimo seria escolher o melhor preço SKU a SKU e o GA seria desnecessário | `GAMMA_CANIBALIZACAO` | IA | |
| Q1.6 | Custo de armazenagem do estoque que sobra: 4,5% em 90 dias, mais 4,5% para itens com 180 dias ou mais | Ignorar o estoque remanescente | Liga a idade do estoque ao caixa | `TAXA_ARMAZENAGEM_90D`, `SOBRETAXA_ENVELHECIDO` | IA | |
| Q1.7 | Subconjunto do BF escolhido por regra (2 por categoria, mais níveis, até 200.000 combinações) | Lista fixa de SKUs | Funciona após remover SKUs e em outras instâncias | `main.escolher_subconjunto`, `--subconjunto` | IA | |
| Q1.8 | GA: codificação por índice do preço permitido, torneio, crossover uniforme, mutação de cerca de 1/n por gene, elitismo, parada por paciência | Codificação binária, roleta, crossover de 1 ponto | Todo cromossomo respeita margem e estoque; o torneio tolera fitness penalizado; os SKUs não têm ordem natural | `genetic_algorithm.ga` | IA | |
| Q1.9 | Duas configurações: A (da CLI) e B (3× população, torneio +2, mutação ×1,5) | Grade grande de parâmetros | O enunciado pede ao menos duas; B mostra o efeito de maior pressão seletiva | `main.configs_ga`, `--ga-*` | IA | |
| Q1.10 | Greedy: mochila de múltipla escolha por upgrades com razão Δganho/Δcusto ≥ 1 | Gastar todo o orçamento | O caixa vai para a dívida: um incentivo que não se paga reduz o caixa. A sensibilidade "verba carimbada" mostra o custo de gastar tudo | `greedy.greedy_incentivos(min_razao)` | IA | |
| Q1.11 | Incentivos: cupom (100%/100%), cashback (resgate 65%, percepção 75%), frete subsidiado (percepção 130%) | Um único tipo de incentivo | O enunciado cita os três tipos; os parâmetros são premissas | `INCENTIVOS` | IA | |
| Q1.12 | BF vetorizado em blocos | Só a versão ingênua | Otimização demonstrada: cerca de 122× mais rápido, com memória limitada pelo bloco | `brute_force.brute_force_vetorizado` | IA | |

## Questão 2

| # | Decisão | Alternativa considerada | Motivo | Onde | Origem | Validação do grupo ✏️ |
|---|---|---|---|---|---|---|
| Q2.1 | Demanda Poisson com intensidade lognormal; metade do choque comum a todos os SKUs | Normal contínua; choques independentes | Vendas inteiras; a correlação de mercado impede que o risco "suma" no portfólio | `modelo.gerar_cenarios`, `RHO_DEMANDA_COMUM` | IA | |
| Q2.2 | Concorrente: passeio aleatório lognormal + efeito *buy box* (logística, K = 15, tolerância de 2%) | Elasticidade cruzada simples | Sem esse efeito o problema fica degenerado (preço máximo sempre ótimo e P(pagar) próxima de 100%) — isso foi testado; o *buy box* usa o `preco_concorrente_base` do catálogo | `modelo.fator_concorrencia`, `K_BUYBOX`, `DELTA_BUYBOX` | IA | |
| Q2.3 | Custo logístico com choque comum e persistente (AR(1), φ = 0,6) | Ruído independente por semana | Um reajuste de frete dura algumas semanas | `PHI_LOGISTICA` | IA | |
| Q2.4 | Gap de referência do concorrente (`gap_ref`) fixado nos dados originais | Recalcular o gap após o cenário-surpresa | Sem isso, "concorrente −8%" seria neutralizado pela normalização | `modelo.carregar` | IA | |
| Q2.5 | **DP por SKU**: estado (semana, estoque), PMF do MC, peso ρ no tempo | Estado conjunto (semana, estoques, caixa) | O estado conjunto teria cerca de 10⁵⁰ estados; o risco conjunto é medido pelo MC | `dynamic_programming.resolver_catalogo` | IA | |
| Q2.6 | V₁₃(s) = 0: estoque que sobra não gera nem consome caixa no horizonte | Valor residual do estoque | Coerente com T1 (o custo só é reposto quando há venda) | `resolver_sku` | IA | |
| Q2.7 | ρ escolhido nos cenários de **construção** pelo critério max P(pagar) e, no empate, max caixa | Fixar ρ = 0 | Liga o objetivo de probabilidade ao objetivo da DP sem usar os cenários de validação | `main.main`, `--grade-rho` | IA | |
| Q2.8 | Bottom-Up como solver; Top-Down com memo para provar equivalência e medir o reuso | Só um dos dois | O enunciado pede para mostrar onde ocorre a reutilização de subproblemas | `resolver_sku`, `resolver_top_down` | IA | |
| Q2.9 | Caminho ótimo reconstruído ao longo do estoque **esperado**; a validação usa a política π_t(s) em malha fechada | Só a trajetória fixa | A malha fechada falha muito menos (2 contra 61 falhas em 5.000) | `reconstruir`, `monte_carlo.simular` | IA | |
| Q2.10 | Greedy mede o ganho de cada promoção **descontando o valor de guardar o estoque** (V da DP) | Ganho bruto de vendas | Vender antes só vale se superar o valor da unidade mais tarde; é a integração DP → Greedy | `greedy.planejar` | IA | |
| Q2.11 | Recursos: destaque R$ 350 (+15%, no máximo 3 por semana), mídia R$ 1.200 (+35%), frete promocional (custo por unidade); 1 por SKU por semana; razão ≥ 1 | Outros custos e efeitos | Premissas; o enunciado não fornece valores | `RECURSOS` | IA | |
| Q2.12 | Dívida não paga = falha, com o saldo rolado (caixa negativo) | Encerrar o cenário na primeira falha | Permite medir a probabilidade de cada compromisso separadamente | `monte_carlo.simular` | IA | |
| Q2.13 | Percentil de risco P5 (VaR 95%) | P10 | Mais conservador que o limite de falha de 10% do config | `--percentil-risco` | IA | |
