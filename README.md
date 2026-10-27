# CP6 — Dynamic Programming · Grupo G01

Duas questões, cada uma com três algoritmos integrados:

| Questão | Fluxo | Relatório |
|---|---|---|
| **1 — Operação Caixa 90** | Brute Force (benchmark exato) → Algoritmo Genético (catálogo) → Greedy (orçamento promocional) | [`questao_01/RELATORIO.md`](questao_01/RELATORIO.md) |
| **2 — Plano de Sobrevivência 12 semanas** | Monte Carlo (estimação) → Programação Dinâmica (preço por semana e estoque) → Greedy (recursos promocionais) → Monte Carlo (validação) | [`questao_02/RELATORIO.md`](questao_02/RELATORIO.md) |

A complexidade e o desempenho de todos os algoritmos estão em [`BIG_O.md`](BIG_O.md).

## Ambiente

- **Python 3.10 ou superior.** Testado com 3.13.17. A versão usada em cada execução fica gravada no `manifesto_dados.json`.
- **Dependências:** numpy, pandas e matplotlib (ver `requirements.txt`).
- **Testes:** usam apenas `unittest`, da biblioteca padrão. O pytest também os executa, se estiver instalado.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Reproduzir os resultados principais

```bash
python reproduzir.py               # testes + Questão 1 + Questão 2 (modo completo, cerca de 1 min)
```

Ou, passo a passo:

```bash
python -m unittest discover -s tests -v                      # 49 testes, cerca de 20 s
python questao_01/main.py --seed 28 --modo completo           # cerca de 15 s
python questao_02/main.py --seed 28 --cenarios 5000           # cerca de 30 s
```

**Comportamento padrão da seed.** Sem `--seed`, cada questão usa `seed_base` e `seed_final` do config (28 e 1028). Com `--seed S`, usa `seed_base = S` e `seed_final = S + 1000`. A mesma seed gera exatamente os mesmos resultados, o que é verificado nos testes.

**Onde ficam os resultados.** Em `questao_0X/saida/`, com:
- tabelas CSV e figuras PNG;
- `resultado.json`;
- `manifesto_dados.json`;
- `log_execucao.txt`.

## Seeds e reprodutibilidade

Todo componente aleatório recebe uma seed explícita, derivada de `seed_base` (construção) ou `seed_final` (execução e validação). As seeds efetivas ficam listadas no manifesto.

| Componente | Seed |
|---|---|
| Q1: sorteio dos preços permitidos | `seed_base` |
| Q1: GA no subconjunto (cada execução) | `seed_base + i` |
| Q1: GA no catálogo (cada execução) | `seed_final + i` |
| Q1: população de teste da otimização de código | `seed_base` |
| Q2: cenários de construção | `seed_base` |
| Q2: Poisson da construção | `seed_base + 1` |
| Q2: Greedy | `seed_base + 2` |
| Q2: cenários de validação | `seed_final` |
| Q2: vendas na simulação | `seed_final + 7` |
| Q2: cenários de estresse | `seed_final + 100..102` |

## Manifesto de dados

O `data/manifesto_dados.json` é o **manifesto oficial** e não é alterado. A cada execução, cada questão gera o próprio `saida/manifesto_dados.json` com:
- seeds efetivas, quantidade e lista de SKUs, metas financeiras;
- **SHA-256** de cada arquivo de dados e do conjunto (`sha256_dataset`);
- conferência dos hashes com o manifesto oficial (`dados_conferem_com_manifesto_oficial`);
- parâmetros da execução, incluindo os de cenário-surpresa;
- comando usado e versões do Python e das bibliotecas.

## Cenário-surpresa (defesa técnica)

Todas as alterações previstas no enunciado são feitas **por parâmetro**, sem mudar o código:

| Alteração | Questão 1 | Questão 2 |
|---|---|---|
| Meta de caixa +10% | `--meta-fator 1.10` | `--meta-fator 1.10` (multiplica as dívidas) |
| Concorrente reduz o preço em 8% | — (não há concorrente na Q1) | `--concorrente-fator 0.92` |
| Custo logístico +15% | `--logistica-fator 1.15` | `--logistica-fator 1.15` |
| Orçamento promocional −25% | `--orcamento-fator 0.75` | `--orcamento-fator 0.75` |
| Um SKU deixa de estar disponível | `--remover-sku Q1-023` | `--remover-sku Q2-003` |
| Mutação ou população do GA | `--ga-mutacao 0.10 --ga-pop 80` (também `--ga-torneio`, `--ga-elitismo`, `--ga-geracoes`) | — |
| Mais incerteza / outro caixa inicial | — | `--incerteza-fator 2 --caixa-inicial 50000` |
| Outra instância | `--grupo G07`, ou `--config` e `--catalogo` | `--grupo G07`, ou `--config` e `--catalogo` |

Os fatores podem ser combinados. Os SKUs do subconjunto do Brute Force e os produtos representativos da Q2 são escolhidos por regra, não por nome; por isso remover qualquer SKU não quebra a execução.

```bash
python cenarios_surpresa.py        # roda todos os casos acima e gera cenarios_surpresa/resumo.md
python questao_01/main.py --help   # lista completa de opções
```

## Estrutura

```
CP6/
├── data/                       # dados oficiais (não alterar: hashes conferidos a cada execução)
├── comum.py                    # CLI comum, manifesto, SHA-256, versões
├── questao_01/
│   ├── modelo.py               # premissas, validação de entrada, cenário-surpresa, fitness
│   ├── brute_force.py          # (A)
│   ├── genetic_algorithm.py    # (B)
│   ├── greedy.py               # (C)
│   ├── escalabilidade.py       # medição pequena/média/grande
│   ├── main.py                 # integração + CLI
│   └── RELATORIO.md
├── questao_02/
│   ├── modelo.py               # premissas, validação, cenários de incerteza
│   ├── monte_carlo.py          # (A) construção e validação
│   ├── dynamic_programming.py  # (B)
│   ├── greedy.py               # (C)
│   ├── main.py                 # integração + CLI
│   └── RELATORIO.md
├── tests/                      # unittest: casos pequenos, limites, entradas inválidas, integração
├── BIG_O.md
├── cenarios_surpresa.py
├── reproduzir.py
└── requirements.txt
```

## Testes

`python -m unittest discover -s tests -v` executa 49 testes:

- **Casos pequenos calculados à mão.**
  - Margem de um SKU.
  - DP com demanda determinística.
  - Brute Force com 27 combinações.
  - Simulação sem vendas, com pagamento e com falha de dívida.
- **Equivalências.**
  - BF ingênuo igual ao vetorizado, inclusive com blocos pequenos.
  - DP Bottom-Up igual à Top-Down, com e sem memo.
  - Greedy com heap igual ao ingênuo.
  - GA chega ao ótimo do BF.
- **Limites.**
  - Orçamento zero.
  - Estoque zero.
  - Empate de preços.
  - Um único SKU.
  - Incerteza zero.
  - Vendas nunca acima do estoque.
- **Entradas inválidas.**
  - Coluna faltando, valor negativo, texto em coluna numérica, NaN, SKU duplicado.
  - Dívida fora do horizonte.
  - Fatores ≤ 0.
  - SKU inexistente.
  - Parâmetros do GA e da linha de comando inválidos.
- **Integração.**
  - Mesma seed gera o mesmo resultado.
  - Manifesto gerado com hashes conferidos.
  - Cenário-surpresa pela linha de comando.
  - Arquivo de dados alterado é detectado.