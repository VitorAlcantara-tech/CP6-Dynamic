"""
Modelo compartilhado da Questão 2 (Plano de Sobrevivência em 12 Semanas):
premissas, leitura dos dados, geração de cenários de incerteza e medição.
Usado por monte_carlo.py, programacao_dinamica.py, greedy.py e main.py.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent.parent      # raiz do projeto (comum.py)
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))
from comum import checar_fator, remover_skus  # noqa: E402

# ---------------------------------------------------------------------------
# PREMISSAS (explícitas e ajustáveis)
# ---------------------------------------------------------------------------
# Caixa operacional = contribuição: receita líquida de taxa e frete MENOS a
# reposição do custo do produto vendido (mesma definição da Questão 1).
RHO_DEMANDA_COMUM = 0.5      # correlação do choque de demanda entre SKUs (fator de mercado)
RHO_CONC_COMUM = 0.5         # parte do movimento do concorrente que é de mercado
PHI_LOGISTICA = 0.6          # persistência semanal (AR(1)) do choque de custo logístico
# Concorrência (efeito "buy box" do marketplace): a participação do SKU cai de forma
# logística quando o preço fica acima do concorrente. g = p / p_concorrente.
#   share(g) = 1 / (1 + exp(K·(g − 1 − DELTA)))   e o fator de demanda é
#   share(g) / share(g0), com g0 = preco_atual / preco_concorrente_base
#   (normalizado: no preço atual e concorrente parado o fator é 1 = demanda_base).
K_BUYBOX = 15.0              # sensibilidade ao gap de preço
DELTA_BUYBOX = 0.02          # tolerância: até 2% acima do concorrente quase não perde share

# Recursos promocionais escassos (etapa C). Um recurso por SKU por semana.
RECURSOS = {
    "destaque": dict(custo_fixo=350.0, uplift=0.15, limite_semana=3),   # vitrine/home
    "midia": dict(custo_fixo=1200.0, uplift=0.35, limite_semana=None),  # anúncio patrocinado
    # frete promocional: empresa paga o frete do cliente (= custo_logistico_base por
    # unidade vendida); o cliente percebe isso como corte de preço de mesmo valor.
    "frete_promocional": dict(custo_fixo=0.0, uplift=None, limite_semana=None),
}


COLUNAS_CATALOGO = ["sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial",
                    "demanda_base_semanal", "elasticidade", "taxa_marketplace",
                    "custo_logistico_base", "preco_concorrente_base"]
CHAVES_CONFIG = ["seed_base", "seed_final", "numero_semanas", "caixa_inicial", "dividas",
                 "variabilidade", "orcamento_promocional_semanal", "niveis_preco_relativos",
                 "limite_risco_falha"]


def caminhos(pasta_dados: Path, grupo: str = "G01"):
    pasta = Path(pasta_dados)
    return pasta / f"q2_config_{grupo}.json", pasta / f"q2_catalogo_{grupo}.csv"


def carregar(pasta_dados: Path = None, grupo: str = "G01", config_path=None, catalogo_path=None):
    """Lê config e catálogo, valida e guarda gap_ref = preco_atual / preco_concorrente_base
    ORIGINAL (a demanda_base foi observada com esse gap; se o concorrente mudar de preço
    num cenário-surpresa, a demanda reage em relação a esse ponto de referência)."""
    pc, pk = caminhos(pasta_dados, grupo) if pasta_dados is not None else (None, None)
    config_path, catalogo_path = Path(config_path or pc), Path(catalogo_path or pk)
    for p in (config_path, catalogo_path):
        if not p.exists():
            raise FileNotFoundError(f"arquivo não encontrado: {p}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    df = pd.read_csv(catalogo_path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    validar_entrada(cfg, df)
    df["gap_ref"] = df["preco_atual"] / df["preco_concorrente_base"]
    return cfg, df


def semanas_dividas(cfg: dict) -> dict:
    """{'semana_4': 85683.08, ...} -> {4: 85683.08, ...} (ordenado por semana)."""
    out = {}
    for k, v in cfg["dividas"].items():
        try:
            w = int(str(k).split("_")[-1])
        except ValueError:
            raise ValueError(f"chave de dívida inválida: {k!r} (use 'semana_<n>')") from None
        out[w] = float(v)
    return dict(sorted(out.items()))


def validar_entrada(cfg: dict, df: pd.DataFrame):
    """Rejeita entradas inválidas com mensagens claras (ValueError)."""
    faltando = [c for c in CHAVES_CONFIG if c not in cfg]
    if faltando:
        raise ValueError(f"config sem as chaves: {faltando}")
    T = cfg["numero_semanas"]
    if not isinstance(T, int) or T < 1:
        raise ValueError("numero_semanas deve ser inteiro >= 1")
    niveis = cfg["niveis_preco_relativos"]
    if not isinstance(niveis, list) or not niveis or any(
            (not isinstance(m, (int, float))) or m <= 0 for m in niveis):
        raise ValueError("niveis_preco_relativos deve ser uma lista não vazia de números > 0")
    v = cfg["variabilidade"]
    for k in ["sigma_demanda", "sigma_preco_concorrente", "sigma_custo_logistico"]:
        if k not in v or v[k] < 0:
            raise ValueError(f"variabilidade.{k} ausente ou negativo")
    if cfg["orcamento_promocional_semanal"] < 0:
        raise ValueError("orcamento_promocional_semanal não pode ser negativo")
    if not 0 <= cfg["limite_risco_falha"] <= 1:
        raise ValueError("limite_risco_falha deve estar em [0, 1]")
    divs = semanas_dividas(cfg)
    if not divs:
        raise ValueError("config sem dívidas")
    for w, val in divs.items():
        if not 1 <= w <= T:
            raise ValueError(f"dívida na semana {w} fora do horizonte 1..{T}")
        if val < 0:
            raise ValueError(f"dívida negativa na semana {w}")

    faltando = [c for c in COLUNAS_CATALOGO if c not in df.columns]
    if faltando:
        raise ValueError(f"catálogo sem as colunas: {faltando}")
    if df.empty:
        raise ValueError("catálogo vazio")
    if df["sku"].duplicated().any():
        raise ValueError(f"SKUs duplicados: {df.loc[df.sku.duplicated(), 'sku'].tolist()}")
    num = COLUNAS_CATALOGO[2:]
    try:
        df[num] = df[num].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"coluna numérica com valor não numérico: {e}") from None
    if df[num].isna().any().any():
        raise ValueError("catálogo com valores ausentes (NaN)")
    regras = {
        "custo_unitario": df.custo_unitario > 0, "preco_atual": df.preco_atual > 0,
        "estoque_inicial": (df.estoque_inicial >= 0) & (df.estoque_inicial == df.estoque_inicial.round()),
        "demanda_base_semanal": df.demanda_base_semanal >= 0, "elasticidade": df.elasticidade > 0,
        "taxa_marketplace": (df.taxa_marketplace >= 0) & (df.taxa_marketplace < 1),
        "custo_logistico_base": df.custo_logistico_base >= 0,
        "preco_concorrente_base": df.preco_concorrente_base > 0,
    }
    for col, ok in regras.items():
        if not ok.all():
            raise ValueError(f"valores inválidos em '{col}' nos SKUs {df.loc[~ok, 'sku'].tolist()}")


def aplicar_cenario(cfg: dict, df: pd.DataFrame, meta_fator=1.0, logistica_fator=1.0,
                    orcamento_fator=1.0, remover=(), concorrente_fator=1.0, incerteza_fator=1.0,
                    caixa_inicial=None):
    """Cenário-surpresa: altera dados/config por parâmetro, sem tocar no código.
    meta_fator multiplica as dívidas; concorrente_fator multiplica o preço do concorrente
    (0.92 = concorrente reduz 8%); incerteza_fator multiplica os três sigmas."""
    for nome, v in [("meta_fator", meta_fator), ("logistica_fator", logistica_fator),
                    ("orcamento_fator", orcamento_fator), ("concorrente_fator", concorrente_fator),
                    ("incerteza_fator", incerteza_fator)]:
        checar_fator(nome, v)
    cfg = json.loads(json.dumps(cfg))                     # cópia profunda
    df = remover_skus(df.copy(), list(remover))
    cfg["dividas"] = {k: v * meta_fator for k, v in cfg["dividas"].items()}
    cfg["orcamento_promocional_semanal"] *= orcamento_fator
    cfg["variabilidade"] = {k: v * incerteza_fator for k, v in cfg["variabilidade"].items()}
    if caixa_inicial is not None:
        cfg["caixa_inicial"] = float(caixa_inicial)
    df["custo_logistico_base"] = df["custo_logistico_base"] * logistica_fator
    df["preco_concorrente_base"] = df["preco_concorrente_base"] * concorrente_fator
    validar_entrada(cfg, df)
    return cfg, df


def replicar_catalogo(df: pd.DataFrame, vezes: int) -> pd.DataFrame:
    """Catálogo sintético maior (para o teste de escalabilidade)."""
    partes = []
    for k in range(vezes):
        c = df.copy()
        c["sku"] = c["sku"] + ("" if k == 0 else f"-r{k}")
        partes.append(c)
    return pd.concat(partes, ignore_index=True)


def gerar_cenarios(df: pd.DataFrame, cfg: dict, n: int, seed: int, escala_sigma: float = 1.0):
    """Gera n cenários de 12 semanas com TRÊS fontes de incerteza.

    1) Demanda:  eps[n,t,j] = exp(σd·(ρ·Zmercado_t + √(1-ρ²)·Zsku_tj) − σd²/2)
       (lognormal com média 1; choque de mercado comum a todos os SKUs)
    2) Concorrente: passeio aleatório log-normal do preço do concorrente,
       comp[n,t,j] = C_t / C_0, com passos σc·(ρ·Zmercado + √(1-ρ²)·Zsku)
    3) Custo logístico: multiplicador semanal comum, AR(1) com persistência φ,
       logm[n,t] = exp(σL·X_t − σL²/2),  X_t = φ·X_{t-1} + √(1-φ²)·Z_t
    As vendas são inteiras: demanda ~ Poisson(λ), com λ modulado pelos 3 choques
    (mistura Poisson-lognormal, que gera a sobredispersão típica de e-commerce).
    """
    rng = np.random.default_rng(seed)
    T, J = cfg["numero_semanas"], len(df)
    v = cfg["variabilidade"]
    sd = v["sigma_demanda"] * escala_sigma
    sc = v["sigma_preco_concorrente"] * escala_sigma
    sl = v["sigma_custo_logistico"] * escala_sigma

    zm = rng.standard_normal((n, T, 1))
    zi = rng.standard_normal((n, T, J))
    eps = np.exp(sd * (RHO_DEMANDA_COMUM * zm + np.sqrt(1 - RHO_DEMANDA_COMUM ** 2) * zi) - sd ** 2 / 2)

    cm = rng.standard_normal((n, T, 1))
    ci = rng.standard_normal((n, T, J))
    passos = sc * (RHO_CONC_COMUM * cm + np.sqrt(1 - RHO_CONC_COMUM ** 2) * ci)
    comp = np.exp(np.cumsum(passos, axis=1))

    x = np.zeros((n, T))
    zl = rng.standard_normal((n, T))
    x[:, 0] = zl[:, 0]
    for t in range(1, T):
        x[:, t] = PHI_LOGISTICA * x[:, t - 1] + np.sqrt(1 - PHI_LOGISTICA ** 2) * zl[:, t]
    logm = np.exp(sl * x - sl ** 2 / 2)

    return dict(eps=eps.astype(np.float32), comp=comp.astype(np.float32),
                logm=logm.astype(np.float32), n=n, seed=seed)


def share(g):
    return 1.0 / (1.0 + np.exp(K_BUYBOX * (g - 1.0 - DELTA_BUYBOX)))


def fator_concorrencia(df: pd.DataFrame, m, comp):
    """Fator de demanda pelo gap contra o concorrente (comp = C_t / C_0)."""
    p0 = df["preco_atual"].values
    c0 = df["preco_concorrente_base"].values
    g_ref = df["gap_ref"].values if "gap_ref" in df else p0 / c0
    g = p0 * m / (c0 * comp)
    return share(g) / share(g_ref)


def intensidade(df: pd.DataFrame, m, comp, eps):
    """Demanda média (Poisson) λ = base · m^(−e) · fator_concorrência · choque_demanda."""
    e = df["elasticidade"].values
    base = df["demanda_base_semanal"].values
    return base * np.power(m, -e) * fator_concorrencia(df, m, comp) * eps


def contrib_unit(df: pd.DataFrame, m, logm_esperado=1.0):
    """Contribuição por unidade (R$) a um multiplicador m e custo logístico esperado."""
    p = df["preco_atual"].values * m
    return (p * (1 - df["taxa_marketplace"].values)
            - df["custo_logistico_base"].values * logm_esperado
            - df["custo_unitario"].values)


def medir(func, *args, **kwargs):
    """Tempo (time.perf_counter) e pico de memória (tracemalloc) de uma chamada."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, pico / 1024 ** 2  # MiB
