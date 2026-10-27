"""
Modelo econômico compartilhado pelos 3 algoritmos (Brute Force, GA e Greedy):
premissas, leitura dos dados, preços permitidos e função de avaliação (fitness).
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent.parent      # raiz do projeto (comum.py)
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))
from comum import checar_fator, remover_skus  # noqa: E402

GAMMA_CANIBALIZACAO = 0.5    # sensibilidade de share dentro da categoria
MAX_FRAC_COM_DESCONTO = 0.30  # no máx. 30% dos SKUs com preço < preço atual
DESCONTO_PROFUNDO = 0.90     # multiplicador < 0.90 = desconto "profundo"
IDADE_MIN_DESC_PROFUNDO = 180  # desconto profundo só p/ estoque com >= 180 dias
PENALIDADE = 1e6             # penalidade por SKU acima do limite de descontos
# Custo de caixa do estoque que SOBRA ao fim dos 90 dias (armazenagem no
# fulfillment/CD + capital parado). É aqui que idade_estoque entra no caixa:
TAXA_ARMAZENAGEM_90D = 0.045       # 1,5% a.m. sobre o custo, por 3 meses
SOBRETAXA_ENVELHECIDO = 0.045      # +4,5% p/ itens com >= 180 dias (armazenagem de longo prazo)

# Incentivos comerciais (etapa C). Para cada opção:
#   custo_unit  = quanto a empresa paga por unidade vendida
#   desc_perc   = desconto percebido pelo cliente (fração do preço) -> move demanda
INCENTIVOS = {
    # cupom: o cliente percebe 100% do valor e a empresa paga 100%
    "cupom_5":     dict(tipo="cupom", s=0.05, percebido=1.00, resgate=1.00),
    "cupom_10":    dict(tipo="cupom", s=0.10, percebido=1.00, resgate=1.00),
    # cashback: valor diferido (cliente percebe 75%), mas só 65% é resgatado (breakage)
    "cashback_5":  dict(tipo="cashback", s=0.05, percebido=0.75, resgate=0.65),
    "cashback_10": dict(tipo="cashback", s=0.10, percebido=0.75, resgate=0.65),
    # frete subsidiado: empresa absorve o frete (= custo_logistico) cobrado do cliente;
    # efeito "frete grátis" faz o cliente perceber 130% do valor
    "frete_gratis": dict(tipo="frete", s=None, percebido=1.30, resgate=1.00),
}

COLUNAS_CATALOGO = ["sku", "categoria", "custo_unitario", "preco_atual", "estoque",
                    "demanda_base_90d", "elasticidade", "taxa_marketplace", "custo_logistico",
                    "idade_estoque_dias"]
CHAVES_CONFIG = ["seed_base", "seed_final", "meta_caixa_90d", "divida_total_90d",
                 "orcamento_promocional", "margem_minima_percentual", "niveis_preco_relativos"]


def carregar(config_path: Path, catalogo_path: Path):
    config_path, catalogo_path = Path(config_path), Path(catalogo_path)
    for p in (config_path, catalogo_path):
        if not p.exists():
            raise FileNotFoundError(f"arquivo não encontrado: {p}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    df = pd.read_csv(catalogo_path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    validar_entrada(cfg, df)
    return cfg, df


def validar_entrada(cfg: dict, df: pd.DataFrame):
    """Rejeita entradas inválidas com mensagens claras (ValueError)."""
    faltando = [c for c in CHAVES_CONFIG if c not in cfg]
    if faltando:
        raise ValueError(f"config sem as chaves: {faltando}")
    niveis = cfg["niveis_preco_relativos"]
    if not isinstance(niveis, list) or not niveis or any((not isinstance(m, (int, float))) or m <= 0
                                                         for m in niveis):
        raise ValueError("niveis_preco_relativos deve ser uma lista não vazia de números > 0")
    if cfg["orcamento_promocional"] < 0:
        raise ValueError("orcamento_promocional não pode ser negativo")
    if not 0 <= cfg["margem_minima_percentual"] < 1:
        raise ValueError("margem_minima_percentual deve estar em [0, 1)")
    if cfg["meta_caixa_90d"] < 0 or cfg["divida_total_90d"] < 0:
        raise ValueError("meta de caixa e dívida não podem ser negativas")

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
        "estoque": df.estoque >= 0, "demanda_base_90d": df.demanda_base_90d >= 0,
        "elasticidade": df.elasticidade > 0, "custo_logistico": df.custo_logistico >= 0,
        "taxa_marketplace": (df.taxa_marketplace >= 0) & (df.taxa_marketplace < 1),
        "idade_estoque_dias": df.idade_estoque_dias >= 0,
    }
    for col, ok in regras.items():
        if not ok.all():
            raise ValueError(f"valores inválidos em '{col}' nos SKUs {df.loc[~ok, 'sku'].tolist()}")


def aplicar_cenario(cfg: dict, df: pd.DataFrame, meta_fator=1.0, logistica_fator=1.0,
                    orcamento_fator=1.0, remover=()):
    """Cenário-surpresa: altera dados/config por parâmetro, sem tocar no código."""
    for nome, v in [("meta_fator", meta_fator), ("logistica_fator", logistica_fator),
                    ("orcamento_fator", orcamento_fator)]:
        checar_fator(nome, v)
    cfg = dict(cfg)
    df = remover_skus(df.copy(), list(remover))
    cfg["meta_caixa_90d"] = cfg["meta_caixa_90d"] * meta_fator
    cfg["orcamento_promocional"] = cfg["orcamento_promocional"] * orcamento_fator
    df["custo_logistico"] = df["custo_logistico"] * logistica_fator
    validar_entrada(cfg, df)
    return cfg, df


def margem(df: pd.DataFrame, m, custo_incentivo_unit=0.0):
    """Margem de contribuição % sobre o preço praticado."""
    p = df["preco_atual"].values * m
    contrib = (p * (1 - df["taxa_marketplace"].values) - df["custo_logistico"].values
               - df["custo_unitario"].values - custo_incentivo_unit)
    return contrib / p


def gerar_precos_permitidos(df, cfg):
    """Conjunto discreto de preços por SKU, gerado com seed_base.

    1) Sorteio (seed_base): cada SKU recebe de 4 a 7 dos níveis do config,
       sempre incluindo 1.00 (preço atual).
    2) Filtro de margem: remove níveis com margem < margem_minima.
    3) Filtro anti-desconto indiscriminado: níveis < 0.90 só para itens com
       idade_estoque >= 180 dias.
    4) Se nada sobrar, o SKU fica fora da campanha (piso de margem é rígido).
    """
    rng = np.random.default_rng(cfg["seed_base"])
    niveis = np.array(cfg["niveis_preco_relativos"])
    mmin = cfg["margem_minima_percentual"]
    permitidos, sinalizados = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        k = rng.integers(min(4, len(niveis)), len(niveis) + 1)
        sorteio = set(rng.choice(niveis, size=k, replace=False).round(2)) | {1.0}
        cand = sorted(sorteio)
        ok = []
        for m in cand:
            marg = margem(df.iloc[[i]], m)[0]
            if marg < mmin:
                continue
            if m < DESCONTO_PROFUNDO and row["idade_estoque_dias"] < IDADE_MIN_DESC_PROFUNDO:
                continue
            ok.append(m)
        if not ok:
            # nenhum preço sorteado respeita o piso -> SKU fica FORA da campanha
            # (não vende nos 90 dias; o estoque continua gerando custo de armazenagem)
            ok = [1.0]
            sinalizados.append(row["sku"])
        permitidos.append(np.array(ok))
    return permitidos, sinalizados


@dataclass
class Instancia:
    """Instância pré-computada (vetorizada) de um conjunto de SKUs."""
    df: pd.DataFrame
    permitidos: list          # lista de arrays de multiplicadores
    cat_idx: np.ndarray       # índice inteiro da categoria de cada SKU
    max_desc: int             # máx. de SKUs com desconto

    @classmethod
    def criar(cls, df, permitidos, inativos=()):
        df = df.reset_index(drop=True).copy()
        df["ativo"] = (~df["sku"].isin(list(inativos))).astype(float)
        df["taxa_hold"] = TAXA_ARMAZENAGEM_90D + SOBRETAXA_ENVELHECIDO * (
            df["idade_estoque_dias"] >= IDADE_MIN_DESC_PROFUNDO)
        cats = pd.factorize(df["categoria"])[0]
        max_desc = int(np.floor(MAX_FRAC_COM_DESCONTO * len(df)))
        return cls(df, permitidos, cats, max_desc)

    @property
    def n(self):
        return len(self.df)

    # matriz (n x Lmax) com multiplicadores; posições vazias = NaN
    def tabela_m(self):
        L = max(len(p) for p in self.permitidos)
        T = np.full((self.n, L), np.nan)
        for i, p in enumerate(self.permitidos):
            T[i, :len(p)] = p
        return T


def avaliar_vetorizado(inst: Instancia, M: np.ndarray, detalhado=False):
    """Avalia uma MATRIZ de políticas M (B x n multiplicadores) de uma vez.

    Demanda:  q = min(estoque, base * m^(-e) * (m / m_cat)^(-gamma))
       m_cat = média dos multiplicadores dos SKUs da mesma categoria (canibalização:
       quem fica mais barato que os pares da categoria rouba participação).
    Caixa operacional (contribuição em caixa):
       q * [p(1 - taxa) - custo_logistico - custo_unitario]
       - (estoque - q) * custo_unitario * taxa_hold      (armazenagem do que sobra)
    Restrição: nº de SKUs com m < 1 <= max_desc (penalidade).
    """
    d = inst.df
    M = np.atleast_2d(M)
    B, n = M.shape
    # média por categoria (vetorizada com bincount por linha)
    ncat = inst.cat_idx.max() + 1
    soma = np.zeros((B, ncat))
    for c in range(ncat):  # ncat é pequeno (6): loop barato
        mask = inst.cat_idx == c
        soma[:, c] = M[:, mask].mean(axis=1)
    m_cat = soma[:, inst.cat_idx]

    e = d["elasticidade"].values
    dem = d["demanda_base_90d"].values * M ** (-e) * (M / m_cat) ** (-GAMMA_CANIBALIZACAO)
    q = np.minimum(d["estoque"].values, dem) * d["ativo"].values
    p = d["preco_atual"].values * M
    contrib_unit = (p * (1 - d["taxa_marketplace"].values) - d["custo_logistico"].values
                    - d["custo_unitario"].values)
    hold = (d["estoque"].values - q) * d["custo_unitario"].values * d["taxa_hold"].values
    caixa = (q * contrib_unit - hold).sum(axis=1)
    n_desc = (M < 1.0 - 1e-9).sum(axis=1)
    excesso = np.maximum(0, n_desc - inst.max_desc)
    fitness = caixa - PENALIDADE * excesso
    if detalhado:
        return dict(q=q, p=p, contrib_unit=contrib_unit, caixa=caixa, hold=hold,
                    fitness=fitness, n_desc=n_desc)
    return fitness


def avaliar_ingenuo(df_rows, ms, max_desc):
    """Versão INGÊNUA (antes da otimização): uma política por vez, em Python puro,
    recalculando médias de categoria com dicionários a cada chamada."""
    cat_soma, cat_cont = {}, {}
    for row, m in zip(df_rows, ms):
        cat_soma[row["categoria"]] = cat_soma.get(row["categoria"], 0) + m
        cat_cont[row["categoria"]] = cat_cont.get(row["categoria"], 0) + 1
    total, n_desc = 0.0, 0
    for row, m in zip(df_rows, ms):
        m_cat = cat_soma[row["categoria"]] / cat_cont[row["categoria"]]
        dem = (row["demanda_base_90d"] * m ** (-row["elasticidade"])
               * (m / m_cat) ** (-GAMMA_CANIBALIZACAO))
        q = min(row["estoque"], dem) * row["ativo"]
        p = row["preco_atual"] * m
        total += q * (p * (1 - row["taxa_marketplace"]) - row["custo_logistico"]
                      - row["custo_unitario"])
        total -= (row["estoque"] - q) * row["custo_unitario"] * row["taxa_hold"]
        if m < 1.0 - 1e-9:
            n_desc += 1
    return total - PENALIDADE * max(0, n_desc - max_desc)


def medir(func, *args, **kwargs):
    """Mede tempo (perf_counter) e pico de memória (tracemalloc)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, pico / 1024  # KiB
