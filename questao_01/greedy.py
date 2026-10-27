"""(C) Greedy: distribui o orçamento de incentivos (cupom, cashback, frete
subsidiado) sobre a política de preços produzida pelo GA."""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from modelo import GAMMA_CANIBALIZACAO, INCENTIVOS, Instancia, avaliar_vetorizado


def opcoes_incentivo(inst: Instancia, politica, mmin):
    """Para cada SKU (com o preço DEFINIDO PELO GA), calcula cada opção de incentivo:
    custo total, ganho de contribuição bruto e caixa líquido.
    O efeito na demanda usa o preço efetivo percebido; a canibalização usa os
    preços de tabela (fixos pelo GA), o que torna os SKUs separáveis no Greedy."""
    base = avaliar_vetorizado(inst, politica[None, :], detalhado=True)
    d = inst.df
    q0, p, cu = base["q"][0], base["p"][0], base["contrib_unit"][0]
    m = politica
    e = d["elasticidade"].values
    estoque = d["estoque"].values
    # demanda irrestrita (antes do teto de estoque) no preço do GA
    ncat = inst.cat_idx.max() + 1
    m_cat = np.array([m[inst.cat_idx == c].mean() for c in range(ncat)])[inst.cat_idx]
    dem_irrestrita = d["demanda_base_90d"].values * m ** (-e) * (m / m_cat) ** (-GAMMA_CANIBALIZACAO)

    linhas = []
    hold_u = d["custo_unitario"].values * d["taxa_hold"].values
    for j in range(inst.n):
        if d["ativo"].values[j] == 0:
            continue
        caixa0 = q0[j] * cu[j] - (estoque[j] - q0[j]) * hold_u[j]
        for nome, op in INCENTIVOS.items():
            if op["tipo"] == "frete":
                valor = d["custo_logistico"].values[j]
            else:
                valor = op["s"] * p[j]
            custo_unit = valor * op["resgate"]
            desc_perc = op["percebido"] * valor / p[j]
            if desc_perc >= 0.95:
                continue
            # piso de margem continua valendo após o incentivo
            marg = (cu[j] - custo_unit) / p[j]
            if marg < mmin:
                continue
            q1 = min(estoque[j], dem_irrestrita[j] * (1 - desc_perc) ** (-e[j]))
            custo_total = q1 * custo_unit
            # ganho antes de pagar o incentivo (inclui armazenagem evitada)
            ganho_bruto = q1 * cu[j] - (estoque[j] - q1) * hold_u[j] - caixa0
            liquido = ganho_bruto - custo_total          # efeito no caixa
            linhas.append(dict(j=j, sku=d["sku"].values[j], incentivo=nome,
                               custo=custo_total, ganho_bruto=ganho_bruto,
                               liquido=liquido, q0=q0[j], q1=q1))
    return pd.DataFrame(linhas), base


def greedy_incentivos(opcoes: pd.DataFrame, orcamento: float, min_razao: float = 1.0):
    """Greedy de 'upgrades' (mochila de múltipla escolha):
    cada SKU pode ter no máximo 1 incentivo. Estado inicial: nenhum.
    A cada passo escolhe o upgrade (SKU j: opção atual -> opção mais cara)
    com maior razão  Δganho_bruto / Δcusto  (benefício marginal por real),
    aceitando apenas se razão >= min_razao e se cabe no orçamento restante.
      min_razao = 1.0 (regra principal): cada R$ investido precisa voltar >= R$ 1
                  em contribuição -> o incentivo nunca reduz o caixa da dívida.
      min_razao = 0+  (cenário 'verba carimbada'): o orçamento já está reservado
                  e não volta para a dívida; gasta onde houver qualquer ganho.
    Usa heap (max-heap) com reinserções preguiçosas.
    """
    if orcamento < 0:
        raise ValueError("orçamento não pode ser negativo")
    if opcoes.empty:
        return {}, pd.DataFrame(), 0.0
    por_sku = {j: g.sort_values("custo").to_dict("records") for j, g in opcoes.groupby("j")}
    atual = {j: dict(custo=0.0, ganho_bruto=0.0, liquido=0.0, incentivo=None) for j in por_sku}
    restante = orcamento
    heap = []

    def empilhar(j):
        cur = atual[j]
        melhor = None
        for op in por_sku[j]:
            dc = op["custo"] - cur["custo"]
            if dc <= 1e-9:
                continue
            razao = (op["ganho_bruto"] - cur["ganho_bruto"]) / dc
            if razao < min_razao or razao <= 0:
                continue
            if melhor is None or razao > melhor[0]:
                melhor = (razao, dc, op)
        if melhor:
            heapq.heappush(heap, (-melhor[0], j, melhor[1], melhor[2]["incentivo"]))

    for j in por_sku:
        empilhar(j)

    passos = []
    while heap:
        neg_razao, j, dc, nome = heapq.heappop(heap)
        op = next(o for o in por_sku[j] if o["incentivo"] == nome)
        dc_real = op["custo"] - atual[j]["custo"]
        if abs(dc_real - dc) > 1e-6:      # entrada obsoleta
            continue
        if dc > restante:
            # não cabe: tenta opção mais barata/alternativa deste SKU que caiba
            alternativas = [o for o in por_sku[j]
                            if 0 < o["custo"] - atual[j]["custo"] <= restante
                            and (o["ganho_bruto"] - atual[j]["ganho_bruto"])
                            / (o["custo"] - atual[j]["custo"]) >= max(min_razao, 1e-12)]
            if alternativas:
                o = max(alternativas, key=lambda o: (o["ganho_bruto"] - atual[j]["ganho_bruto"])
                        / (o["custo"] - atual[j]["custo"]))
                heapq.heappush(heap, (-(o["ganho_bruto"] - atual[j]["ganho_bruto"])
                                      / (o["custo"] - atual[j]["custo"]),
                                      j, o["custo"] - atual[j]["custo"], o["incentivo"]))
            continue
        restante -= dc
        passos.append(dict(passo=len(passos) + 1, sku=op["sku"], de=atual[j]["incentivo"],
                           para=nome, razao=-neg_razao, delta_custo=dc,
                           delta_liquido=op["liquido"] - atual[j]["liquido"],
                           orcamento_restante=restante))
        atual[j] = dict(custo=op["custo"], ganho_bruto=op["ganho_bruto"],
                        liquido=op["liquido"], incentivo=nome, q1=op["q1"])
        empilhar(j)
    return atual, pd.DataFrame(passos), orcamento - restante


def greedy_ingenuo(opcoes: pd.DataFrame, orcamento: float, min_razao: float = 1.0):
    """Mesma regra, versão ingênua: a cada passo reescaneia TODOS os pares
    (SKU, opção) -> O(S · n · K). Usada para a comparação antes/depois."""
    if orcamento < 0:
        raise ValueError("orçamento não pode ser negativo")
    if opcoes.empty:
        return {}, 0.0
    por_sku = {j: g.to_dict("records") for j, g in opcoes.groupby("j")}
    atual = {j: dict(custo=0.0, ganho_bruto=0.0, liquido=0.0, incentivo=None) for j in por_sku}
    restante = orcamento
    while True:
        melhor = None
        for j, ops in por_sku.items():
            cur = atual[j]
            for op in ops:
                dc = op["custo"] - cur["custo"]
                if dc <= 1e-9 or dc > restante:
                    continue
                r = (op["ganho_bruto"] - cur["ganho_bruto"]) / dc
                if r < min_razao or r <= 0:
                    continue
                if melhor is None or r > melhor[0]:
                    melhor = (r, j, op)
        if melhor is None:
            break
        _, j, op = melhor
        restante -= op["custo"] - atual[j]["custo"]
        atual[j] = dict(custo=op["custo"], ganho_bruto=op["ganho_bruto"],
                        liquido=op["liquido"], incentivo=op["incentivo"])
    return atual, orcamento - restante
