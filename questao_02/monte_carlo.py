"""(A) Monte Carlo.

Dois papéis:
  1) estimar(): sobre os cenários de CONSTRUÇÃO (seed_base, N >= 2.000), estima a
     distribuição da demanda semanal de cada SKU em cada nível de preço. As PMFs
     resultantes alimentam a Programação Dinâmica; as amostras alimentam o Greedy.
  2) simular(): executa um plano completo (preços + promoções) em cenários de
     VALIDAÇÃO independentes (seed_final, N >= 5.000) e mede o risco de pagamento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modelo import contrib_unit, intensidade, semanas_dividas


def estimar(df: pd.DataFrame, cfg: dict, cen: dict):
    niveis = np.array(cfg["niveis_preco_relativos"])
    N, T, J, M = cen["n"], cfg["numero_semanas"], len(df), len(niveis)
    rng = np.random.default_rng(cen["seed"] + 1)
    S = df["estoque_inicial"].values

    lam = np.empty((N, T, J, M), dtype=np.float32)
    dem = np.empty((N, T, J, M), dtype=np.int32)
    for k, m in enumerate(niveis):
        lam[..., k] = intensidade(df, m, cen["comp"], cen["eps"])
        dem[..., k] = rng.poisson(lam[..., k])

    # PMF empírica P[d] por (SKU, semana, nível). Valores >= cap ficam no último
    # bin: como vendas = min(estoque, demanda), demanda acima do estoque é indiferente.
    pmfs = []
    for j in range(J):
        cap = int(min(S[j], dem[:, :, j, :].max()))
        Dj = np.minimum(dem[:, :, j, :], cap)
        P = np.zeros((T, M, cap + 1))
        for t in range(T):
            for k in range(M):
                P[t, k] = np.bincount(Dj[:, t, k], minlength=cap + 1) / N
        pmfs.append(P)

    logm_medio = cen["logm"].mean(axis=0)          # custo logístico esperado por semana

    # Tabela-resumo: comportamento de demanda/caixa por nível de preço (preço fixo 12 sem.)
    linhas = []
    for k, m in enumerate(niveis):
        cu = contrib_unit(df, m, logm_medio.mean())
        tot = dem[:, :, :, k].sum(axis=1)                     # demanda em 12 semanas (N,J)
        vend = np.minimum(tot, S)                             # limitada pelo estoque
        caixa = vend * cu
        for j in range(J):
            linhas.append(dict(
                sku=df.sku.iat[j], nivel=m, preco=df.preco_atual.iat[j] * m,
                demanda_sem_media=dem[:, :, j, k].mean(), demanda_sem_p5=np.percentile(dem[:, :, j, k], 5),
                demanda_sem_p95=np.percentile(dem[:, :, j, k], 95), contrib_unit=cu[j],
                prob_ruptura_12s=(tot[:, j] >= S[j]).mean(),
                caixa_12s_p5=np.percentile(caixa[:, j], 5), caixa_12s_media=caixa[:, j].mean(),
                caixa_12s_p95=np.percentile(caixa[:, j], 95)))
    resumo = pd.DataFrame(linhas)
    agregado = (resumo.groupby("nivel")[["caixa_12s_p5", "caixa_12s_media", "caixa_12s_p95"]]
                .sum().reset_index())
    return dict(niveis=niveis, lam=lam, dem=dem, pmfs=pmfs, logm_medio=logm_medio,
                resumo=resumo, agregado=agregado)


def simular(df: pd.DataFrame, cfg: dict, cen: dict, plano: dict, seed: int):
    """Simula o plano em todos os cenários (vetorizado em N).

    plano:
      'tipo' = 'fechado' -> PI[j, t, s] = índice do nível para estoque s (política da DP)
      'tipo' = 'fixo'    -> IDX[t, j]   = índice do nível pré-definido
      'uplift'[t, j]  multiplicador de demanda (destaque/mídia), 'frete'[t, j] bool,
      'custo_fixo'[t] custo semanal fixo das promoções.
    Regra das dívidas: em cada semana de vencimento (config "dividas", ex.: 4, 8 e 12)
    a dívida é paga se caixa >= dívida;
    caso contrário o compromisso FALHA e o saldo devedor é rolado (caixa negativo).
    """
    niveis = np.array(cfg["niveis_preco_relativos"])
    N, T, J = cen["n"], cfg["numero_semanas"], len(df)
    rng = np.random.default_rng(seed)
    p0 = df["preco_atual"].values
    tx = df["taxa_marketplace"].values
    L0 = df["custo_logistico_base"].values
    c = df["custo_unitario"].values
    e = df["elasticidade"].values
    semanas_div = semanas_dividas(cfg)              # {semana: valor}
    ordem_div = list(semanas_div)
    uplift = plano.get("uplift", np.ones((T, J)))
    frete = plano.get("frete", np.zeros((T, J), dtype=bool))
    custo_fixo = plano.get("custo_fixo", np.zeros(T))

    s = np.tile(df["estoque_inicial"].values, (N, 1)).astype(np.int64)
    caixa = np.full(N, cfg["caixa_inicial"], dtype=float)
    hist_caixa = np.zeros((N, T))
    hist_estoque = np.zeros((N, T + 1, J), dtype=np.int32)
    hist_estoque[:, 0] = s
    hist_nivel = np.zeros((N, T, J), dtype=np.int8)
    caixa_antes_div = np.zeros((N, len(ordem_div)))
    pago = np.zeros((N, len(ordem_div)), dtype=bool)
    receita = np.zeros(N); contrib_prod = np.zeros(N); gasto_promo = np.zeros(N)
    unidades = np.zeros(N)
    jj = np.arange(J)[None, :]

    for t in range(T):
        if plano["tipo"] == "fechado":
            idx = plano["PI"][jj, t, np.minimum(s, plano["PI"].shape[2] - 1)]
        else:
            idx = np.broadcast_to(plano["IDX"][t], (N, J))
        m = niveis[idx]
        p = p0 * m
        lam = intensidade(df, m, cen["comp"][:, t], cen["eps"][:, t]) * uplift[t]
        # frete promocional: cliente percebe corte de preço = frete pago pela empresa
        lam = np.where(frete[t], lam * np.clip(1 - L0 / p, 0.05, None) ** (-e), lam)
        q = np.minimum(s, rng.poisson(lam))
        s = s - q
        L = L0 * cen["logm"][:, t:t + 1]
        cu = p * (1 - tx) - L - c
        promo = (q * L0 * frete[t]).sum(axis=1) + custo_fixo[t]
        caixa += (q * cu).sum(axis=1) - promo
        receita += (q * p).sum(axis=1)
        contrib_prod += (q * cu).sum(axis=1)
        gasto_promo += promo
        unidades += q.sum(axis=1)
        if t + 1 in semanas_div:
            k = ordem_div.index(t + 1)
            caixa_antes_div[:, k] = caixa
            pago[:, k] = caixa >= semanas_div[t + 1]
            caixa -= semanas_div[t + 1]           # paga (ou rola a falta como saldo negativo)
        hist_caixa[:, t] = caixa
        hist_estoque[:, t + 1] = s
        hist_nivel[:, t] = idx

    return dict(semanas_div=ordem_div, caixa_final=caixa, hist_caixa=hist_caixa, hist_estoque=hist_estoque,
                hist_nivel=hist_nivel, caixa_antes_div=caixa_antes_div, pago=pago,
                receita=receita, contrib=contrib_prod, gasto_promo=gasto_promo,
                unidades=unidades, estoque_final=s)


def resumir(sim: dict, cfg: dict, nome: str, percentil_risco: float = 5):
    pago = sim["pago"]
    falha_qualquer = ~pago.all(axis=1)
    return dict(
        politica=nome,
        **{f"p_semana_{w}": pago[:, k].mean() for k, w in enumerate(sim["semanas_div"])},
        p_todas=pago.all(axis=1).mean(),
        caixa_final_medio=sim["caixa_final"].mean(),
        caixa_final_p_risco=np.percentile(sim["caixa_final"], percentil_risco),
        caixa_final_p50=np.percentile(sim["caixa_final"], 50),
        margem_contrib_pct=(100 * (sim["contrib"] - sim["gasto_promo"]).sum() / sim["receita"].sum()
                            if sim["receita"].sum() > 0 else 0.0),
        unidades_medias=sim["unidades"].mean(),
        estoque_final_medio=sim["estoque_final"].sum(axis=1).mean(),
        gasto_promo_medio=sim["gasto_promo"].mean(),
        cenarios_falha=int(falha_qualquer.sum()),
        atende_limite_risco=bool(falha_qualquer.mean() <= cfg["limite_risco_falha"]),
    )
