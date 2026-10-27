"""(C) Greedy semanal de recursos promocionais escassos.

Recebe a política da DP e, semana a semana:
  1) segue o plano de preços da DP ao longo do estoque esperado (π_t(s_t));
  2) para cada SKU e recurso (destaque, mídia, frete promocional) calcula o ganho
     marginal com as amostras do Monte Carlo de construção, JÁ DESCONTANDO o custo
     de oportunidade do estoque dado pela DP:
         Δ = E[ r·Q1 + V_{t+1}(s − Q1) ] − E[ r·Q0 + V_{t+1}(s − Q0) ]
     (vender uma unidade a mais hoje só vale se valer mais que guardá-la);
  3) ordena por razão Δ / custo (benefício marginal por real) e aceita enquanto
     razão >= 1 (o recurso se paga), houver orçamento semanal, no máximo 1 recurso
     por SKU e no máximo 'limite_semana' destaques;
  4) atualiza o estoque esperado do plano com as vendas extras e passa à semana seguinte
     (o plano da DP é reconsultado no novo estoque).
Uplift de demanda é Poisson aditiva: Poisson(λ(1+u)) = Poisson(λ) + Poisson(λu),
o que dá números aleatórios comuns entre "com" e "sem" recurso.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modelo import RECURSOS


def planejar(df: pd.DataFrame, cfg: dict, est: dict, dp: dict, seed: int, min_razao: float = 1.0):
    rng = np.random.default_rng(seed)
    niveis = est["niveis"]
    T, J = cfg["numero_semanas"], len(df)
    orc = cfg["orcamento_promocional_semanal"]
    L0 = df["custo_logistico_base"].values
    p0 = df["preco_atual"].values
    e = df["elasticidade"].values
    S0 = df["estoque_inicial"].values
    w = (1.0 + dp["rho"]) ** -np.arange(T, dtype=float)

    s_plan = S0.astype(float)
    uplift = np.ones((T, J)); frete = np.zeros((T, J), dtype=bool); custo_fixo = np.zeros(T)
    alocacoes, trajetoria, uso = [], [], []

    for t in range(T):
        cand = []
        base_vendas = np.zeros(J)
        k_sem = np.zeros(J, dtype=int)
        for j in range(J):
            s = int(round(s_plan[j]))
            k = int(dp["PI"][j, t, min(s, dp["PI"].shape[2] - 1)])
            k_sem[j] = k
            V1 = dp["V"][j][t + 1]
            r = dp["r"][j][t, k]
            d0 = est["dem"][:, t, j, k]
            lam = est["lam"][:, t, j, k]
            q0 = np.minimum(s, d0)
            v0 = (r * q0 + V1[s - q0]).mean()
            base_vendas[j] = q0.mean()
            if s == 0:
                continue
            p = p0[j] * niveis[k]
            for nome, rec in RECURSOS.items():
                if nome == "frete_promocional":
                    fator = max(1 - L0[j] / p, 0.05) ** (-e[j])
                else:
                    fator = 1 + rec["uplift"]
                q1 = np.minimum(s, d0 + rng.poisson(lam * (fator - 1)))
                ganho = (r * q1 + V1[s - q1]).mean() - v0          # em R$ ponderados por w_t
                custo = rec["custo_fixo"] + (L0[j] * q1.mean() if nome == "frete_promocional" else 0)
                if custo <= 0:
                    continue
                razao = ganho / (w[t] * custo)
                cand.append(dict(semana=t + 1, j=j, sku=df.sku.iat[j], recurso=nome, nivel=niveis[k],
                                 estoque_plano=s, custo=custo, ganho=ganho / w[t],
                                 liquido=ganho / w[t] - custo, razao=razao,
                                 vendas_extra=q1.mean() - q0.mean(), fator=fator))
        # seleção gulosa por razão (benefício marginal por real)
        cand.sort(key=lambda c: -c["razao"])
        restante, usados, n_dest = orc, set(), 0
        for c in cand:
            if c["razao"] < min_razao or c["ganho"] <= 0:
                break
            if c["j"] in usados or c["custo"] > restante:
                continue
            lim = RECURSOS[c["recurso"]]["limite_semana"]
            if lim is not None and c["recurso"] == "destaque" and n_dest >= lim:
                continue
            usados.add(c["j"]); restante -= c["custo"]
            if c["recurso"] == "destaque":
                n_dest += 1
            if c["recurso"] == "frete_promocional":
                frete[t, c["j"]] = True
            else:
                uplift[t, c["j"]] = c["fator"]
                custo_fixo[t] += c["custo"]
            base_vendas[c["j"]] += c["vendas_extra"]
            alocacoes.append(c)
        uso.append(dict(semana=t + 1, orcamento=orc, usado=orc - restante,
                        n_candidatos=len(cand), n_alocados=len(usados)))
        for j in range(J):
            trajetoria.append(dict(semana=t + 1, sku=df.sku.iat[j], estoque_inicio=s_plan[j],
                                   nivel=niveis[k_sem[j]], preco=p0[j] * niveis[k_sem[j]],
                                   vendas_esp=base_vendas[j],
                                   recurso=next((a["recurso"] for a in alocacoes
                                                 if a["semana"] == t + 1 and a["j"] == j), "-")))
        s_plan = np.maximum(0, s_plan - base_vendas)

    plano_promo = dict(uplift=uplift, frete=frete, custo_fixo=custo_fixo)
    return plano_promo, pd.DataFrame(alocacoes), pd.DataFrame(trajetoria), pd.DataFrame(uso)
