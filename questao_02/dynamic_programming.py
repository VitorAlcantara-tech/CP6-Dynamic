"""(B) Programação Dinâmica — trajetória semanal de preços por SKU.

FORMULAÇÃO (por SKU j)
  Estado    (t, s): semana t ∈ {1..12} e estoque remanescente s ∈ {0..S_j}
  Decisão   m ∈ níveis de preço permitidos (multiplicadores do config)
  Aleatório D ~ PMF_{j,t,m}(d) estimada pelo Monte Carlo de construção
  Transição s' = s − min(s, D)                     (vende o que houver em estoque)
  Recompensa r_t(m) · min(s, D), com
            r_t(m) = w_t · [m·p0·(1 − taxa) − E_MC[custo_logístico_t] − custo_unitário]
            w_t = (1 + ρ)^−(t−1): peso do caixa que chega antes (ρ calibrado pelo MC)
  Bellman   V_t(s) = max_m  Σ_d P_{t,m}(d) · [ r_t(m)·min(s,d) + V_{t+1}(s − min(s,d)) ]
  Base      V_13(s) = 0  (o estoque que sobra mantém seu valor de custo: não gera nem
            consome caixa no horizonte)
  Objetivo  V_1(S_j) = máxima contribuição (ponderada) esperada em 12 semanas.

Por que por SKU: o caixa acopla os SKUs, mas um estado conjunto (t, s_1..s_24, caixa)
é intratável (Π S_j ≈ 10^50 estados). Decompomos por SKU com o peso w_t como "preço
sombra" do caixa no tempo; o risco conjunto (as 3 dívidas) é medido pelo Monte Carlo,
que escolhe ρ e valida o plano.

REUTILIZAÇÃO DE SUBPROBLEMAS: V_{t+1}(s') é calculado UMA vez e lido por todos os
pares (s, m, d) com s − min(s, d) = s'. Ex.: V_{t+1}(0) é lido por todo (s, m, d ≥ s).
O contador 'leituras' / 'estados' mede isso.
"""
from __future__ import annotations

import numpy as np

from modelo import contrib_unit


def pesos(T: int, rho: float):
    return (1.0 + rho) ** -np.arange(T, dtype=float)


def resolver_sku(pmf: np.ndarray, r: np.ndarray, S: int):
    """Bottom-up (t = 12 → 1), vetorizado em s e d.
    pmf: (T, M, D+1) · r: (T, M) recompensa unitária ponderada · S: estoque inicial.
    Retorna V (T+1, S+1), PI (T, S+1) com o índice do nível ótimo e contadores."""
    T, M, D1 = pmf.shape
    V = np.zeros((T + 1, S + 1))
    PI = np.zeros((T, S + 1), dtype=np.int8)
    sv = np.arange(S + 1)[:, None]
    q = np.minimum(sv, np.arange(D1)[None, :])        # vendas (S+1, D+1)
    prox = sv - q                                     # estado seguinte s'
    leituras = 0
    for t in range(T - 1, -1, -1):
        Vnext = V[t + 1][prox]                        # REUSO: lê V_{t+1} já resolvido
        valores = np.empty((M, S + 1))
        for k in range(M):
            valores[k] = (pmf[t, k][None, :] * (r[t, k] * q + Vnext)).sum(axis=1)
            leituras += int((pmf[t, k] > 0).sum()) * (S + 1)
        # empate (ex.: s = 0) -> prefere o preço mais alto (não queima margem à toa)
        k_opt = (M - 1) - np.argmax(valores[::-1], axis=0)
        PI[t] = k_opt
        V[t] = valores[k_opt, np.arange(S + 1)]
    return V, PI, dict(estados=T * (S + 1), leituras=leituras)


def resolver_top_down(pmf: np.ndarray, r: np.ndarray, S: int, memo: bool = True,
                      T_lim: int | None = None):
    """Mesma recorrência, Top-Down. Com memo=True cada (t, s) é resolvido uma vez;
    com memo=False a árvore de chamadas explode (usado só para demonstrar o reuso)."""
    T, M, D1 = pmf.shape
    T = T_lim or T
    cache = {}
    cont = dict(chamadas=0, acertos_cache=0)
    suportes = [[np.nonzero(pmf[t, k])[0] for k in range(M)] for t in range(T)]

    def V(t, s):
        cont["chamadas"] += 1
        if t == T:
            return 0.0
        if memo and (t, s) in cache:
            cont["acertos_cache"] += 1
            return cache[(t, s)]
        melhor = -np.inf
        for k in range(M):
            tot = 0.0
            for d in suportes[t][k]:
                qd = min(s, d)
                tot += pmf[t, k, d] * (r[t, k] * qd + V(t + 1, s - qd))
            melhor = max(melhor, tot)
        if memo:
            cache[(t, s)] = melhor
        return melhor

    v = V(0, S)
    cont["subproblemas_distintos"] = len(cache)
    return v, cont


def resolver_catalogo(df, est: dict, rho: float):
    """Roda a DP para todos os SKUs. Recompensa usa o custo logístico esperado (MC)."""
    T = len(est["logm_medio"])
    w = pesos(T, rho)
    S = df["estoque_inicial"].values
    Smax = int(S.max())
    PI_all = np.zeros((len(df), T, Smax + 1), dtype=np.int8)
    Vs, rs, info = [], [], dict(estados=0, leituras=0)
    for j in range(len(df)):
        r = np.array([[w[t] * contrib_unit(df.iloc[[j]], m, est["logm_medio"][t])[0]
                       for m in est["niveis"]] for t in range(T)])
        V, PI, c = resolver_sku(est["pmfs"][j], r, int(S[j]))
        PI_all[j, :, :S[j] + 1] = PI
        PI_all[j, :, S[j] + 1:] = PI[:, -1:]
        Vs.append(V); rs.append(r)
        info["estados"] += c["estados"]; info["leituras"] += c["leituras"]
    return dict(PI=PI_all, V=Vs, r=rs, rho=rho, info=info)


def reconstruir(pmf: np.ndarray, PI: np.ndarray, S: int, niveis, ajuste_vendas=None):
    """Reconstrói o caminho ótimo ao longo do estoque ESPERADO:
    s_1 = S;  m_t = π_t(round(s_t));  s_{t+1} = s_t − E[min(s_t, D)]."""
    T = PI.shape[0]
    s = float(S)
    caminho = []
    for t in range(T):
        si = int(round(s))
        k = int(PI[t, si])
        P = pmf[t, k]
        ev = float((P * np.minimum(si, np.arange(len(P)))).sum())
        if ajuste_vendas is not None:
            ev = ajuste_vendas(t, si, k, ev)
        caminho.append(dict(semana=t + 1, estoque_inicio=si, nivel=niveis[k], idx=k, vendas_esp=ev))
        s = max(0.0, s - ev)
    return caminho
