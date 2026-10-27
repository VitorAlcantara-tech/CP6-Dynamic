"""
QUESTÃO 2 — Plano de Sobrevivência em 12 Semanas

Nesta questão utilizamos 3 algoritmos:
  (A) monte_carlo.py          -> estima a distribuição de demanda/caixa por SKU, semana
                                 e nível de preço (cenários de construção, seed_base) e, no
                                 fim, valida o plano em cenários NOVOS (seed_final)
  (B) dynamic_programming.py  -> estado (semana, estoque): escolhe o preço de cada
                                 semana a partir das PMFs do Monte Carlo
  (C) greedy.py               -> segue o plano da DP e aloca destaque / mídia / frete
                                 promocional a cada semana dentro do orçamento
modelo.py contém premissas, dados e geração de cenários usados pelos três.

Exemplos:
  python questao_02/main.py                                   # instância oficial
  python questao_02/main.py --seed 14 --cenarios 5000
  python questao_02/main.py --concorrente-fator 0.92 --logistica-fator 1.15
  python questao_02/main.py --meta-fator 1.10 --orcamento-fator 0.75 --remover-sku Q2-003
  python questao_02/main.py --help
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE.parent) not in sys.path:
    sys.path.insert(0, str(BASE.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from comum import gerar_manifesto, parser_base, resolver_seeds
from modelo import aplicar_cenario, carregar, caminhos, gerar_cenarios, medir, replicar_catalogo, semanas_dividas
import monte_carlo as mc
import dynamic_programming as pdm
import greedy as gr

MIN_CONSTRUCAO, MIN_VALIDACAO = 2000, 5000      # mínimos exigidos pelo enunciado


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = parser_base("Questão 2 — Plano de Sobrevivência (Monte Carlo -> DP -> Greedy -> Monte Carlo)")
    g = p.add_argument_group("Monte Carlo e DP")
    g.add_argument("--cenarios", type=int, default=MIN_VALIDACAO, help="cenários de VALIDAÇÃO")
    g.add_argument("--cenarios-construcao", type=int, default=MIN_CONSTRUCAO,
                   help="cenários de CONSTRUÇÃO (estimação das PMFs)")
    g.add_argument("--grade-rho", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10],
                   help="valores de ρ testados na calibração")
    g.add_argument("--percentil-risco", type=float, default=5, help="percentil do caixa final reportado")
    s = p.add_argument_group("cenário-surpresa (específicos da Q2)")
    s.add_argument("--concorrente-fator", type=float, default=1.0,
                   help="multiplica o preço do concorrente. Ex.: 0.92 = concorrente reduz 8%%")
    s.add_argument("--incerteza-fator", type=float, default=1.0, help="multiplica os três sigmas")
    s.add_argument("--caixa-inicial", type=float, default=None, help="substitui o caixa inicial")
    args = p.parse_args(argv)
    if args.cenarios < 1 or args.cenarios_construcao < 1:
        p.error("--cenarios e --cenarios-construcao devem ser >= 1")
    if not 0 < args.percentil_risco < 100:
        p.error("--percentil-risco deve estar em (0, 100)")
    if any(r < 0 for r in args.grade_rho):
        p.error("--grade-rho só aceita valores >= 0")
    return args


# ---------------------------------------------------------------------------
# Escolhas automáticas (independentes da instância)
# ---------------------------------------------------------------------------
def escolher_representativos(df):
    """3 SKUs que contam histórias diferentes: maior ticket (maior preço), mais vendido
    (maior demanda base semanal) e mais encalhado (maior cobertura estoque/demanda)."""
    cob = df.estoque_inicial / df.demanda_base_semanal.clip(lower=1e-9)
    ordem = [df.preco_atual.idxmax(), df.demanda_base_semanal.idxmax(), cob.idxmax()]
    escolhidos = []
    for cand in ordem:
        if cand in escolhidos:
            restantes = [i for i in cob.sort_values().index if i not in escolhidos]
            cand = restantes[0] if restantes else cand
        if cand not in escolhidos:
            escolhidos.append(cand)
    return [df.sku[i] for i in escolhidos]


def horizonte_sem_memo(pmf, limite_chamadas=2e6):
    """Maior horizonte T' cuja árvore SEM memo cabe em ~limite_chamadas chamadas."""
    ramos = sum(int((pmf[0, k] > 0).sum()) for k in range(pmf.shape[1]))
    T = 1
    while T < pmf.shape[0] and ramos ** (T + 1) <= limite_chamadas:
        T += 1
    return T


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_fluxo(saida, cfg, args):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    caixas = [
        (0.2, "Monte Carlo\n(construção)",
         f"{args.cenarios_construcao:,} cenários · seed {cfg['seed_base']}\ndemanda, concorrente,\ncusto logístico".replace(",", "."), "#dbe9f6"),
        (3.4, "Programação\nDinâmica", "estado (semana, estoque)\nBellman bottom-up\nπ_t(s) e V_t(s)", "#e5f3e0"),
        (6.6, "Greedy\nsemanal", "destaque · mídia · frete\nΔ valor (via V da DP) / R$\norçamento semanal", "#fdf0d5"),
        (9.8, "Monte Carlo\n(validação)",
         f"{args.cenarios:,} cenários NOVOS\nseed {cfg['seed_final']} · P(pagar)\ncaixa, P{args.percentil_risco:g}, falhas".replace(",", "."), "#f6dede"),
    ]
    for x, titulo, sub, cor in caixas:
        ax.add_patch(FancyBboxPatch((x, 0.5), 2.9, 2.6, boxstyle="round,pad=0.05", fc=cor, ec="#444"))
        ax.text(x + 1.45, 2.5, titulo, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + 1.45, 1.25, sub, ha="center", va="center", fontsize=8.5)
    rot = ["PMF da demanda\npor (SKU, semana, preço)", "plano de preços +\nvalor marginal do estoque",
           "plano completo\n(preços + promoções)"]
    for i, x in enumerate([3.1, 6.3, 9.5]):
        ax.annotate("", xy=(x + 0.3, 1.8), xytext=(x, 1.8), arrowprops=dict(arrowstyle="-|>", lw=2))
        ax.text(x + 0.15, 3.35, rot[i], ha="center", fontsize=7.5, color="#333")
    ax.annotate("calibra ρ (peso do caixa no tempo) nos cenários de construção",
                xy=(4.8, 0.45), xytext=(1.6, 0.05), fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#777", connectionstyle="arc3,rad=-0.2"))
    fig.tight_layout(); fig.savefig(saida / "fig0_fluxo.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Fluxo principal
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    rapido, graficos = args.modo == "rapido", not args.sem_graficos
    saida = Path(args.saida) if args.saida else BASE / "saida"
    saida.mkdir(parents=True, exist_ok=True)

    pc, pk = caminhos(args.dados, args.grupo)
    cfg_path, cat_path = args.config or pc, args.catalogo or pk
    cfg, df = carregar(config_path=cfg_path, catalogo_path=cat_path)
    seeds = resolver_seeds(cfg, args)
    cfg, df = aplicar_cenario(cfg, df, args.meta_fator, args.logistica_fator, args.orcamento_fator,
                              args.remover_sku, args.concorrente_fator, args.incerteza_fator,
                              args.caixa_inicial)
    T, J = cfg["numero_semanas"], len(df)
    divs = semanas_dividas(cfg)
    sb, sf = cfg["seed_base"], cfg["seed_final"]
    seed_sim, seed_greedy = sf + 7, sb + 2          # sorteio das vendas na simulação / greedy
    pr = args.percentil_risco
    rel = {}
    if graficos:
        figura_fluxo(saida, cfg, args)

    # ======================= (A) MONTE CARLO — construção =======================
    (cen, t_c1, m_c1) = medir(gerar_cenarios, df, cfg, args.cenarios_construcao, sb)
    (est, t_c2, m_c2) = medir(mc.estimar, df, cfg, cen)
    est["resumo"].to_csv(saida / "01_mc_demanda_caixa_por_preco.csv", index=False)
    est["agregado"].to_csv(saida / "01b_mc_caixa_horizonte_por_nivel_fixo.csv", index=False)

    # ============== calibração de ρ nos cenários de CONSTRUÇÃO ===================
    calib, dps = [], {}
    for rho in args.grade_rho:
        (dp, t_dp, m_dp) = medir(pdm.resolver_catalogo, df, est, rho)
        dps[rho] = (dp, t_dp, m_dp)
        r = mc.resumir(mc.simular(df, cfg, cen, dict(tipo="fechado", PI=dp["PI"]), seed=seed_sim),
                       cfg, f"DP ρ={rho}", pr)
        r["rho"] = rho
        calib.append(r)
    tab_calib = pd.DataFrame(calib)
    tab_calib.to_csv(saida / "02_calibracao_rho_construcao.csv", index=False)
    rho = float(tab_calib.sort_values(["p_todas", "caixa_final_medio"], ascending=False).iloc[0]["rho"])
    dp, t_dp, m_dp = dps[rho]
    rel["rho_escolhido"] = rho

    # ======================= (B) DP — trajetória e reuso ========================
    linhas = []
    for j in range(J):
        for c in pdm.reconstruir(est["pmfs"][j], dp["PI"][j], int(df.estoque_inicial[j]), est["niveis"]):
            linhas.append(dict(sku=df.sku[j], **{k: v for k, v in c.items() if k != "idx"}))
    traj_dp = pd.DataFrame(linhas)
    traj_dp.to_csv(saida / "03_dp_trajetoria_otima.csv", index=False)
    tab_precos = traj_dp.pivot(index="sku", columns="semana", values="nivel")
    tab_precos.to_csv(saida / "03b_dp_precos_por_semana.csv")

    representativos = escolher_representativos(df)
    j1 = int(np.where(df.sku == representativos[0])[0][0])        # maior ticket
    S1 = int(df.estoque_inicial[j1])
    t0 = time.perf_counter()
    v_memo, c_memo = pdm.resolver_top_down(est["pmfs"][j1], dp["r"][j1], S1, memo=True)
    t_memo = time.perf_counter() - t0
    assert abs(v_memo - dp["V"][j1][0, S1]) < 1e-6 * max(1, abs(v_memo)), "Top-Down e Bottom-Up devem coincidir"
    Tc = horizonte_sem_memo(est["pmfs"][j1])
    t0 = time.perf_counter()
    v3m, c3m = pdm.resolver_top_down(est["pmfs"][j1], dp["r"][j1], S1, memo=True, T_lim=Tc)
    t3m = time.perf_counter() - t0
    t0 = time.perf_counter()
    v3n, c3n = pdm.resolver_top_down(est["pmfs"][j1], dp["r"][j1], S1, memo=False, T_lim=Tc)
    t3n = time.perf_counter() - t0
    assert abs(v3m - v3n) < 1e-6 * max(1, abs(v3m))
    reuso = pd.DataFrame([
        dict(caso=f"Top-Down COM memo, {T} semanas ({df.sku[j1]})", chamadas=c_memo["chamadas"],
             subproblemas=c_memo["subproblemas_distintos"], acertos_cache=c_memo["acertos_cache"], tempo_s=t_memo),
        dict(caso=f"Top-Down COM memo, {Tc} semanas", chamadas=c3m["chamadas"],
             subproblemas=c3m["subproblemas_distintos"], acertos_cache=c3m["acertos_cache"], tempo_s=t3m),
        dict(caso=f"Top-Down SEM memo, {Tc} semanas", chamadas=c3n["chamadas"],
             subproblemas=np.nan, acertos_cache=0, tempo_s=t3n),
        dict(caso=f"Bottom-Up, catálogo ({J} SKUs)", chamadas=np.nan, subproblemas=dp["info"]["estados"],
             acertos_cache=dp["info"]["leituras"], tempo_s=t_dp),
    ])
    reuso.to_csv(saida / "04_dp_reuso_subproblemas.csv", index=False)

    # ======================= (C) GREEDY sobre o plano da DP =====================
    (g, t_g, m_g) = medir(gr.planejar, df, cfg, est, dp, seed_greedy)
    promo, aloc, traj_g, uso = g
    aloc.to_csv(saida / "05_greedy_alocacoes.csv", index=False)
    uso.to_csv(saida / "05b_greedy_uso_orcamento.csv", index=False)
    traj_g.to_csv(saida / "05c_plano_completo_semanal.csv", index=False)
    promo_vc, aloc_vc, _, _ = gr.planejar(df, cfg, est, dp, seed_greedy, min_razao=1e-9)
    aloc_vc.to_csv(saida / "05d_greedy_alocacoes_orcamento_carimbado.csv", index=False)

    # ==================== (A) MONTE CARLO — validação (novos) ====================
    (val, t_v0, m_v0) = medir(gerar_cenarios, df, cfg, args.cenarios, sf)
    idx_aberto = np.array([[int(np.where(np.isclose(est["niveis"],
                                                   traj_dp[traj_dp.sku == s].nivel.iat[t]))[0][0])
                            for s in df.sku] for t in range(T)])
    politicas = {f"Fixo {m:.2f}": dict(tipo="fixo", IDX=np.full((T, J), k))
                 for k, m in enumerate(est["niveis"])}
    politicas["DP malha aberta (trajetória fixa)"] = dict(tipo="fixo", IDX=idx_aberto)
    politicas["DP (política π_t(s))"] = dict(tipo="fechado", PI=dp["PI"])
    politicas["DP + Greedy (orç. carimbado)"] = dict(tipo="fechado", PI=dp["PI"], **promo_vc)
    politicas["DP + Greedy (FINAL)"] = dict(tipo="fechado", PI=dp["PI"], **promo)
    sims, resumos = {}, []
    for nome, pl in politicas.items():
        (sim, t_v, m_v) = medir(mc.simular, df, cfg, val, pl, seed_sim)
        sims[nome] = sim
        r = mc.resumir(sim, cfg, nome, pr)
        r["tempo_simulacao_s"] = t_v
        resumos.append(r)
    tab_val = pd.DataFrame(resumos)
    tab_val.to_csv(saida / "06_validacao_politicas.csv", index=False)
    final = sims["DP + Greedy (FINAL)"]
    t_valid = float(tab_val.set_index("politica").loc["DP + Greedy (FINAL)", "tempo_simulacao_s"]) + t_v0

    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    nivel_atual = f"Fixo {est['niveis'][int(np.argmin(np.abs(est['niveis'] - 1.0)))]:.2f}"
    nivel_max = f"Fixo {est['niveis'].max():.2f}"
    comparadas = list(dict.fromkeys([nivel_atual, nivel_max, "DP (política π_t(s))", "DP + Greedy (FINAL)"]))
    dist = pd.DataFrame({n: np.percentile(sims[n]["caixa_final"], pct) for n in comparadas},
                        index=[f"P{p}" for p in pct])
    dist.to_csv(saida / "07_distribuicao_caixa_final.csv")

    falha = ~final["pago"].all(axis=1)
    drivers = pd.DataFrame(dict(
        choque_demanda_medio=val["eps"].mean(axis=(1, 2)),
        concorrente_rel_final=val["comp"][:, -1].mean(axis=1),
        custo_logistico_medio=val["logm"].mean(axis=1),
        caixa_final=final["caixa_final"], falhou=falha))
    perfil = drivers.groupby("falhou").mean().T
    perfil.to_csv(saida / "08_perfil_cenarios_falha.csv")
    detalhes = drivers[drivers.falhou].copy()
    for k, w in enumerate(final["semanas_div"]):
        detalhes[f"caixa_antes_s{w}"] = final["caixa_antes_div"][falha, k]
    detalhes.to_csv(saida / "08b_cenarios_falha.csv")

    tab_est = None
    if not rapido:
        estresse = []
        for i, esc in enumerate([1.0, 2.0, 3.0]):
            v2 = gerar_cenarios(df, cfg, args.cenarios, sf + 100 + i, escala_sigma=esc)
            for nome in [nivel_atual, "DP (política π_t(s))", "DP + Greedy (FINAL)"]:
                r = mc.resumir(mc.simular(df, cfg, v2, politicas[nome], seed_sim), cfg, nome, pr)
                r["escala_sigma"] = esc
                estresse.append(r)
        tab_est = pd.DataFrame(estresse)
        tab_est.to_csv(saida / "09_estresse_incerteza.csv", index=False)

    # ============================ FIGURAS ======================================
    if graficos:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        cores = dict(zip([nivel_atual, nivel_max, "DP + Greedy (FINAL)"], ["#9aa5b1", "#e0a458", "#2b6a8f"]))
        for nome, cor in cores.items():
            x = sims[nome]["caixa_final"] / 1e3
            ax.hist(x, bins=80, alpha=.55, color=cor, label=nome)
            ax.axvline(np.percentile(x, pr), color=cor, ls="--", lw=1.2)
        ax.axvline(0, color="crimson", lw=1.5, label="caixa = 0 após pagar as dívidas")
        ax.set_xlabel(f"Caixa final após as dívidas (R$ mil) — tracejado = P{pr:g}")
        ax.set_ylabel(f"cenários (de {args.cenarios:,})".replace(",", ".")); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title("Distribuição do caixa final — validação Monte Carlo (cenários novos)")
        fig.tight_layout(); fig.savefig(saida / "fig1_distribuicao_caixa_final.png", dpi=150); plt.close(fig)

        fig, axes = plt.subplots(2, len(representativos), figsize=(4.7 * len(representativos), 6.6),
                                 sharex=True, squeeze=False)
        sem = np.arange(1, T + 1)
        for c, sku in enumerate(representativos):
            j = int(np.where(df.sku == sku)[0][0])
            niv = est["niveis"][final["hist_nivel"][:, :, j]]
            ax = axes[0, c]
            ax.fill_between(sem, np.percentile(niv, 10, axis=0), np.percentile(niv, 90, axis=0),
                            alpha=.25, color="#2b6a8f", label="P10–P90 (validação)")
            ax.plot(sem, np.median(niv, axis=0), "o-", color="#2b6a8f", label="mediana (validação)")
            tp = traj_dp[traj_dp.sku == sku]
            ax.plot(tp.semana, tp.nivel, "s--", color="#e07a5f", ms=4, label="caminho ótimo reconstruído")
            ax.set_title(f"{sku} · {df.categoria[j]} · preço")
            ax.set_ylim(est["niveis"].min() - 0.02, est["niveis"].max() + 0.02); ax.grid(alpha=.3)
            if c == 0:
                ax.set_ylabel("multiplicador de preço"); ax.legend(fontsize=7)
            est_j = final["hist_estoque"][:, :, j]
            ax = axes[1, c]
            s_ax = np.arange(0, T + 1)
            ax.fill_between(s_ax, np.percentile(est_j, 10, axis=0), np.percentile(est_j, 90, axis=0),
                            alpha=.25, color="#4a7c59")
            ax.plot(s_ax, est_j.mean(axis=0), "o-", color="#4a7c59", ms=4)
            ax.set_xlabel("semana"); ax.grid(alpha=.3)
            if c == 0:
                ax.set_ylabel("estoque (un.)")
        fig.suptitle("Trajetória semanal de preço e estoque — plano final (DP + Greedy)")
        fig.tight_layout(); fig.savefig(saida / "fig2_trajetorias_preco_estoque.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.4))
        hc = final["hist_caixa"] / 1e3
        for lo, hi, a in [(1, 99, .15), (5, 95, .25), (25, 75, .35)]:
            ax.fill_between(sem, np.percentile(hc, lo, axis=0), np.percentile(hc, hi, axis=0),
                            color="#2b6a8f", alpha=a, label=f"P{lo}–P{hi}")
        ax.plot(sem, np.median(hc, axis=0), color="#1d3f57", lw=2, label="mediana")
        ax.axhline(0, color="crimson", lw=1)
        topo = np.percentile(hc, 99, axis=0).max()
        for w_, v_ in divs.items():
            ax.axvline(w_, color="gray", ls=":")
            ax.text(w_, topo, f" paga R$ {v_ / 1e3:.0f} mil", fontsize=8)
        ax.set_xlabel("semana"); ax.set_ylabel("caixa após pagamentos (R$ mil)")
        ax.set_title("Caixa semanal sob incerteza — plano final"); ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(saida / "fig3_caixa_semanal.png", dpi=150); plt.close(fig)

        mapas = representativos[1:] or representativos
        fig, axes = plt.subplots(1, len(mapas), figsize=(6 * len(mapas), 4), squeeze=False)
        for ax, sku in zip(axes[0], mapas):
            j = int(np.where(df.sku == sku)[0][0])
            S = int(df.estoque_inicial[j])
            mapa = est["niveis"][dp["PI"][j, :, :S + 1]]
            im = ax.imshow(mapa.T, aspect="auto", origin="lower", cmap="RdYlGn",
                           extent=[0.5, T + 0.5, 0, max(S, 1)],
                           vmin=est["niveis"].min(), vmax=est["niveis"].max())
            tp = traj_dp[traj_dp.sku == sku]
            ax.plot(tp.semana, tp.estoque_inicio, "k.-", label="estoque esperado no caminho ótimo")
            ax.set_title(f"π_t(s) — {sku}"); ax.set_xlabel("semana"); ax.set_ylabel("estoque remanescente")
            ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=axes[0].tolist(), label="multiplicador de preço ótimo")
        fig.savefig(saida / "fig4_mapa_politica_dp.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        nomes = list(dict.fromkeys([f"Fixo {est['niveis'][max(0, int(np.argmin(np.abs(est['niveis'] - 0.88))))]:.2f}",
                                    nivel_atual, nivel_max, "DP malha aberta (trajetória fixa)",
                                    "DP + Greedy (FINAL)"]))
        sel = tab_val.set_index("politica").loc[nomes]
        cols = [f"p_semana_{w}" for w in divs] + ["p_todas"]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(sel)); w_ = 0.8 / len(cols)
        for i, col in enumerate(cols):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w_, sel[col] * 100, w_,
                   label=col.replace("p_", "").replace("_", " "))
        ax.axhline(100 * (1 - cfg["limite_risco_falha"]), color="crimson", ls="--", lw=1,
                   label=f"mínimo {100 * (1 - cfg['limite_risco_falha']):.0f}%")
        ax.set_xticks(x, sel.index, fontsize=8); ax.set_ylabel("% dos cenários"); ax.legend(fontsize=8)
        ax.set_title("Probabilidade de pagamento no prazo — validação"); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(saida / "fig5_prob_pagamento.png", dpi=150); plt.close(fig)

    # ====================== ESCALABILIDADE (3 tamanhos) =========================
    tab_esc = None
    if not rapido:
        escala = []
        n_peq = max(2, J // 3)
        for nome, dfx in [(f"pequeno ({n_peq} SKUs)", df.iloc[:n_peq].reset_index(drop=True)),
                          (f"médio ({J} SKUs)", df),
                          (f"grande ({3 * J} SKUs)", replicar_catalogo(df, 3))]:
            (cx, a1, b1) = medir(gerar_cenarios, dfx, cfg, args.cenarios_construcao, sb)
            (ex, a2, b2) = medir(mc.estimar, dfx, cfg, cx)
            (dx, a3, b3) = medir(pdm.resolver_catalogo, dfx, ex, rho)
            (gx, a4, b4) = medir(gr.planejar, dfx, cfg, ex, dx, seed_greedy)
            (vx, a5, b5) = medir(gerar_cenarios, dfx, cfg, args.cenarios, sf)
            (_, a6, b6) = medir(mc.simular, dfx, cfg, vx, dict(tipo="fechado", PI=dx["PI"], **gx[0]), seed_sim)
            escala.append(dict(entrada=nome, J=len(dfx), soma_estoques=int(dfx.estoque_inicial.sum()),
                               estoque_max=int(dfx.estoque_inicial.max()), estados_dp=dx["info"]["estados"],
                               mc_constr_s=a1 + a2, mc_constr_mib=max(b1, b2), dp_s=a3, dp_mib=b3,
                               greedy_s=a4, greedy_mib=b4, mc_valid_s=a5 + a6, mc_valid_mib=max(b5, b6)))
            del cx, ex, dx, gx, vx
        for fator in [1, 2, 4]:
            dfs = df.copy()
            dfs["estoque_inicial"] = dfs["estoque_inicial"] * fator
            ests = mc.estimar(dfs, cfg, gerar_cenarios(dfs, cfg, args.cenarios_construcao, sb))
            (dps_, t, m) = medir(pdm.resolver_catalogo, dfs, ests, rho)
            escala.append(dict(entrada=f"DP com estoque ×{fator} ({J} SKUs)", J=J,
                               soma_estoques=int(dfs.estoque_inicial.sum()),
                               estoque_max=int(dfs.estoque_inicial.max()),
                               estados_dp=dps_["info"]["estados"], dp_s=t, dp_mib=m))
        tab_esc = pd.DataFrame(escala)
        tab_esc.to_csv(saida / "10_escalabilidade.csv", index=False)
        if graficos:
            e3 = tab_esc.iloc[:3]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for col, lab in [("mc_constr", "MC construção"), ("dp", "DP"), ("greedy", "Greedy"),
                             ("mc_valid", "MC validação")]:
                axes[0].plot(e3.J, e3[f"{col}_s"], "o-", label=lab)
                axes[1].plot(e3.J, e3[f"{col}_mib"], "o-", label=lab)
            for ax, yl in zip(axes, ["tempo (s)", "pico de memória (MiB)"]):
                ax.set_xlabel("nº de SKUs (J)"); ax.set_ylabel(yl); ax.grid(alpha=.3); ax.legend(fontsize=8)
                ax.set_xticks(e3.J)
            fig.suptitle("Escalabilidade: entradas pequena, média e grande (perf_counter + tracemalloc)")
            fig.tight_layout(); fig.savefig(saida / "fig6_escalabilidade.png", dpi=150); plt.close(fig)

    # ============================ RESULTADO =====================================
    rf = tab_val.set_index("politica").loc["DP + Greedy (FINAL)"]
    rel.update(dict(
        n_construcao=args.cenarios_construcao, n_validacao=args.cenarios,
        p_pagamento={**{f"semana_{w}": float(rf[f"p_semana_{w}"]) for w in divs}, "todas": float(rf.p_todas)},
        caixa_final_esperado=rf.caixa_final_medio, percentil_risco=pr,
        caixa_final_percentil_risco=rf.caixa_final_p_risco,
        margem_contrib_pct=rf.margem_contrib_pct, estoque_final_medio=rf.estoque_final_medio,
        estoque_inicial=int(df.estoque_inicial.sum()), cenarios_falha=int(rf.cenarios_falha),
        atende_limite_risco=bool(rf.atende_limite_risco), skus_representativos=representativos,
        greedy=dict(alocacoes=len(aloc), gasto_planejado=float(uso.usado.sum()),
                    orcamento_total=float(uso.orcamento.sum())),
        dp_info=dp["info"],
        tempos=dict(mc_construcao_s=t_c1 + t_c2, mc_construcao_mib=max(m_c1, m_c2), dp_s=t_dp, dp_mib=m_dp,
                    greedy_s=t_g, greedy_mib=m_g, mc_validacao_s=t_valid),
    ))
    (saida / "resultado.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    avisos = []
    if args.cenarios < MIN_VALIDACAO:
        avisos.append(f"cenários de validação ({args.cenarios}) abaixo do mínimo exigido ({MIN_VALIDACAO})")
    if args.cenarios_construcao < MIN_CONSTRUCAO:
        avisos.append(f"cenários de construção ({args.cenarios_construcao}) abaixo do mínimo ({MIN_CONSTRUCAO})")
    gerar_manifesto(
        saida / "manifesto_dados.json", "questao_02", args.grupo, [cfg_path, cat_path], args.dados,
        seeds=dict(seeds, cenarios_construcao=sb, poisson_construcao=sb + 1, greedy=seed_greedy,
                   cenarios_validacao=sf, vendas_simulacao=seed_sim,
                   cenarios_estresse=[sf + 100 + i for i in range(3)]),
        n_skus=J, skus_ativos=df.sku.tolist(),
        metas=dict(caixa_inicial=cfg["caixa_inicial"], dividas=cfg["dividas"],
                   orcamento_promocional_semanal=cfg["orcamento_promocional_semanal"],
                   limite_risco_falha=cfg["limite_risco_falha"]),
        parametros=dict(modo=args.modo, cenarios_validacao=args.cenarios,
                        cenarios_construcao=args.cenarios_construcao, grade_rho=args.grade_rho,
                        rho_escolhido=rho, percentil_risco=pr, meta_fator=args.meta_fator,
                        logistica_fator=args.logistica_fator, orcamento_fator=args.orcamento_fator,
                        concorrente_fator=args.concorrente_fator, incerteza_fator=args.incerteza_fator,
                        caixa_inicial=cfg["caixa_inicial"], remover_sku=args.remover_sku, avisos=avisos),
        comando=["python", "questao_02/main.py", *(argv if argv is not None else sys.argv[1:])])
    return dict(rel=rel, calib=tab_calib, tab_val=tab_val, dist=dist, perfil=perfil, estresse=tab_est,
                reuso=reuso, aloc=aloc, uso=uso, precos=tab_precos, escala=tab_esc,
                resumo_mc=est["agregado"], saida=saida, avisos=avisos)


def imprimir(out):
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for k in ["resumo_mc", "calib", "precos", "reuso", "aloc", "uso", "tab_val", "dist", "perfil",
              "estresse", "escala"]:
        if out[k] is not None:
            print(f"\n=== {k} ===\n", out[k].round(3).to_string())
    print("\n=== RESULTADO ===\n", json.dumps(out["rel"], indent=2, ensure_ascii=False, default=float))
    for a in out["avisos"]:
        print("AVISO:", a)
    print(f"\nSaídas em: {out['saida']}")


if __name__ == "__main__":
    imprimir(main())
