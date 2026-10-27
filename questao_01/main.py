"""
QUESTÃO 1 — Operação Caixa 90

Nesta questão utilizamos 3 algoritmos, integrados em um único fluxo:
  (A) brute_force.py       -> ótimo exato de um subconjunto de SKUs (benchmark)
  (B) genetic_algorithm.py -> validado contra o BF no MESMO subconjunto; depois gera a
                              política de preços do catálogo completo
  (C) greedy.py            -> recebe a política do GA e distribui o orçamento
                              promocional por benefício marginal por real
modelo.py contém o modelo econômico e a função de avaliação usados pelos três.

Exemplos:
  python questao_01/main.py                                   # instância oficial (seed do config)
  python questao_01/main.py --seed 14 --modo completo
  python questao_01/main.py --meta-fator 1.10 --logistica-fator 1.15 --orcamento-fator 0.75
  python questao_01/main.py --remover-sku Q1-005 --ga-pop 60 --ga-mutacao 0.05
  python questao_01/main.py --help
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
import numpy as np
import pandas as pd

from comum import gerar_manifesto, parser_base, resolver_seeds
from modelo import (Instancia, aplicar_cenario, avaliar_ingenuo, avaliar_vetorizado, carregar,
                    gerar_precos_permitidos, medir)
from brute_force import brute_force_ingenuo, brute_force_vetorizado
from genetic_algorithm import ConfigGA, ga
from greedy import greedy_incentivos, greedy_ingenuo, opcoes_incentivo
from escalabilidade import medir_escalabilidade

LIMITE_COMBINACOES_BF = 200_000   # acima disso o BF deixa de ser "pequeno" (ver BIG_O.md)
TAMANHO_SUBCONJUNTO = 6


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = parser_base("Questão 1 — Operação Caixa 90 (Brute Force -> GA -> Greedy)")
    g = p.add_argument_group("Algoritmo Genético (configuração A = principal)")
    g.add_argument("--ga-pop", type=int, default=40, help="tamanho da população (config A)")
    g.add_argument("--ga-mutacao", type=float, default=None,
                   help="prob. de mutação por gene (padrão 1/n)")
    g.add_argument("--ga-torneio", type=int, default=2, help="tamanho do torneio (config A)")
    g.add_argument("--ga-elitismo", type=int, default=1, help="nº de elites (config A)")
    g.add_argument("--ga-geracoes", type=int, default=500, help="máximo de gerações")
    g.add_argument("--ga-paciencia", type=int, default=80, help="gerações sem melhora p/ parar")
    g.add_argument("--ga-execucoes", type=int, default=None,
                   help="execuções (seeds) por configuração (padrão: 5 completo / 2 rápido)")
    b = p.add_argument_group("Brute Force")
    b.add_argument("--subconjunto", nargs="+", default=None, metavar="SKU",
                   help="SKUs do subconjunto do BF (padrão: escolha automática)")
    b.add_argument("--tamanho-subconjunto", type=int, default=TAMANHO_SUBCONJUNTO)
    return p.parse_args(argv)


def configs_ga(args, n: int):
    """Config A vem da CLI; B é a variante de maior pressão seletiva
    (3x população, torneio +2, mutação x1,5, elitismo +1)."""
    pm_a = args.ga_mutacao if args.ga_mutacao is not None else 1.0 / n
    a = ConfigGA(f"A: pop={args.ga_pop}, torneio={args.ga_torneio}, pm={pm_a:.3f}",
                 populacao=args.ga_pop, torneio_k=args.ga_torneio, p_mutacao=pm_a,
                 elitismo=args.ga_elitismo, geracoes_max=args.ga_geracoes, paciencia=args.ga_paciencia)
    pop_b, pm_b = 3 * args.ga_pop, min(1.0, 1.5 * pm_a)
    b = ConfigGA(f"B: pop={pop_b}, torneio={args.ga_torneio + 2}, pm={pm_b:.3f}",
                 populacao=pop_b, torneio_k=args.ga_torneio + 2, p_mutacao=pm_b,
                 elitismo=args.ga_elitismo + 1, geracoes_max=args.ga_geracoes, paciencia=args.ga_paciencia)
    return [a, b]


# ---------------------------------------------------------------------------
# Escolha automática do subconjunto do Brute Force
# ---------------------------------------------------------------------------
def escolher_subconjunto(df, permitidos, inativos, k=TAMANHO_SUBCONJUNTO,
                         limite=LIMITE_COMBINACOES_BF):
    """Regra (independe da instância): até 2 SKUs ativos por categoria, priorizando as
    categorias e SKUs com mais níveis de preço permitidos (maior espaço de busca), para
    que a canibalização dentro da categoria atue. Para quando o produto dos níveis
    atingiria o limite de combinações. Empates: ordem alfabética de categoria e SKU."""
    ativos = df[~df.sku.isin(inativos)].copy()
    ativos["niveis"] = [len(permitidos[i]) for i in ativos.index]
    ativos = ativos.sort_values(["niveis", "sku"], ascending=[False, True])
    por_cat = {c: g.head(2) for c, g in ativos.groupby("categoria")}
    ordem = sorted(por_cat, key=lambda c: (-np.log(por_cat[c].niveis).sum(), c))
    escolhidos, combs = [], 1
    for c in ordem:
        for _, r in por_cat[c].iterrows():
            if len(escolhidos) < k and combs * r.niveis <= limite:
                escolhidos.append(r.sku); combs *= r.niveis
    for _, r in ativos.iterrows():                       # completa se faltar
        if len(escolhidos) < k and r.sku not in escolhidos and combs * r.niveis <= limite:
            escolhidos.append(r.sku); combs *= r.niveis
    return escolhidos


# ---------------------------------------------------------------------------
# Fluxo principal
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    rapido = args.modo == "rapido"
    graficos = not args.sem_graficos
    saida = Path(args.saida) if args.saida else BASE / "saida"
    saida.mkdir(parents=True, exist_ok=True)

    cfg_path = args.config or Path(args.dados) / f"q1_config_{args.grupo}.json"
    cat_path = args.catalogo or Path(args.dados) / f"q1_catalogo_{args.grupo}.csv"
    cfg, df = carregar(cfg_path, cat_path)
    seeds = resolver_seeds(cfg, args)
    cfg, df = aplicar_cenario(cfg, df, args.meta_fator, args.logistica_fator,
                              args.orcamento_fator, args.remover_sku)
    n_exec = args.ga_execucoes or (2 if rapido else 5)
    n_exec_sub = 3 if rapido else 10

    mmin = cfg["margem_minima_percentual"]
    permitidos, sinal = gerar_precos_permitidos(df, cfg)
    df_perm = df[["sku", "categoria", "idade_estoque_dias"]].copy()
    df_perm["precos_permitidos"] = [", ".join(f"{m:.2f}" for m in p) for p in permitidos]
    df_perm["n_niveis"] = [len(p) for p in permitidos]
    df_perm["fora_da_campanha"] = df_perm.sku.isin(sinal)
    df_perm.to_csv(saida / "00_precos_permitidos.csv", index=False)
    rel = {"sinalizados_sem_preco_viavel": sinal}

    # ---------------- (A) BRUTE FORCE no subconjunto -----------------------
    sub_skus = args.subconjunto or escolher_subconjunto(df, permitidos, sinal, args.tamanho_subconjunto)
    if not sub_skus:
        raise ValueError("nenhum SKU ativo para o subconjunto do Brute Force")
    faltando = sorted(set(sub_skus) - set(df.sku))
    if faltando:
        raise ValueError(f"--subconjunto com SKUs inexistentes: {faltando}")
    idx_sub = [int(df.index[df.sku == s][0]) for s in sub_skus]
    inst_sub = Instancia.criar(df.loc[idx_sub], [permitidos[i] for i in idx_sub], sinal)
    n_comb_prev = int(np.prod([len(p) for p in inst_sub.permitidos]))
    if n_comb_prev > 5 * LIMITE_COMBINACOES_BF:
        raise ValueError(f"subconjunto com {n_comb_prev} combinações: grande demais para Brute Force")

    (bf_ing, t_ing, mem_ing) = medir(brute_force_ingenuo, inst_sub)
    (bf_vet, t_vet, mem_vet) = medir(brute_force_vetorizado, inst_sub)
    f_bf, pol_bf, n_comb = bf_vet
    assert abs(bf_ing[0] - f_bf) < 1e-4, "BF ingênuo e vetorizado devem coincidir"

    # ---------------- (B1) GA no MESMO subconjunto --------------------------
    linhas_gap = []
    for s in range(n_exec_sub):
        seed = cfg["seed_base"] + s
        (r, t, mem) = medir(ga, inst_sub, ConfigGA("sub", populacao=30, geracoes_max=150,
                                                    paciencia=40, seed=seed))
        linhas_gap.append(dict(execucao=s + 1, seed=seed, fitness_ga=r["fitness"],
                               otimo_bf=f_bf, gap_pct=100 * (f_bf - r["fitness"]) / max(abs(f_bf), 1e-9),
                               avaliacoes=r["avaliacoes"],
                               frac_espaco=r["avaliacoes"] / n_comb, tempo_s=t, mem_kib=mem,
                               politica=", ".join(f"{m:.2f}" for m in r["politica"])))
    tab_gap = pd.DataFrame(linhas_gap)
    tab_gap.to_csv(saida / "01_bruteforce_vs_ga.csv", index=False)
    tab_sub = pd.DataFrame(dict(sku=sub_skus, categoria=inst_sub.df.categoria,
                                precos_permitidos=[", ".join(f"{m:.2f}" for m in p)
                                                   for p in inst_sub.permitidos],
                                preco_bf=pol_bf,
                                preco_ga_exec1=[float(x) for x in linhas_gap[0]["politica"].split(", ")]))
    tab_sub.to_csv(saida / "01b_politicas_subconjunto.csv", index=False)

    # ---------------- (B2) GA no catálogo completo — 2 configurações -------
    inst = Instancia.criar(df, permitidos, sinal)
    configs = configs_ga(args, inst.n)
    comp, historicos = [], {}
    for c in configs:
        res_runs = []
        for s in range(n_exec):
            c.seed = cfg["seed_final"] + s
            res_runs.append(medir(ga, inst, c))
        fits = [r["fitness"] for r, _, _ in res_runs]
        comp.append(dict(config=c.nome, populacao=c.populacao, torneio=c.torneio_k,
                         p_mutacao=c.p_mutacao, elitismo=c.elitismo, execucoes=n_exec,
                         melhor=max(fits), media=np.mean(fits), desvio=np.std(fits),
                         geracoes_media=np.mean([r["geracoes"] for r, _, _ in res_runs]),
                         avaliacoes_media=np.mean([r["avaliacoes"] for r, _, _ in res_runs]),
                         tempo_medio_s=np.mean([t for _, t, _ in res_runs]),
                         mem_pico_kib=np.mean([m for _, _, m in res_runs])))
        historicos[c.nome] = res_runs
    tab_comp = pd.DataFrame(comp)
    tab_comp.to_csv(saida / "02_comparacao_configs_ga.csv", index=False)

    todas = [(r["fitness"], r["politica"], nome) for nome, runs in historicos.items()
             for r, _, _ in runs]
    f_ga, pol_ga, cfg_venc = max(todas, key=lambda x: x[0])

    if graficos:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, (nome, runs) in zip(axes, historicos.items()):
            for k, (r, _, _) in enumerate(runs):
                ax.plot(r["hist_melhor"] / 1e3, lw=1.6, alpha=.85,
                        label=f"melhor (seed {cfg['seed_final'] + k})")
            ax.plot(runs[0][0]["hist_media"] / 1e3, "--", color="gray", lw=1, label="média da pop. (1ª seed)")
            ax.set_title(f"Config {nome}"); ax.set_xlabel("Geração"); ax.grid(alpha=.3)
        axes[0].set_ylabel("Fitness = caixa operacional (R$ mil)")
        axes[1].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"GA — fitness por geração (catálogo completo, {inst.n} SKUs)")
        fig.tight_layout(); fig.savefig(saida / "fig1_fitness_por_geracao.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.4))
        for nome, runs in historicos.items():
            r = max(runs, key=lambda x: x[0]["fitness"])[0]
            ax.plot(r["hist_melhor"] / 1e3, lw=2, label=f"{nome} — melhor execução")
        ax.axhline(cfg["meta_caixa_90d"] / 1e3, color="crimson", ls=":", label="meta de caixa 90d")
        ax.set_xlabel("Geração"); ax.set_ylabel("R$ mil"); ax.grid(alpha=.3)
        ax.set_title("Convergência: melhor fitness por geração (A vs B)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(saida / "fig2_convergencia_A_vs_B.png", dpi=150); plt.close(fig)

    # ---------------- (C) GREEDY sobre a política do GA ----------------------
    opcoes, base = opcoes_incentivo(inst, pol_ga, mmin)
    opcoes.to_csv(saida / "03a_opcoes_incentivo.csv", index=False)
    (gr, t_gr, mem_gr) = medir(greedy_incentivos, opcoes, cfg["orcamento_promocional"], 1.0)
    alocacao, passos, usado = gr
    gr_ing = greedy_ingenuo(opcoes, cfg["orcamento_promocional"], 1.0)
    assert abs(gr_ing[1] - usado) < 1e-6, "greedy ingênuo e com heap devem coincidir"
    passos.to_csv(saida / "03_greedy_passos.csv", index=False)
    aloc_vc, passos_vc, usado_vc = greedy_incentivos(opcoes, cfg["orcamento_promocional"], 1e-9)
    passos_vc.to_csv(saida / "03b_greedy_passos_verba_carimbada.csv", index=False)

    d = inst.df
    q0, p, cu = base["q"][0], base["p"][0], base["contrib_unit"][0]
    hold0 = base["hold"][0]

    def tabela_ad(aloc):
        linhas = []
        for j in range(inst.n):
            a = aloc.get(j, dict(incentivo=None, custo=0.0, liquido=0.0))
            q1 = a.get("q1", q0[j]) if a["incentivo"] else q0[j]
            ativo = d.ativo[j] == 1
            linhas.append(dict(
                sku=d.sku[j], categoria=d.categoria[j], preco_atual=d.preco_atual[j],
                mult_ga=pol_ga[j] if ativo else np.nan, preco_ga=p[j] if ativo else np.nan,
                margem_pct=100 * cu[j] / p[j] if ativo else np.nan,
                unid_antes=q0[j], caixa_antes=q0[j] * cu[j] - hold0[j],
                incentivo=a["incentivo"] or ("-" if ativo else "fora da campanha"),
                custo_incentivo=a["custo"],
                unid_depois=q1, caixa_depois=q0[j] * cu[j] - hold0[j] + a["liquido"],
                estoque=d.estoque[j], estoque_rem_depois=d.estoque[j] - q1))
        return pd.DataFrame(linhas)

    antes_depois = tabela_ad(alocacao)
    antes_depois.to_csv(saida / "04_antes_depois_greedy.csv", index=False)
    ad_vc = tabela_ad(aloc_vc)
    ad_vc.to_csv(saida / "04b_antes_depois_verba_carimbada.csv", index=False)

    base_atual = avaliar_vetorizado(inst, np.ones((1, inst.n)), detalhado=True)
    caixa_ga = antes_depois.caixa_antes.sum()
    caixa_final = antes_depois.caixa_depois.sum()
    receita_final = float(np.nansum(antes_depois.unid_depois * antes_depois.preco_ga))
    meta, divida, orc = cfg["meta_caixa_90d"], cfg["divida_total_90d"], cfg["orcamento_promocional"]
    final = dict(
        caixa_preco_atual=float(base_atual["caixa"][0]),
        caixa_politica_ga=caixa_ga,
        caixa_final_ga_greedy=caixa_final,
        meta_caixa=meta,
        meta_atingida=bool(caixa_final >= meta),
        folga_ou_insuficiencia=caixa_final - meta,
        divida_90d=divida,
        cobertura_divida_pct=100 * caixa_final / divida if divida > 0 else None,
        receita_bruta=receita_final,
        margem_contribuicao_media_pct=(100 * float(np.nansum(
            antes_depois.unid_depois * (antes_depois.margem_pct / 100) * antes_depois.preco_ga)
            - antes_depois.custo_incentivo.sum()) / receita_final) if receita_final > 0 else None,
        custo_armazenagem_estoque_rem=float(hold0.sum()),
        unidades_vendidas=antes_depois.unid_depois.sum(),
        estoque_inicial_un=int(d.estoque.sum()),
        estoque_remanescente_un=antes_depois.estoque_rem_depois.sum(),
        estoque_remanescente_custo=float((antes_depois.estoque_rem_depois * d.custo_unitario).sum()),
        orcamento_promocional=orc,
        orcamento_usado=usado,
        orcamento_usado_pct=100 * usado / orc if orc > 0 else 0.0,
        skus_com_desconto=int(((pol_ga < 1) & (d.ativo == 1)).sum()),
        skus_mantidos=int(((pol_ga == 1) & (d.ativo == 1)).sum()),
        skus_com_aumento=int(((pol_ga > 1) & (d.ativo == 1)).sum()),
        skus_fora_campanha=sinal,
        cenario_verba_carimbada=dict(
            orcamento_usado=usado_vc,
            caixa_liquido=float(ad_vc.caixa_depois.sum()),
            ganho_bruto=float(sum(a.get("ganho_bruto", 0) for a in aloc_vc.values())),
            unidades_extra=float(ad_vc.unid_depois.sum() - antes_depois.unid_antes.sum())),
        config_vencedora=cfg_venc,
    )

    ad = antes_depois[~antes_depois.incentivo.isin(["-", "fora da campanha"])].sort_values("caixa_depois")
    if graficos and len(ad):
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(ad) + 1.5))
        y = np.arange(len(ad))
        ax.barh(y - 0.2, ad.caixa_antes / 1e3, 0.4, label="antes (só preço GA)", color="#8aa6c1")
        ax.barh(y + 0.2, ad.caixa_depois / 1e3, 0.4, label="depois (GA + incentivo)", color="#2b6a8f")
        ax.set_yticks(y, [f"{s} · {i}" for s, i in zip(ad.sku, ad.incentivo)], fontsize=8)
        ax.set_xlabel("Caixa operacional (R$ mil)"); ax.legend(fontsize=8); ax.grid(axis="x", alpha=.3)
        ax.set_title("Greedy: SKUs que receberam incentivo")
        fig.tight_layout(); fig.savefig(saida / "fig3_greedy_antes_depois.png", dpi=150); plt.close(fig)

    # ---------------- otimização de código: antes/depois ---------------------
    otim = [dict(etapa=f"Brute Force ({inst_sub.n} SKUs, {n_comb} combinações)",
                 antes="loop Python + dicts por combinação", depois="numpy vetorizado em blocos",
                 tempo_antes_s=t_ing, tempo_depois_s=t_vet, speedup=t_ing / t_vet,
                 mem_antes_kib=mem_ing, mem_depois_kib=mem_vet)]
    rows = d.to_dict("records")
    P = np.random.default_rng(cfg["seed_base"]).integers(0, [len(x) for x in permitidos], size=(120, inst.n))
    Mpop = inst.tabela_m()[np.arange(inst.n), P]
    rep = 10 if rapido else 50
    t0 = time.perf_counter()
    for _ in range(rep):
        f1 = [avaliar_ingenuo(rows, Mpop[i], inst.max_desc) for i in range(len(Mpop))]
    t_pop_ing = (time.perf_counter() - t0) / rep
    t0 = time.perf_counter()
    for _ in range(rep):
        f2 = avaliar_vetorizado(inst, Mpop)
    t_pop_vet = (time.perf_counter() - t0) / rep
    assert np.allclose(f1, f2)
    otim.append(dict(etapa=f"Fitness de 1 geração do GA (120×{inst.n})",
                     antes="loop Python por indivíduo", depois="matriz numpy",
                     tempo_antes_s=t_pop_ing, tempo_depois_s=t_pop_vet,
                     speedup=t_pop_ing / t_pop_vet, mem_antes_kib=np.nan, mem_depois_kib=np.nan))
    otim = pd.DataFrame(otim)
    otim.to_csv(saida / "05_otimizacao_codigo.csv", index=False)

    # ---------------- escalabilidade: entradas pequena / média / grande -------
    tab_esc = None
    if not rapido:
        tab_esc = medir_escalabilidade(df, cfg, inst_sub, configs[0], saida, graficos)

    rel.update(dict(
        subconjunto=sub_skus, n_combinacoes_bf=n_comb, otimo_bf=f_bf,
        politica_bf=pol_bf.tolist(), tempo_bf_ingenuo_s=t_ing, tempo_bf_vet_s=t_vet,
        mem_bf_ingenuo_kib=mem_ing, mem_bf_vet_kib=mem_vet,
        gap_medio_pct=float(tab_gap.gap_pct.mean()), gap_max_pct=float(tab_gap.gap_pct.max()),
        execucoes_otimas=int((tab_gap.gap_pct < 1e-6).sum()),
        greedy_tempo_s=t_gr, greedy_mem_kib=mem_gr, final=final,
    ))
    (saida / "resultado.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    gerar_manifesto(
        saida / "manifesto_dados.json", "questao_01", args.grupo, [cfg_path, cat_path], args.dados,
        seeds=dict(seeds, ga_subconjunto=[cfg["seed_base"] + s for s in range(n_exec_sub)],
                   ga_catalogo=[cfg["seed_final"] + s for s in range(n_exec)],
                   precos_permitidos=cfg["seed_base"]),
        n_skus=len(df), skus_ativos=[s for s in df.sku if s not in sinal],
        metas=dict(meta_caixa_90d=meta, divida_total_90d=divida, orcamento_promocional=orc,
                   margem_minima_percentual=mmin),
        parametros=dict(modo=args.modo, meta_fator=args.meta_fator, logistica_fator=args.logistica_fator,
                        orcamento_fator=args.orcamento_fator, remover_sku=args.remover_sku,
                        ga=[dict(nome=c.nome, populacao=c.populacao, torneio=c.torneio_k,
                                 p_mutacao=c.p_mutacao, elitismo=c.elitismo,
                                 geracoes_max=c.geracoes_max, paciencia=c.paciencia) for c in configs],
                        subconjunto=sub_skus),
        comando=["python", "questao_01/main.py", *(argv if argv is not None else sys.argv[1:])])
    return dict(rel=rel, tab_gap=tab_gap, tab_sub=tab_sub, tab_comp=tab_comp, antes_depois=antes_depois,
                passos=passos, otim=otim, perm=df_perm, escala=tab_esc, saida=saida)


def imprimir(out):
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== Preços permitidos ===\n", out["perm"].to_string(index=False))
    print("\n=== (A) Brute Force x (B) GA no subconjunto ===\n", out["tab_sub"].to_string(index=False))
    print(out["tab_gap"].drop(columns="politica").round(4).to_string(index=False))
    print("\n=== Comparação de configurações do GA ===\n", out["tab_comp"].round(3).to_string(index=False))
    print("\n=== (C) Greedy: passos ===\n", out["passos"].round(2).to_string(index=False))
    print("\n=== Antes/depois ===\n", out["antes_depois"].round(2).to_string(index=False))
    print("\n=== Otimização de código ===\n", out["otim"].round(5).to_string(index=False))
    if out["escala"] is not None:
        print("\n=== Escalabilidade ===\n", out["escala"].round(4).to_string(index=False))
    print("\n=== RESULTADO FINAL ===")
    for k, v in out["rel"]["final"].items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSaídas em: {out['saida']}")


if __name__ == "__main__":
    imprimir(main())
