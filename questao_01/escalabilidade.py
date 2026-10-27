"""Medição real dos 3 algoritmos em entradas pequena, média e grande
(time.perf_counter + tracemalloc). Usado por main.py no modo completo."""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from brute_force import brute_force_ingenuo, brute_force_vetorizado
from genetic_algorithm import ga
from greedy import greedy_incentivos, greedy_ingenuo, opcoes_incentivo
from modelo import Instancia, gerar_precos_permitidos, medir


def replicar(df: pd.DataFrame, vezes: int) -> pd.DataFrame:
    partes = []
    for k in range(vezes):
        c = df.copy()
        if k:
            c["sku"] = c["sku"] + f"-r{k}"
        partes.append(c)
    return pd.concat(partes, ignore_index=True)


def medir_escalabilidade(df, cfg, inst_sub, config_ga, saida, graficos=True):
    linhas = []
    # ---- Brute Force: k-2, k-1 e k SKUs do subconjunto (Π L_j combinações) ----
    k = inst_sub.n
    for nome, kk in zip(["pequena", "média", "grande"], [max(1, k - 2), max(1, k - 1), k]):
        sub = Instancia.criar(inst_sub.df.iloc[:kk], inst_sub.permitidos[:kk])
        n_comb = int(np.prod([len(p) for p in sub.permitidos]))
        (_, t_v, m_v) = medir(brute_force_vetorizado, sub)
        (_, t_i, m_i) = medir(brute_force_ingenuo, sub)
        linhas.append(dict(algoritmo="Brute Force", entrada=nome, tamanho=f"k={kk} SKUs",
                           n=kk, combinacoes=n_comb, tempo_s=t_v, mem_kib=m_v,
                           tempo_ingenuo_s=t_i, mem_ingenuo_kib=m_i))

    # ---- GA e Greedy: catálogo com 1/3, 1x e 3x o número de SKUs ----
    n = len(df)
    tamanhos = [("pequena", df.iloc[: max(2, n // 3)].reset_index(drop=True)),
                ("média", df), ("grande", replicar(df, 3))]
    for nome, dfx in tamanhos:
        perm, sinal = gerar_precos_permitidos(dfx, cfg)
        inst = Instancia.criar(dfx, perm, sinal)
        c = replace(config_ga, p_mutacao=1.0 / inst.n, seed=cfg["seed_final"])
        (r, t, m) = medir(ga, inst, c)
        linhas.append(dict(algoritmo="GA", entrada=nome, tamanho=f"n={inst.n} SKUs", n=inst.n,
                           populacao=c.populacao, geracoes=r["geracoes"], avaliacoes=r["avaliacoes"],
                           tempo_s=t, mem_kib=m, tempo_por_geracao_ms=1000 * t / max(r["geracoes"], 1)))
        opcoes, _ = opcoes_incentivo(inst, r["politica"], cfg["margem_minima_percentual"])
        (_, tg, mg) = medir(greedy_incentivos, opcoes, cfg["orcamento_promocional"], 1e-9)
        (_, ti, mi) = medir(greedy_ingenuo, opcoes, cfg["orcamento_promocional"], 1e-9)
        linhas.append(dict(algoritmo="Greedy", entrada=nome, tamanho=f"n={inst.n} SKUs", n=inst.n,
                           pares_sku_opcao=len(opcoes), tempo_s=tg, mem_kib=mg,
                           tempo_ingenuo_s=ti, mem_ingenuo_kib=mi))

    # ---- quando o BF deixa de ser viável: extrapolação para o catálogo completo ----
    perm, _ = gerar_precos_permitidos(df, cfg)
    total = float(np.prod([len(p) for p in perm], dtype=float))
    bf_g = [l for l in linhas if l["algoritmo"] == "Brute Force"][-1]
    taxa_vet = bf_g["combinacoes"] / bf_g["tempo_s"]
    taxa_ing = bf_g["combinacoes"] / bf_g["tempo_ingenuo_s"]
    linhas.append(dict(algoritmo="Brute Force (extrapolado)", entrada="catálogo completo",
                       tamanho=f"n={len(df)} SKUs", n=len(df), combinacoes=total,
                       tempo_s=total / taxa_vet, tempo_ingenuo_s=total / taxa_ing))
    tab = pd.DataFrame(linhas)
    tab.to_csv(saida / "06_escalabilidade.csv", index=False)

    if graficos:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, alg in zip(axes, ["Brute Force", "GA", "Greedy"]):
            t = tab[tab.algoritmo == alg]
            x = t.combinacoes if alg == "Brute Force" else t.n
            ax.plot(x, t.tempo_s, "o-", label="tempo (s)")
            ax2 = ax.twinx()
            ax2.plot(x, t.mem_kib, "s--", color="#e07a5f", label="memória (KiB)")
            ax.set_xlabel("combinações (Π L_j)" if alg == "Brute Force" else "nº de SKUs (n)")
            if alg == "Brute Force":
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel("tempo (s)"); ax2.set_ylabel("pico de memória (KiB)")
            ax.set_title(alg); ax.grid(alpha=.3)
        fig.suptitle("Escalabilidade (entradas pequena, média e grande)")
        fig.tight_layout(); fig.savefig(saida / "fig4_escalabilidade.png", dpi=150); plt.close(fig)
    return tab
