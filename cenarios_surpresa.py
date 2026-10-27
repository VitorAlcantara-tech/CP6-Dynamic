"""
Roda os cenários-surpresa da defesa técnica SÓ com parâmetros de linha de comando
(nenhuma linha de código é alterada) e consolida os resultados.

  python cenarios_surpresa.py            # todos os cenários (cerca de 2 min)
  python cenarios_surpresa.py --questao 1

Saída: cenarios_surpresa/resumo.csv, cenarios_surpresa/resumo.md e uma pasta por cenário.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "cenarios_surpresa"

CENARIOS = {
    1: [
        ("base", []),
        ("meta de caixa +10%", ["--meta-fator", "1.10"]),
        ("custo logístico +15%", ["--logistica-fator", "1.15"]),
        ("orçamento promocional -25%", ["--orcamento-fator", "0.75"]),
        ("SKU Q1-023 indisponível", ["--remover-sku", "Q1-023"]),
        ("GA: população 80, mutação 0,10", ["--ga-pop", "80", "--ga-mutacao", "0.10"]),
        ("combinado: meta +10%, logística +15%, orçamento -25%",
         ["--meta-fator", "1.10", "--logistica-fator", "1.15", "--orcamento-fator", "0.75"]),
    ],
    2: [
        ("base", []),
        ("dívidas +10%", ["--meta-fator", "1.10"]),
        ("concorrente reduz o preço em 8%", ["--concorrente-fator", "0.92"]),
        ("custo logístico +15%", ["--logistica-fator", "1.15"]),
        ("orçamento promocional -25%", ["--orcamento-fator", "0.75"]),
        ("SKU Q2-003 indisponível", ["--remover-sku", "Q2-003"]),
        ("combinado: concorrente -8% e logística +15%",
         ["--concorrente-fator", "0.92", "--logistica-fator", "1.15"]),
    ],
}


def resumo_q1(r):
    f = r["final"]
    return {"caixa final (R$)": f["caixa_final_ga_greedy"], "meta (R$)": f["meta_caixa"],
            "folga (R$)": f["folga_ou_insuficiencia"], "meta atingida": f["meta_atingida"],
            "cobertura dívida (%)": f["cobertura_divida_pct"], "orçamento usado (R$)": f["orcamento_usado"],
            "gap GA×BF (%)": r["gap_max_pct"], "SKUs fora da campanha": len(f["skus_fora_campanha"])}


def resumo_q2(r):
    out = {f"P pagar sem. {k.split('_')[-1]}": v for k, v in r["p_pagamento"].items() if k != "todas"}
    out.update({"P pagar todas": r["p_pagamento"]["todas"], "caixa final médio (R$)": r["caixa_final_esperado"],
                f"caixa final P{r['percentil_risco']:g} (R$)": r["caixa_final_percentil_risco"],
                "cenários com falha": r["cenarios_falha"], "atende limite de risco": r["atende_limite_risco"],
                "ρ escolhido": r["rho_escolhido"], "gasto promocional planejado (R$)": r["greedy"]["gasto_planejado"]})
    return out


def para_markdown(df: pd.DataFrame) -> str:
    """Tabela markdown sem dependências extras (não exige o pacote tabulate)."""
    def fmt(v):
        if isinstance(v, bool):
            return "sim" if v else "não"
        if isinstance(v, float):
            if v != v:
                return "-"
            if v.is_integer() and abs(v) < 1000:
                return str(int(v))
            if abs(v) < 1:
                return f"{v:.4f}".replace(".", ",")
            if abs(v) < 1000:
                return f"{v:.1f}".replace(".", ",")
            return f"{v:,.0f}".replace(",", ".")
        return str(v)
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    linhas = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questao", type=int, choices=[1, 2], nargs="+", default=[1, 2])
    ap.add_argument("--so-resumo", action="store_true", help="só refaz resumo.md a partir do resumo.csv")
    args = ap.parse_args()
    linhas = []
    for q in ([] if args.so_resumo else args.questao):
        for i, (nome, extra) in enumerate(CENARIOS[q]):
            pasta = SAIDA / f"q{q}_{i:02d}"
            cmd = [sys.executable, str(RAIZ / f"questao_0{q}" / "main.py"), "--modo", "rapido",
                   "--sem-graficos", "--saida", str(pasta), *extra]
            print(f"[Q{q}] {nome}: {' '.join(cmd[1:])}", flush=True)
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            r = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
            linhas.append({"questão": q, "cenário": nome,
                           "parâmetros": " ".join(extra) or "(nenhum)",
                           "tempo (s)": round(time.perf_counter() - t0, 1),
                           **(resumo_q1(r) if q == 1 else resumo_q2(r))})
    SAIDA.mkdir(exist_ok=True)
    if args.so_resumo:
        tab = pd.read_csv(SAIDA / "resumo.csv")
    else:
        tab = pd.DataFrame(linhas)
        tab.to_csv(SAIDA / "resumo.csv", index=False)
    with open(SAIDA / "resumo.md", "w", encoding="utf-8") as f:
        for q in args.questao:
            t = tab[tab["questão"] == q].dropna(axis=1, how="all").drop(columns="questão")
            f.write(f"## Questão {q}\n\n{para_markdown(t)}\n\n")
    print(f"\nResumo em {SAIDA / 'resumo.md'}")


if __name__ == "__main__":
    main()
