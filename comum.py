"""
Utilitários compartilhados pelas duas questões:
  - argumentos de linha de comando comuns (seed, dados, cenário-surpresa);
  - manifesto_dados.json (seed, nº de SKUs, metas, hash SHA-256 dos dados, versões);
  - verificação dos hashes contra data/manifesto_dados.json (manifesto oficial).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DADOS_PADRAO = RAIZ / "data"


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def parser_base(descricao: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=descricao,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("dados e reprodutibilidade")
    g.add_argument("--dados", type=Path, default=DADOS_PADRAO, help="pasta com config e catálogo")
    g.add_argument("--grupo", default="G01", help="sufixo dos arquivos (q?_config_<grupo>.json)")
    g.add_argument("--config", type=Path, default=None, help="caminho explícito do config JSON")
    g.add_argument("--catalogo", type=Path, default=None, help="caminho explícito do catálogo CSV")
    g.add_argument("--seed", type=int, default=None,
                   help="substitui seed_base do config; seed_final = seed + 1000 (salvo --seed-final)")
    g.add_argument("--seed-final", type=int, default=None, help="substitui seed_final do config")
    g.add_argument("--modo", choices=["rapido", "completo"], default="completo",
                   help="rapido: menos repetições, sem estudo de escalabilidade")
    g.add_argument("--saida", type=Path, default=None, help="pasta de saída (padrão: <questao>/saida)")
    g.add_argument("--sem-graficos", action="store_true", help="não gera figuras (útil em testes)")

    s = p.add_argument_group("cenário-surpresa (alterações sem mexer no código)")
    s.add_argument("--meta-fator", type=float, default=1.0,
                   help="multiplica a meta de caixa (Q1) ou as dívidas (Q2). Ex.: 1.10 = +10%%")
    s.add_argument("--logistica-fator", type=float, default=1.0,
                   help="multiplica o custo logístico. Ex.: 1.15 = +15%%")
    s.add_argument("--orcamento-fator", type=float, default=1.0,
                   help="multiplica o orçamento promocional. Ex.: 0.75 = -25%%")
    s.add_argument("--remover-sku", action="append", default=[], metavar="SKU",
                   help="remove um SKU do catálogo (pode repetir)")
    return p


def resolver_seeds(cfg: dict, args) -> dict:
    """Aplica --seed / --seed-final ao config e devolve as seeds efetivas."""
    if args.seed is not None:
        cfg["seed_base"] = int(args.seed)
        cfg["seed_final"] = int(args.seed) + 1000
    if args.seed_final is not None:
        cfg["seed_final"] = int(args.seed_final)
    return dict(seed_base=cfg["seed_base"], seed_final=cfg["seed_final"])


def checar_fator(nome: str, valor: float):
    if not (valor > 0):
        raise ValueError(f"{nome} deve ser > 0 (recebido {valor})")


def remover_skus(df, skus: list[str]):
    if not skus:
        return df
    faltando = sorted(set(skus) - set(df["sku"]))
    if faltando:
        raise ValueError(f"SKU(s) inexistente(s) no catálogo: {faltando}")
    out = df[~df["sku"].isin(skus)].reset_index(drop=True)
    if out.empty:
        raise ValueError("catálogo vazio após remover SKUs")
    return out


# ---------------------------------------------------------------------------
# Manifesto e hashes
# ---------------------------------------------------------------------------
def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 16), b""):
            h.update(bloco)
    return h.hexdigest()


def versoes() -> dict:
    out = dict(python=sys.version.split()[0], plataforma=platform.platform())
    for mod in ["numpy", "pandas", "matplotlib"]:
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def verificar_hashes(arquivos: list[Path], pasta_dados: Path) -> dict:
    """Compara o SHA-256 de cada arquivo com o manifesto oficial (se existir)."""
    oficial_path = Path(pasta_dados) / "manifesto_dados.json"
    oficial = {}
    if oficial_path.exists():
        oficial = json.loads(oficial_path.read_text(encoding="utf-8")).get("arquivos", {})
    status = {}
    for a in arquivos:
        h = sha256_arquivo(a)
        esperado = oficial.get(Path(a).name)
        status[Path(a).name] = dict(
            sha256=h,
            status=("sem manifesto oficial" if not oficial else
                    "não listado" if esperado is None else
                    "ok" if esperado == h else "DIVERGENTE"))
    return status


def gerar_manifesto(destino: Path, questao: str, grupo: str, arquivos: list[Path], pasta_dados: Path,
                    seeds: dict, n_skus: int, skus_ativos: list[str], metas: dict, parametros: dict,
                    comando: list[str] | None = None) -> dict:
    hashes = verificar_hashes(arquivos, pasta_dados)
    conj = hashlib.sha256("".join(f"{n}:{v['sha256']}\n" for n, v in sorted(hashes.items()))
                          .encode()).hexdigest()
    m = dict(
        questao=questao, grupo=grupo,
        gerado_em_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        comando=" ".join(comando if comando is not None else sys.argv),
        seeds=seeds, quantidade_skus=n_skus, skus=skus_ativos, metas_financeiras=metas,
        arquivos=hashes, sha256_dataset=conj,
        dados_conferem_com_manifesto_oficial=all(v["status"] in ("ok", "sem manifesto oficial")
                                                 for v in hashes.values()),
        parametros_execucao=parametros, ambiente=versoes(),
    )
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return m
