"""
Comando único para reproduzir os resultados principais:

    python reproduzir.py            # testes + Questão 1 + Questão 2 (modo completo, ~1 min)
    python reproduzir.py --rapido   # testes + as duas questões em modo rápido
    python reproduzir.py --seed 14  # mesma sequência com outra seed

Equivale a:
    python -m unittest discover -s tests
    python questao_01/main.py --modo completo [--seed S]
    python questao_02/main.py --modo completo --cenarios 5000 [--seed S]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def rodar(cmd, log: Path | None = None):
    """Executa o comando; se log for dado, grava a saída também em arquivo (evidência)."""
    print("\n$", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, cwd=RAIZ)
        return
    r = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-3000:], r.stderr, sep="", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ python {' '.join(cmd[1:])}\n\n{r.stdout}\n{r.stderr}", encoding="utf-8")
    r.check_returncode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sem-testes", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    modo = ["--modo", "rapido" if a.rapido else "completo"]
    seed = ["--seed", str(a.seed)] if a.seed is not None else []
    print(f"Python {sys.version.split()[0]} em {py}")
    if not a.sem_testes:
        rodar([py, "-m", "unittest", "discover", "-s", "tests"])
    rodar([py, "questao_01/main.py", *modo, *seed], RAIZ / "questao_01" / "saida" / "log_execucao.txt")
    rodar([py, "questao_02/main.py", *modo, "--cenarios", "5000", *seed],
          RAIZ / "questao_02" / "saida" / "log_execucao.txt")
    print("\nResultados em questao_01/saida e questao_02/saida (ver manifesto_dados.json em cada uma).")


if __name__ == "__main__":
    main()
