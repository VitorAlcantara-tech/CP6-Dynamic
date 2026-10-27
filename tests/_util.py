"""Importa os módulos de uma questão isoladamente.

As duas questões têm módulos com o mesmo nome (modelo.py, greedy.py, main.py).
Para testá-las no mesmo processo, cada arquivo de teste importa os módulos da sua
questão com o sys.path apontando para a pasta dela e depois limpa o cache de
módulos — as referências já importadas continuam válidas."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data"
CONFLITANTES = ["modelo", "greedy", "main", "monte_carlo", "dynamic_programming",
                "brute_force", "genetic_algorithm", "escalabilidade"]


def importar_questao(pasta: str, modulos: list[str]) -> SimpleNamespace:
    caminho = str(RAIZ / pasta)
    for m in CONFLITANTES:
        sys.modules.pop(m, None)
    sys.path.insert(0, caminho)
    if str(RAIZ) not in sys.path:
        sys.path.insert(1, str(RAIZ))
    try:
        ns = SimpleNamespace(**{m: importlib.import_module(m) for m in modulos})
    finally:
        while caminho in sys.path:          # main.py também insere a própria pasta
            sys.path.remove(caminho)
        for m in CONFLITANTES:
            sys.modules.pop(m, None)
    return ns
