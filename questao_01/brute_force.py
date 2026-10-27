"""(A) Brute Force: enumera todas as combinações de preços permitidos de um
subconjunto pequeno e devolve o ótimo exato (benchmark para validar o GA)."""
from __future__ import annotations

import itertools

import numpy as np

from modelo import Instancia, avaliar_ingenuo, avaliar_vetorizado


def brute_force_ingenuo(inst: Instancia):
    rows = inst.df.to_dict("records")
    melhor, melhor_ms = -np.inf, None
    for ms in itertools.product(*inst.permitidos):
        f = avaliar_ingenuo(rows, ms, inst.max_desc)
        if f > melhor:
            melhor, melhor_ms = f, ms
    return melhor, np.array(melhor_ms)


def brute_force_vetorizado(inst: Instancia, bloco=200_000):
    """Enumera TODAS as combinações via produto cartesiano de índices e avalia
    em blocos (memória limitada a 'bloco' linhas por vez)."""
    tamanhos = [len(p) for p in inst.permitidos]
    total = int(np.prod(tamanhos))
    melhor, melhor_ms = -np.inf, None
    for ini in range(0, total, bloco):
        idx = np.arange(ini, min(ini + bloco, total))
        cols = np.unravel_index(idx, tamanhos)            # índices por SKU
        M = np.column_stack([inst.permitidos[j][cols[j]] for j in range(inst.n)])
        f = avaliar_vetorizado(inst, M)
        k = int(np.argmax(f))
        if f[k] > melhor:
            melhor, melhor_ms = float(f[k]), M[k].copy()
    return melhor, melhor_ms, total
