"""(B) Algoritmo Genético: busca a política de preços (um nível permitido por SKU)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelo import PENALIDADE, Instancia, avaliar_vetorizado


@dataclass
class ConfigGA:
    nome: str
    populacao: int = 80
    geracoes_max: int = 400
    torneio_k: int = 3
    p_crossover: float = 0.9
    p_mutacao: float | None = None   # None => 1/n (uma mutação esperada por indivíduo)
    elitismo: int = 2
    paciencia: int = 60              # parada por estagnação
    seed: int = 0

    def __post_init__(self):
        if self.populacao < 2:
            raise ValueError("populacao deve ser >= 2")
        if not 0 <= self.elitismo < self.populacao:
            raise ValueError("elitismo deve estar em [0, populacao)")
        if self.torneio_k < 1:
            raise ValueError("torneio_k deve ser >= 1")
        if not 0 <= self.p_crossover <= 1:
            raise ValueError("p_crossover deve estar em [0, 1]")
        if self.p_mutacao is not None and not 0 <= self.p_mutacao <= 1:
            raise ValueError("p_mutacao deve estar em [0, 1]")
        if self.geracoes_max < 0 or self.paciencia < 1:
            raise ValueError("geracoes_max >= 0 e paciencia >= 1")


def ga(inst: Instancia, cfg: ConfigGA, populacao_inicial=None):
    """Cromossomo: vetor de inteiros, gene j = índice do preço permitido do SKU j.

    - Inicialização: aleatória + 1 indivíduo 'preço atual' (semente plausível).
    - Seleção: torneio de tamanho k (pressão seletiva ajustável, sem escala de fitness,
      robusto a fitness negativos/penalizados).
    - Crossover: uniforme (genes independentes por SKU; não há ordem entre SKUs,
      então crossover de 1 ponto introduziria viés posicional).
    - Mutação: troca o índice por outro preço permitido, prob. p_m por gene.
    - Elitismo: os 'e' melhores passam intactos (garante fitness monotônica).
    - Parada: geracoes_max OU 'paciencia' gerações sem melhoria do melhor.
    """
    rng = np.random.default_rng(cfg.seed)
    n = inst.n
    L = np.array([len(p) for p in inst.permitidos])
    T = inst.tabela_m()
    pm = cfg.p_mutacao if cfg.p_mutacao is not None else 1.0 / n

    def decod(G):
        return T[np.arange(n), G]

    pop = rng.integers(0, L, size=(cfg.populacao, n))
    idx_atual = np.array([int(np.argmin(np.abs(p - 1.0))) for p in inst.permitidos])
    pop[0] = idx_atual
    if populacao_inicial is not None:
        pop[1:1 + len(populacao_inicial)] = populacao_inicial

    fit = avaliar_vetorizado(inst, decod(pop))
    hist_melhor, hist_media = [fit.max()], [fit[fit > -PENALIDADE / 2].mean()]
    melhor_global, sem_melhora, avaliacoes = fit.max(), 0, cfg.populacao

    for g in range(cfg.geracoes_max):
        ordem = np.argsort(-fit)
        elite = pop[ordem[:cfg.elitismo]].copy()

        # seleção por torneio (vetorizada): escolhe k candidatos por vaga
        n_filhos = cfg.populacao - cfg.elitismo
        cand = rng.integers(0, cfg.populacao, size=(2 * n_filhos, cfg.torneio_k))
        venc = cand[np.arange(len(cand)), np.argmax(fit[cand], axis=1)]
        pais1, pais2 = pop[venc[:n_filhos]], pop[venc[n_filhos:]]

        # crossover uniforme
        faz_cx = rng.random(n_filhos) < cfg.p_crossover
        mascara = rng.random((n_filhos, n)) < 0.5
        filhos = np.where(mascara & faz_cx[:, None], pais2, pais1)

        # mutação: novo índice uniforme dentro dos permitidos daquele SKU
        mut = rng.random((n_filhos, n)) < pm
        novos = (rng.random((n_filhos, n)) * L).astype(int)
        filhos = np.where(mut, novos, filhos)

        pop = np.vstack([elite, filhos])
        fit = avaliar_vetorizado(inst, decod(pop))
        avaliacoes += n_filhos

        hist_melhor.append(fit.max())
        validos = fit[fit > -PENALIDADE / 2]
        hist_media.append(validos.mean() if len(validos) else np.nan)

        if fit.max() > melhor_global + 1e-6:
            melhor_global, sem_melhora = fit.max(), 0
        else:
            sem_melhora += 1
            if sem_melhora >= cfg.paciencia:
                break

    k = int(np.argmax(fit))
    return dict(fitness=float(fit[k]), politica=decod(pop[k:k + 1])[0],
                hist_melhor=np.array(hist_melhor), hist_media=np.array(hist_media),
                geracoes=len(hist_melhor) - 1, avaliacoes=avaliacoes)
