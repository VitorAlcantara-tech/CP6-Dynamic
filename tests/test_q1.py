"""Testes da Questão 1 (Brute Force -> GA -> Greedy).
Executar:  python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _util import DADOS, importar_questao

Q = importar_questao("questao_01", ["modelo", "brute_force", "genetic_algorithm", "greedy", "main"])
M, BF, GA, GR, MAIN = Q.modelo, Q.brute_force, Q.genetic_algorithm, Q.greedy, Q.main
CFG_PATH, CAT_PATH = DADOS / "q1_config_G01.json", DADOS / "q1_catalogo_G01.csv"


def catalogo_mini():
    """3 SKUs, 2 de uma categoria (canibalização ativa), números redondos."""
    return pd.DataFrame(dict(
        sku=["A", "B", "C"], categoria=["X", "X", "Y"],
        custo_unitario=[50.0, 40.0, 100.0], preco_atual=[100.0, 90.0, 200.0],
        estoque=[100, 50, 30], demanda_base_90d=[80, 60, 20], elasticidade=[1.5, 2.0, 1.2],
        taxa_marketplace=[0.1, 0.1, 0.1], custo_logistico=[5.0, 5.0, 10.0],
        idade_estoque_dias=[200, 30, 100]))


def config_mini():
    return dict(seed_base=1, seed_final=1001, meta_caixa_90d=1000.0, divida_total_90d=900.0,
                orcamento_promocional=500.0, margem_minima_percentual=0.05,
                niveis_preco_relativos=[0.8, 0.9, 1.0, 1.1])


def inst_mini(niveis=(0.9, 1.0, 1.1)):
    df = catalogo_mini()
    return M.Instancia.criar(df, [np.array(niveis)] * len(df))


class TestEntrada(unittest.TestCase):
    def test_instancia_oficial_valida(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        self.assertEqual(len(df), 30)
        self.assertIn("meta_caixa_90d", cfg)

    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            M.carregar(DADOS / "nao_existe.json", CAT_PATH)

    def test_entradas_invalidas(self):
        casos = {
            "coluna faltando": lambda c, d: d.drop(columns="elasticidade"),
            "estoque negativo": lambda c, d: d.assign(estoque=[-1, 5, 5]),
            "taxa >= 1": lambda c, d: d.assign(taxa_marketplace=[1.0, 0.1, 0.1]),
            "preço zero": lambda c, d: d.assign(preco_atual=[0.0, 90.0, 200.0]),
            "sku duplicado": lambda c, d: d.assign(sku=["A", "A", "C"]),
            "texto em coluna numérica": lambda c, d: d.assign(custo_unitario=["x", 40, 100]),
            "NaN": lambda c, d: d.assign(custo_logistico=[np.nan, 5.0, 10.0]),
            "catálogo vazio": lambda c, d: d.iloc[0:0],
        }
        for nome, f in casos.items():
            with self.subTest(nome), self.assertRaises(ValueError):
                M.validar_entrada(config_mini(), f(config_mini(), catalogo_mini()))

    def test_config_invalida(self):
        casos = [dict(niveis_preco_relativos=[]), dict(niveis_preco_relativos=[0.9, -1]),
                 dict(margem_minima_percentual=1.5), dict(orcamento_promocional=-1)]
        for alt in casos:
            with self.subTest(alt), self.assertRaises(ValueError):
                M.validar_entrada({**config_mini(), **alt}, catalogo_mini())
        cfg = config_mini(); del cfg["meta_caixa_90d"]
        with self.assertRaises(ValueError):
            M.validar_entrada(cfg, catalogo_mini())


class TestCenarioSurpresa(unittest.TestCase):
    def test_fatores_e_remocao(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        c2, d2 = M.aplicar_cenario(cfg, df, meta_fator=1.10, logistica_fator=1.15,
                                   orcamento_fator=0.75, remover=["Q1-005"])
        self.assertAlmostEqual(c2["meta_caixa_90d"], cfg["meta_caixa_90d"] * 1.10)
        self.assertAlmostEqual(c2["orcamento_promocional"], cfg["orcamento_promocional"] * 0.75)
        self.assertAlmostEqual(d2.custo_logistico.iloc[0], df.custo_logistico.iloc[0] * 1.15)
        self.assertNotIn("Q1-005", set(d2.sku))
        self.assertEqual(len(d2), 29)
        self.assertEqual(cfg["meta_caixa_90d"], 362210.17, "o config original não pode ser alterado")

    def test_cenario_invalido(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        with self.assertRaises(ValueError):
            M.aplicar_cenario(cfg, df, remover=["NAO-EXISTE"])
        with self.assertRaises(ValueError):
            M.aplicar_cenario(cfg, df, meta_fator=0)


class TestModelo(unittest.TestCase):
    def test_margem_manual(self):
        df = catalogo_mini().iloc[[0]]
        # p=100: 100*0.9 - 5 - 50 = 35 -> 35%
        self.assertAlmostEqual(M.margem(df, 1.0)[0], 0.35)
        self.assertAlmostEqual(M.margem(df, 1.1)[0], (110 * 0.9 - 55) / 110)

    def test_precos_permitidos_respeitam_regras(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        perm, sinal = M.gerar_precos_permitidos(df, cfg)
        for i, p in enumerate(perm):
            if df.sku[i] in sinal:
                continue
            self.assertTrue((M.margem(df.iloc[[i]], p) >= cfg["margem_minima_percentual"]).all())
            if df.idade_estoque_dias[i] < M.IDADE_MIN_DESC_PROFUNDO:
                self.assertTrue((p >= M.DESCONTO_PROFUNDO).all())

    def test_precos_permitidos_reprodutiveis(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        a, _ = M.gerar_precos_permitidos(df, cfg)
        b, _ = M.gerar_precos_permitidos(df, cfg)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        c, _ = M.gerar_precos_permitidos(df, {**cfg, "seed_base": 999})
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_vetorizado_igual_ingenuo(self):
        inst = inst_mini()
        rows = inst.df.to_dict("records")
        rng = np.random.default_rng(0)
        M_ = rng.choice([0.9, 1.0, 1.1], size=(50, 3))
        f_vet = M.avaliar_vetorizado(inst, M_)
        f_ing = [M.avaliar_ingenuo(rows, M_[i], inst.max_desc) for i in range(50)]
        np.testing.assert_allclose(f_vet, f_ing)

    def test_estoque_limita_vendas_e_inativo_nao_vende(self):
        df = catalogo_mini()
        inst = M.Instancia.criar(df, [np.array([1.0])] * 3, inativos=["C"])
        det = M.avaliar_vetorizado(inst, np.array([[0.5, 1.0, 1.0]]), detalhado=True)
        self.assertLessEqual(det["q"][0, 0], df.estoque[0])
        self.assertEqual(det["q"][0, 2], 0)

    def test_penalidade_limite_de_descontos(self):
        inst = inst_mini()                 # 3 SKUs -> no máx. floor(0.3*3) = 0 com desconto
        f = M.avaliar_vetorizado(inst, np.array([[0.9, 1.0, 1.0]]))
        self.assertLess(f[0], -M.PENALIDADE / 2)


class TestBruteForce(unittest.TestCase):
    def test_ingenuo_igual_vetorizado_e_blocos(self):
        inst = inst_mini()
        f1, p1 = BF.brute_force_ingenuo(inst)
        f2, p2, total = BF.brute_force_vetorizado(inst)
        f3, p3, _ = BF.brute_force_vetorizado(inst, bloco=7)      # força várias partições
        self.assertEqual(total, 27)
        self.assertAlmostEqual(f1, f2); self.assertAlmostEqual(f2, f3)
        np.testing.assert_allclose(p1, p2); np.testing.assert_allclose(p2, p3)

    def test_um_unico_sku_e_um_unico_preco(self):
        inst = M.Instancia.criar(catalogo_mini().iloc[[0]], [np.array([1.0])])
        f, p, total = BF.brute_force_vetorizado(inst)
        self.assertEqual(total, 1); self.assertEqual(p.tolist(), [1.0])


class TestGA(unittest.TestCase):
    def test_encontra_otimo_do_bf_em_instancia_pequena(self):
        inst = inst_mini((0.8, 0.9, 1.0, 1.05, 1.1))
        f_bf, _, _ = BF.brute_force_vetorizado(inst)
        r = GA.ga(inst, GA.ConfigGA("t", populacao=20, geracoes_max=100, paciencia=30, seed=3))
        self.assertAlmostEqual(r["fitness"], f_bf, places=6)

    def test_reprodutivel_e_elitismo_monotonico(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        perm, sinal = M.gerar_precos_permitidos(df, cfg)
        inst = M.Instancia.criar(df, perm, sinal)
        c = GA.ConfigGA("t", populacao=20, geracoes_max=40, paciencia=40, seed=5)
        r1, r2 = GA.ga(inst, c), GA.ga(inst, c)
        self.assertEqual(r1["fitness"], r2["fitness"])
        np.testing.assert_array_equal(r1["politica"], r2["politica"])
        self.assertTrue(np.all(np.diff(r1["hist_melhor"]) >= -1e-9))

    def test_config_invalida(self):
        for kw in [dict(populacao=1), dict(elitismo=50, populacao=10), dict(p_mutacao=1.5),
                   dict(torneio_k=0), dict(paciencia=0)]:
            with self.subTest(kw), self.assertRaises(ValueError):
                GA.ConfigGA("x", **kw)


class TestGreedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        perm, sinal = M.gerar_precos_permitidos(df, cfg)
        cls.inst = M.Instancia.criar(df, perm, sinal)
        cls.pol = np.array([p.max() for p in perm])
        cls.opcoes, _ = GR.opcoes_incentivo(cls.inst, cls.pol, cfg["margem_minima_percentual"])
        cls.orc = cfg["orcamento_promocional"]

    def test_respeita_orcamento_e_heap_igual_ingenuo(self):
        for orc in [0.0, 1000.0, self.orc]:
            for mr in [1.0, 1e-9]:
                with self.subTest(orc=orc, min_razao=mr):
                    aloc, passos, usado = GR.greedy_incentivos(self.opcoes, orc, mr)
                    _, usado_ing = GR.greedy_ingenuo(self.opcoes, orc, mr)
                    self.assertLessEqual(usado, orc + 1e-6)
                    self.assertAlmostEqual(usado, usado_ing, places=6)
                    if len(passos):
                        self.assertTrue((passos.razao >= mr - 1e-12).all())

    def test_orcamento_zero_nao_aloca(self):
        aloc, passos, usado = GR.greedy_incentivos(self.opcoes, 0.0)
        self.assertEqual(usado, 0.0)
        self.assertTrue(all(a["incentivo"] is None for a in aloc.values()))

    def test_entradas_limite(self):
        self.assertEqual(GR.greedy_incentivos(pd.DataFrame(), 100.0)[2], 0.0)
        with self.assertRaises(ValueError):
            GR.greedy_incentivos(self.opcoes, -1.0)

    def test_um_incentivo_por_sku(self):
        aloc, _, _ = GR.greedy_incentivos(self.opcoes, self.orc, 1e-9)
        self.assertEqual(len(aloc), len(set(aloc)))


class TestSubconjunto(unittest.TestCase):
    def test_escolha_automatica(self):
        cfg, df = M.carregar(CFG_PATH, CAT_PATH)
        perm, sinal = M.gerar_precos_permitidos(df, cfg)
        sub = MAIN.escolher_subconjunto(df, perm, sinal)
        self.assertEqual(len(sub), 6)
        self.assertTrue(set(sub).isdisjoint(sinal))
        comb = np.prod([len(perm[int(df.index[df.sku == s][0])]) for s in sub])
        self.assertLessEqual(comb, MAIN.LIMITE_COMBINACOES_BF)
        self.assertEqual(sub, MAIN.escolher_subconjunto(df, perm, sinal))


class TestExecucaoCompleta(unittest.TestCase):
    ARGS = ["--modo", "rapido", "--sem-graficos", "--ga-execucoes", "1",
            "--subconjunto", "Q1-003", "Q1-015", "Q1-005", "Q1-011"]

    def test_reprodutivel_com_seed_e_gera_manifesto(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            o1 = MAIN.main(self.ARGS + ["--seed", "14", "--saida", d1])
            o2 = MAIN.main(self.ARGS + ["--seed", "14", "--saida", d2])
            self.assertEqual(o1["rel"]["final"]["caixa_final_ga_greedy"],
                             o2["rel"]["final"]["caixa_final_ga_greedy"])
            man = json.loads((Path(d1) / "manifesto_dados.json").read_text(encoding="utf-8"))
            self.assertEqual(man["seeds"]["seed_base"], 14)
            self.assertEqual(man["seeds"]["seed_final"], 1014)
            self.assertEqual(man["quantidade_skus"], 30)
            self.assertTrue(man["dados_conferem_com_manifesto_oficial"])
            self.assertIn("meta_caixa_90d", man["metas_financeiras"])
            self.assertTrue((Path(d1) / "resultado.json").exists())

    def test_cenario_surpresa_pela_cli(self):
        with tempfile.TemporaryDirectory() as d:
            o = MAIN.main(self.ARGS + ["--saida", d, "--meta-fator", "1.10", "--orcamento-fator", "0.75",
                                       "--logistica-fator", "1.15", "--remover-sku", "Q1-006",
                                       "--ga-pop", "30", "--ga-mutacao", "0.05"])
            f = o["rel"]["final"]
            self.assertAlmostEqual(f["meta_caixa"], 362210.17 * 1.10, places=4)
            self.assertAlmostEqual(f["orcamento_promocional"], 129986.3 * 0.75, places=4)
            self.assertEqual(len(o["antes_depois"]), 29)
            self.assertIn("pop=30", o["tab_comp"].config.iloc[0])


if __name__ == "__main__":
    unittest.main()
