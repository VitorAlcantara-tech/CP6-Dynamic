"""Testes da Questão 2 (Monte Carlo -> DP -> Greedy -> Monte Carlo).
Executar:  python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _util import DADOS, importar_questao

Q = importar_questao("questao_02", ["modelo", "monte_carlo", "dynamic_programming", "greedy", "main"])
M, MC, DP, GR, MAIN = Q.modelo, Q.monte_carlo, Q.dynamic_programming, Q.greedy, Q.main


def config_mini(**alt):
    cfg = dict(seed_base=1, seed_final=1001, numero_semanas=4, caixa_inicial=1000.0,
               dividas={"semana_2": 300.0, "semana_4": 500.0},
               variabilidade=dict(sigma_demanda=0.1, sigma_preco_concorrente=0.02, sigma_custo_logistico=0.05),
               orcamento_promocional_semanal=1000.0, niveis_preco_relativos=[0.9, 1.0, 1.1],
               limite_risco_falha=0.1)
    cfg.update(alt)
    return cfg


def catalogo_mini(demanda=(5, 3)):
    df = pd.DataFrame(dict(
        sku=["A", "B"], categoria=["X", "Y"], custo_unitario=[50.0, 20.0], preco_atual=[100.0, 60.0],
        estoque_inicial=[30, 10], demanda_base_semanal=list(demanda), elasticidade=[1.5, 1.2],
        taxa_marketplace=[0.1, 0.1], custo_logistico_base=[5.0, 4.0], preco_concorrente_base=[100.0, 60.0]))
    df["gap_ref"] = df.preco_atual / df.preco_concorrente_base
    return df


class TestEntrada(unittest.TestCase):
    def test_instancia_oficial_valida(self):
        cfg, df = M.carregar(DADOS, "G01")
        self.assertEqual(len(df), 24)
        self.assertEqual(M.semanas_dividas(cfg), {4: 85683.08, 8: 110673.98, 12: 160655.78})
        self.assertIn("gap_ref", df)

    def test_entradas_invalidas(self):
        casos = {
            "sem dívidas": (config_mini(dividas={}), catalogo_mini()),
            "dívida fora do horizonte": (config_mini(dividas={"semana_9": 1.0}), catalogo_mini()),
            "chave de dívida inválida": (config_mini(dividas={"vencimento": 1.0}), catalogo_mini()),
            "sigma negativo": (config_mini(variabilidade=dict(sigma_demanda=-0.1, sigma_preco_concorrente=0,
                                                               sigma_custo_logistico=0)), catalogo_mini()),
            "semanas = 0": (config_mini(numero_semanas=0), catalogo_mini()),
            "estoque fracionário": (config_mini(), catalogo_mini().assign(estoque_inicial=[1.5, 2])),
            "concorrente <= 0": (config_mini(), catalogo_mini().assign(preco_concorrente_base=[0.0, 60.0])),
            "coluna faltando": (config_mini(), catalogo_mini().drop(columns="elasticidade")),
            "limite de risco > 1": (config_mini(limite_risco_falha=2), catalogo_mini()),
        }
        for nome, (cfg, df) in casos.items():
            with self.subTest(nome), self.assertRaises(ValueError):
                M.validar_entrada(cfg, df)

    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            M.carregar(DADOS, "G99")


class TestCenarios(unittest.TestCase):
    def test_formas_e_reprodutibilidade(self):
        cfg, df = config_mini(), catalogo_mini()
        a, b = M.gerar_cenarios(df, cfg, 200, 7), M.gerar_cenarios(df, cfg, 200, 7)
        c = M.gerar_cenarios(df, cfg, 200, 8)
        self.assertEqual(a["eps"].shape, (200, 4, 2)); self.assertEqual(a["logm"].shape, (200, 4))
        for k in ["eps", "comp", "logm"]:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertFalse(np.array_equal(a["eps"], c["eps"]))

    def test_sem_incerteza_e_media_unitaria(self):
        df = catalogo_mini()
        zero = M.gerar_cenarios(df, config_mini(), 50, 1, escala_sigma=0.0)
        for k in ["eps", "comp", "logm"]:
            np.testing.assert_allclose(zero[k], 1.0)
        grande = M.gerar_cenarios(df, config_mini(), 20000, 1)
        self.assertAlmostEqual(float(grande["eps"].mean()), 1.0, delta=0.01)   # lognormal com média 1

    def test_fator_concorrencia(self):
        df = catalogo_mini()
        np.testing.assert_allclose(M.fator_concorrencia(df, 1.0, np.ones(2)), 1.0)
        self.assertTrue((M.fator_concorrencia(df, 1.1, np.ones(2)) < 1).all())
        _, df2 = M.aplicar_cenario(config_mini(), df, concorrente_fator=0.92)
        self.assertTrue((M.fator_concorrencia(df2, 1.0, np.ones(2)) < 1).all(),
                        "concorrente 8% mais barato deve reduzir a demanda no preço atual")


class TestDP(unittest.TestCase):
    def test_caso_deterministico_calculado_a_mao(self):
        # demanda sempre 2; recompensa 10 (preço baixo) ou 30 (preço alto); estoque 3; 2 semanas
        pmf = np.zeros((2, 2, 4)); pmf[:, :, 2] = 1.0
        r = np.array([[10.0, 30.0], [10.0, 30.0]])
        V, PI, info = DP.resolver_sku(pmf, r, 3)
        self.assertAlmostEqual(V[0, 3], 2 * 30 + 1 * 30)       # vende 2 e depois 1
        self.assertTrue((PI[:, 1:] == 1).all())
        self.assertEqual(info["estados"], 2 * 4)

    def test_estoque_zero_e_empate(self):
        pmf = np.zeros((3, 3, 1)); pmf[:, :, 0] = 1.0
        V, PI, _ = DP.resolver_sku(pmf, np.ones((3, 3)), 0)
        self.assertTrue(np.all(V == 0))
        self.assertTrue((PI == 2).all(), "empate deve escolher o preço mais alto")

    def test_bottom_up_igual_top_down_e_memo_reduz_chamadas(self):
        rng = np.random.default_rng(0)
        T, Mn, D, S = 3, 3, 6, 8
        pmf = rng.random((T, Mn, D + 1)); pmf /= pmf.sum(axis=2, keepdims=True)
        r = rng.uniform(-5, 20, (T, Mn))
        V, _, _ = DP.resolver_sku(pmf, r, S)
        v_memo, c_memo = DP.resolver_top_down(pmf, r, S, memo=True)
        v_sem, c_sem = DP.resolver_top_down(pmf, r, S, memo=False)
        self.assertAlmostEqual(V[0, S], v_memo); self.assertAlmostEqual(v_memo, v_sem)
        self.assertLess(c_memo["chamadas"], c_sem["chamadas"])
        self.assertGreater(c_memo["acertos_cache"], 0)

    def test_valor_nao_decresce_com_estoque_se_recompensa_positiva(self):
        rng = np.random.default_rng(1)
        pmf = rng.random((4, 2, 10)); pmf /= pmf.sum(axis=2, keepdims=True)
        V, _, _ = DP.resolver_sku(pmf, rng.uniform(1, 5, (4, 2)), 9)
        self.assertTrue(np.all(np.diff(V[0]) >= -1e-9))

    def test_reconstrucao_do_caminho(self):
        pmf = np.zeros((3, 2, 4)); pmf[:, :, 1] = 1.0
        _, PI, _ = DP.resolver_sku(pmf, np.array([[1.0, 2.0]] * 3), 3)
        cam = DP.reconstruir(pmf, PI, 3, np.array([0.9, 1.1]))
        self.assertEqual([c["estoque_inicio"] for c in cam], [3, 2, 1])
        self.assertTrue(all(c["nivel"] == 1.1 for c in cam))


class TestSimulacao(unittest.TestCase):
    def setUp(self):
        self.df = catalogo_mini(demanda=(0, 0))                 # sem vendas: caixa só muda nas dívidas
        self.plano = dict(tipo="fixo", IDX=np.ones((4, 2), dtype=int))

    def test_paga_todas_quando_ha_caixa(self):
        cfg = config_mini()
        sim = MC.simular(self.df, cfg, M.gerar_cenarios(self.df, cfg, 100, 1), self.plano, seed=1)
        self.assertTrue(sim["pago"].all())
        np.testing.assert_allclose(sim["caixa_final"], 1000 - 300 - 500)
        self.assertEqual(sim["unidades"].sum(), 0)

    def test_falha_e_rola_saldo(self):
        cfg = config_mini(caixa_inicial=400.0)
        sim = MC.simular(self.df, cfg, M.gerar_cenarios(self.df, cfg, 10, 1), self.plano, seed=1)
        self.assertTrue(sim["pago"][:, 0].all()); self.assertFalse(sim["pago"][:, 1].any())
        np.testing.assert_allclose(sim["caixa_final"], -400.0)
        r = MC.resumir(sim, cfg, "teste")
        self.assertEqual(r["p_semana_2"], 1.0); self.assertEqual(r["p_semana_4"], 0.0)
        self.assertEqual(r["cenarios_falha"], 10); self.assertFalse(r["atende_limite_risco"])

    def test_vendas_nunca_excedem_estoque(self):
        cfg, df = config_mini(), catalogo_mini(demanda=(500, 500))
        sim = MC.simular(df, cfg, M.gerar_cenarios(df, cfg, 200, 2), self.plano, seed=2)
        self.assertTrue((sim["hist_estoque"] >= 0).all())
        self.assertTrue((sim["estoque_final"] == 0).all())


class TestGreedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.df = M.carregar(DADOS, "G01")
        cls.est = MC.estimar(cls.df, cls.cfg, M.gerar_cenarios(cls.df, cls.cfg, 300, 3))
        cls.dp = DP.resolver_catalogo(cls.df, cls.est, 0.05)

    def test_orcamento_e_limite_de_destaques(self):
        for mr in [1.0, 1e-9]:
            _, aloc, _, uso = GR.planejar(self.df, self.cfg, self.est, self.dp, seed=1, min_razao=mr)
            self.assertTrue((uso.usado <= uso.orcamento + 1e-6).all())
            if len(aloc):
                dest = aloc[aloc.recurso == "destaque"].groupby("semana").size()
                self.assertTrue((dest <= M.RECURSOS["destaque"]["limite_semana"]).all())
                self.assertFalse(aloc.duplicated(["semana", "sku"]).any(), "1 recurso por SKU/semana")

    def test_orcamento_zero(self):
        cfg = {**self.cfg, "orcamento_promocional_semanal": 0.0}
        promo, aloc, _, uso = GR.planejar(self.df, cfg, self.est, self.dp, seed=1)
        self.assertEqual(len(aloc), 0); self.assertEqual(uso.usado.sum(), 0)
        self.assertTrue((promo["uplift"] == 1).all())


class TestExecucaoCompleta(unittest.TestCase):
    ARGS = ["--modo", "rapido", "--sem-graficos", "--cenarios", "400", "--cenarios-construcao", "300",
            "--grade-rho", "0", "0.1"]

    def test_reprodutivel_e_manifesto(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            o1 = MAIN.main(self.ARGS + ["--seed", "14", "--saida", d1])
            o2 = MAIN.main(self.ARGS + ["--seed", "14", "--saida", d2])
            self.assertEqual(o1["rel"]["caixa_final_esperado"], o2["rel"]["caixa_final_esperado"])
            man = json.loads((Path(d1) / "manifesto_dados.json").read_text(encoding="utf-8"))
            self.assertEqual(man["seeds"]["seed_base"], 14)
            self.assertEqual(man["quantidade_skus"], 24)
            self.assertTrue(man["dados_conferem_com_manifesto_oficial"])
            self.assertTrue(any("abaixo do mínimo" in a for a in man["parametros_execucao"]["avisos"]))

    def test_cenario_surpresa_pela_cli(self):
        with tempfile.TemporaryDirectory() as d:
            o = MAIN.main(self.ARGS + ["--saida", d, "--meta-fator", "1.10", "--concorrente-fator", "0.92",
                                       "--logistica-fator", "1.15", "--orcamento-fator", "0.75",
                                       "--remover-sku", "Q2-003"])
            man = json.loads((Path(d) / "manifesto_dados.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(man["metas_financeiras"]["dividas"]["semana_4"], 85683.08 * 1.10, places=4)
            self.assertEqual(man["quantidade_skus"], 23)
            self.assertNotIn("Q2-003", o["precos"].index)

    def test_argumentos_invalidos(self):
        for extra in [["--cenarios", "0"], ["--percentil-risco", "150"], ["--grade-rho", "-1"]]:
            with self.subTest(extra), self.assertRaises(SystemExit):
                MAIN.parse_args(extra)


if __name__ == "__main__":
    unittest.main()
