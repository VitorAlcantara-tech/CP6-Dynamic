"""Testes dos utilitários de reprodutibilidade (manifesto e hashes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _util import DADOS, RAIZ
import sys
sys.path.insert(0, str(RAIZ))
import comum  # noqa: E402


class TestHashes(unittest.TestCase):
    def test_sha256_conhecido(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.txt"
            f.write_bytes(b"abc")
            self.assertEqual(comum.sha256_arquivo(f), hashlib.sha256(b"abc").hexdigest())

    def test_dados_oficiais_conferem(self):
        arqs = sorted(DADOS.glob("q[12]_*_G01.*"))
        self.assertEqual(len(arqs), 4)
        st = comum.verificar_hashes(arqs, DADOS)
        self.assertTrue(all(v["status"] == "ok" for v in st.values()), st)

    def test_arquivo_alterado_e_detectado(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(DADOS / "manifesto_dados.json", d)
            alvo = Path(d) / "q1_catalogo_G01.csv"
            shutil.copy(DADOS / "q1_catalogo_G01.csv", alvo)
            alvo.write_text(alvo.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            st = comum.verificar_hashes([alvo], Path(d))
            self.assertEqual(st["q1_catalogo_G01.csv"]["status"], "DIVERGENTE")

    def test_manifesto_gerado(self):
        with tempfile.TemporaryDirectory() as d:
            m = comum.gerar_manifesto(Path(d) / "m.json", "q", "G01", [DADOS / "q1_config_G01.json"], DADOS,
                                      seeds=dict(seed_base=1), n_skus=3, skus_ativos=["a"],
                                      metas=dict(meta=1.0), parametros={}, comando=["x"])
            lido = json.loads((Path(d) / "m.json").read_text(encoding="utf-8"))
            self.assertEqual(lido["sha256_dataset"], m["sha256_dataset"])
            for chave in ["seeds", "quantidade_skus", "metas_financeiras", "arquivos", "ambiente"]:
                self.assertIn(chave, lido)
            self.assertIn("python", lido["ambiente"])


class TestSeeds(unittest.TestCase):
    def test_resolver_seeds(self):
        cfg = dict(seed_base=28, seed_final=1028)
        s = comum.resolver_seeds(cfg, argparse.Namespace(seed=14, seed_final=None))
        self.assertEqual(s, dict(seed_base=14, seed_final=1014))
        s = comum.resolver_seeds(dict(seed_base=28, seed_final=1028), argparse.Namespace(seed=None, seed_final=5))
        self.assertEqual(s, dict(seed_base=28, seed_final=5))

    def test_fator_invalido(self):
        for v in [0, -1]:
            with self.assertRaises(ValueError):
                comum.checar_fator("x", v)


if __name__ == "__main__":
    unittest.main()
