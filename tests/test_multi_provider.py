import unittest

from langchain_core.messages import AIMessage

from src.chain.multi_provider import (
    DEFAULT_MODELS,
    DEFAULT_PROMPT_VERSIONS,
    MultiProviderDemo,
    load_prompt,
)

VALID_JSON = (
    '{"resposta":"Carrega em cerca de 6 horas.","energia_kwh":40.0,'
    '"custo_estimado_reais":18.0,"tempo_horas":6.0,"dados_faltantes":[],'
    '"requer_validacao_engenheiro":false}'
)


class FakeModel:
    def __init__(self, name, resposta, erro=None):
        self.model = name
        self.resposta = resposta
        self.erro = erro
        self.calls = 0

    def invoke(self, _prompt_value):
        self.calls += 1
        if self.erro is not None:
            raise self.erro
        return AIMessage(content=self.resposta)


class MultiProviderTest(unittest.TestCase):
    def test_executa_todas_as_combinacoes_modelo_x_prompt(self):
        bom = FakeModel("modelo-bom", VALID_JSON)
        fraco = FakeModel("modelo-fraco", "resposta curta sem contexto tecnico")
        demo = MultiProviderDemo(
            models=[bom, fraco],
            prompt_versions=["v2.1", "v2.2"],
        )
        resultado = demo.executar("Quanto tempo leva para carregar 40 kWh?", modo="alternado")

        self.assertEqual(len(resultado["execucoes"]), 4)
        self.assertEqual(bom.calls, 2)
        self.assertEqual(fraco.calls, 2)
        self.assertEqual(
            { (e["modelo"], e["prompt"]) for e in resultado["execucoes"] },
            {
                ("modelo-bom", "v2.1"),
                ("modelo-bom", "v2.2"),
                ("modelo-fraco", "v2.1"),
                ("modelo-fraco", "v2.2"),
            },
        )

    def test_consolida_melhor_resposta_por_score(self):
        bom = FakeModel("modelo-bom", VALID_JSON)
        fraco = FakeModel("modelo-fraco", "sem schema e sem vocabulário")
        demo = MultiProviderDemo(models=[bom, fraco], prompt_versions=["v2.2"])

        resultado = demo.executar("Qual o custo de carregar uma bateria de 60 kWh?", modo="paralelo")

        execucoes_fraco = [
            e for e in resultado["execucoes"] if e["modelo"] == "modelo-fraco"
        ]
        self.assertEqual(resultado["melhor"]["modelo"], "modelo-bom")
        self.assertTrue(resultado["melhor"]["schema_valid"])
        for execucao in execucoes_fraco:
            self.assertLess(execucao["score"], resultado["melhor"]["score"])
        self.assertIn("6 horas", resultado["resposta_consolidada"])

    def test_modo_paralelo_executa_todos_os_combos(self):
        a = FakeModel("modelo-a", VALID_JSON)
        b = FakeModel("modelo-b", VALID_JSON)
        demo = MultiProviderDemo(models=[a, b], prompt_versions=["v2.1", "v2.2"])

        resultado = demo.executar("Pergunta de teste?", modo="paralelo")

        self.assertEqual(resultado["modo"], "paralelo")
        self.assertEqual(len(resultado["execucoes"]), 4)
        self.assertEqual(a.calls, 2)
        self.assertEqual(b.calls, 2)

    def test_modo_invalido_levanta_erro(self):
        demo = MultiProviderDemo(models=[FakeModel("m", VALID_JSON)], prompt_versions=["v2.2"])
        with self.assertRaises(ValueError):
            demo.executar("Pergunta?", modo="sequencial")

    def test_modelo_que_lanca_erro_nao_impede_as_demais(self):
        erro = FakeModel("modelo-erro", "", erro=RuntimeError("falha"))
        bom = FakeModel("modelo-bom", VALID_JSON)
        demo = MultiProviderDemo(models=[erro, bom], prompt_versions=["v2.2"])

        resultado = demo.executar("Pergunta?", modo="alternado")

        self.assertEqual(len(resultado["execucoes"]), 2)
        faltando = resultado["execucoes"][0]
        self.assertEqual(faltando["erro"], "RuntimeError: falha")
        self.assertEqual(faltando["score"], 0.0)
        self.assertEqual(resultado["melhor"]["modelo"], "modelo-bom")
        self.assertNotEqual(resultado["resposta_consolidada"], "")
        self.assertFalse(resultado["resposta_consolidada"].startswith("Nenhuma combinação"))

    def test_prompt_inexistente_levanta_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("v9.9")

    def test_defaults_sao_coerentes(self):
        self.assertIn("gpt-oss:120b", DEFAULT_MODELS)
        self.assertIn("v2.2", DEFAULT_PROMPT_VERSIONS)

    def test_versao_prompt_com_nome_de_arquivo_tambem_e_aceita(self):
        demo = MultiProviderDemo(
            models=[FakeModel("m", VALID_JSON)],
            prompt_versions=["system_prompt_v2.2.txt"],
        )
        resultado = demo.executar("Pergunta?", modo="alternado")
        self.assertEqual(len(resultado["execucoes"]), 1)
        self.assertEqual(resultado["execucoes"][0]["prompt"], "system_prompt_v2.2.txt")


if __name__ == "__main__":
    unittest.main()