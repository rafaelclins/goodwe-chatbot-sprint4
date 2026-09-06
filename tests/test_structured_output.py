import unittest

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.chain.builder import build_structured_chain
from src.chain.memoria import build_structured_memory_chain
from src.schemas.consulta_recarga import ConsultaRecarga


VALID_JSON = (
    '{"resposta":"Verifique a capacidade elétrica do condomínio.",'
    '"energia_kwh":12.5,"custo_estimado_reais":10.0,'
    '"tempo_horas":0.57,"dados_faltantes":[],'
    '"requer_validacao_engenheiro":true}'
)


class StructuredOutputTest(unittest.TestCase):
    def test_valid_data_returns_pydantic_instance(self):
        model = RunnableLambda(lambda _: VALID_JSON)
        result = build_structured_chain(model=model).invoke(
            {"input": "Avalie esta recarga."}
        )

        self.assertIsInstance(result, ConsultaRecarga)
        self.assertEqual(result.energia_kwh, 12.5)

    def test_field_validator_rejects_blank_response(self):
        with self.assertRaises(ValidationError):
            ConsultaRecarga(resposta="   ")

    def test_structured_memory_chain_preserves_schema_type(self):
        model = RunnableLambda(lambda _: VALID_JSON)
        chain = build_structured_memory_chain(
            model=model,
            max_history_tokens=100,
        )

        result = chain.invoke(
            {"input": "Qual cuidado devo tomar?"},
            config={"configurable": {"session_id": "structured-session"}},
        )

        self.assertIsInstance(result, ConsultaRecarga)


if __name__ == "__main__":
    unittest.main()