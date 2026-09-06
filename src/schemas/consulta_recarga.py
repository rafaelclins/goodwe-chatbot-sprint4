from pydantic import BaseModel, Field, field_validator


class ConsultaRecarga(BaseModel):
	resposta: str = Field(min_length=1)
	energia_kwh: float | None = Field(default=None, ge=0)
	custo_estimado_reais: float | None = Field(default=None, ge=0)
	tempo_horas: float | None = Field(default=None, ge=0)
	dados_faltantes: list[str] = Field(default_factory=list)
	requer_validacao_engenheiro: bool = False

	@field_validator("resposta")
	@classmethod
	def validar_resposta(cls, value: str) -> str:
		resposta = value.strip()
		if not resposta:
			raise ValueError("resposta não pode ser vazia")
		return resposta