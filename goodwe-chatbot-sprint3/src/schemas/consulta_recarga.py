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

	@field_validator("energia_kwh")
	@classmethod
	def validar_energia(cls, value: float | None) -> float | None:
		if value is None:
			return value
		if value <= 0:
			raise ValueError("energia_kwh deve ser um valor positivo; a energia não pode ser zero ou negativa")
		if value < 0.1:
			raise ValueError("energia_kwh deve ser um valor razoável de recarga de EV (mínimo de 0,1 kWh)")
		if value > 1000:
			raise ValueError("energia_kwh excede um valor razoável de recarga de EV (máximo de 1000 kWh)")
		return value

	@field_validator("tempo_horas")
	@classmethod
	def validar_tempo(cls, value: float | None) -> float | None:
		if value is None:
			return value
		if value <= 0:
			raise ValueError("tempo_horas deve ser um valor positivo; o tempo não pode ser zero ou negativo")
		if value > 168:
			raise ValueError("tempo_horas excede uma semana (168 h), valor não esperado para uma recarga de EV")
		return value