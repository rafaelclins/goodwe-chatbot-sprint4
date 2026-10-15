import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
	allowed: bool
	reason: str
	category: str


_CATEGORY_PATTERNS = (
	(
		"charging",
		(r"\b(recarga|carregamento|carregar|veiculo eletrico|veiculos eletricos)\b",),
	),
	(
		"cost_sharing",
		(r"\b(rateio|divisao de custos?|custo|cobrança|cobranca|valor da recarga)\b",),
	),
	(
		"energy_consumption",
		(r"\b(energia|consumo|kwh|kw|bateria|tarifa)\b",),
	),
	(
		"electrical_infrastructure",
		(
			r"\b(infraestrutura eletrica|instalacao eletrica|potencia|demanda|"
			r"quadro|disjuntor|protecao|aterramento|cabos?|tensao|corrente|dr)\b",
		),
	),
	(
		"charger_operation",
		(r"\b(carregador(?:es)?|wallbox|ponto de recarga|operacao|operar|conector)\b",),
	),
)

_SEMANTIC_PATTERNS = {
	"charging": (
		(r"\b(?:calcular|calculo)\b.*\brecarga\b", 2),
	),
	"cost_sharing": (
		(r"\brateio\b", 5),
		(r"\bquanto\s+custa\b", 4),
		(r"\bcusto\b", 2),
		(r"\bdivisao\s+de\s+custos?\b", 5),
	),
	"energy_consumption": (
		(r"\bcalcule\s+apenas\s+a\s+energia\b", 6),
		(r"\bconsumo\s+de\s+energia\b", 3),
	),
	"electrical_infrastructure": (
		(r"\bdr\b", 4),
		(r"\bdisjuntor(?:es)?\b", 4),
		(r"\baterramento\b", 4),
		(r"\bprote[cç][aã]o\b", 4),
		(r"\binfraestrutura\s+eletrica\b", 4),
		(r"\bdemanda\s+contratada\b", 4),
	),
	"charger_operation": (
		(r"\bcomo\s+operar\b", 4),
		(r"\bopera[cç][aã]o\b", 3),
	),
}

_PRODUCT_DETAIL_PATTERNS = (
	r"\bpotencia\s+(?:nominal\s+)?exata\b",
	r"\bcorrente\s+(?:maxima|nominal)\s+exata\b",
	r"\bqual\s+(?:a\s+)?norma\s+especifica\b",
	r"\b(?:ficha tecnica|datasheet|especificacao tecnica)\b",
	r"\bmodelo\s+[a-z0-9][\w-]*.*\b(?:potencia|corrente|norma|especificacao)\b",
)

_LEGAL_ADVICE_PATTERNS = (
	r"\b(?:qual|que)\s+decisao\s+juridica\b",
	r"\b(?:aconselhamento|orientacao|assessoria)\s+juridica\b",
	r"\bcomo\s+(?:devo\s+)?estruturar\s+(?:juridicamente|legalmente)\b",
	r"\bcontrato\s+(?:devo|deve)\s+(?:usar|fazer|assinar)\b",
)

_FINANCIAL_ADVICE_PATTERNS = (
	r"\b(?:aconselhamento|orientacao|assessoria)\s+financeira\b",
	r"\bquanto\s+devo\s+cobrar\b",
	r"\bqual\s+valor\s+devo\s+cobrar\b",
	r"\bcomo\s+(?:devo\s+)?investir\b",
)

_ELECTRICAL_SAFETY_PATTERNS = (
	r"\bposso\s+instalar\b.*\bsem\s+(?:engenheiro|eletricista|profissional)\b",
	r"\b(?:essa|esta)\s+instalacao\s+e\s+segura\b",
	r"\bqual\s+disjuntor\s+devo\s+(?:comprar|instalar|usar)\b",
	r"\bposso\s+ligar\b.*\b(?:sem|direto)\b",
)


def _normalize(text: str) -> str:
	text = unicodedata.normalize("NFKD", text)
	text = "".join(char for char in text if not unicodedata.combining(char))
	return re.sub(r"\s+", " ", text.casefold()).strip()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
	return any(re.search(pattern, text) for pattern in patterns)


def _category_score(normalized: str, category: str, patterns: tuple[str, ...]) -> int:
	score = 1 if _matches_any(normalized, patterns) else 0
	for pattern, weight in _SEMANTIC_PATTERNS.get(category, ()):
		if re.search(pattern, normalized):
			score += weight
	return score


def validate_scope(text: str) -> GuardrailResult:
	normalized = _normalize(text)

	if _matches_any(normalized, _LEGAL_ADVICE_PATTERNS):
		return GuardrailResult(False, "aconselhamento juridico recusado", "legal_advice")
	if _matches_any(normalized, _FINANCIAL_ADVICE_PATTERNS):
		return GuardrailResult(False, "aconselhamento financeiro recusado", "financial_advice")
	if _matches_any(normalized, _ELECTRICAL_SAFETY_PATTERNS):
		return GuardrailResult(
			False,
			"decisao de seguranca eletrica exige profissional",
			"electrical_safety_advice",
		)
	if _matches_any(normalized, _PRODUCT_DETAIL_PATTERNS):
		return GuardrailResult(
			False,
			"especificacao exata nao esta disponivel na base",
			"unsupported_product_detail",
		)

	scored_categories = [
		(category, _category_score(normalized, category, patterns))
		for category, patterns in _CATEGORY_PATTERNS
	]
	category, score = max(
		scored_categories,
		key=lambda item: item[1],
	)
	if score > 0:
		return GuardrailResult(True, "pergunta dentro do dominio", category)

	return GuardrailResult(False, "pergunta fora do dominio permitido", "out_of_scope")
