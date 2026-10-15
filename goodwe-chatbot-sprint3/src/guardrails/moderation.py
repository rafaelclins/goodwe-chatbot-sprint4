import re
import unicodedata

from src.guardrails.scope_validator import GuardrailResult


_INJECTION_PATTERNS = (
	(
		"ignore_rules",
		r"\b(?:ignore|desconsidere)\b.{0,50}\b(?:todas?\s+as?\s+)?"
		r"(?:regras|instrucoes|restricoes|limites)\b",
	),
	(
		"replace_system_prompt",
		r"\b(?:substitua|altere|redefina|reescreva|mude|troque)\b.{0,60}\b"
		r"(?:system\s+prompt|prompt\s+do\s+sistema|instrucoes\s+do\s+sistema)\b",
	),
	(
		"reveal_internal_instructions",
		r"\b(?:revele|mostre|exiba|imprima|vaze)\b.{0,60}\b(?:system\s+prompt|"
		r"prompt\s+do\s+sistema|instrucoes\s+internas|instrucoes\s+ocultas|"
		r"regras\s+internas)\b",
	),
	(
		"privilege_impersonation",
		r"\b(?:sou|finja\s+ser|aja\s+como|assuma\s+o\s+papel\s+de)\b.{0,40}\b"
		r"(?:desenvolvedor|developer|admin|administrador|sistema)\b",
	),
	(
		"scope_redefinition",
		r"\b(?:agora\s+voce\s+e|voce\s+agora\s+e|mude\s+sua\s+funcao|"
		r"ignore\s+o\s+escopo|responda\s+sobre\s+qualquer\s+assunto)\b|"
		r"\b(?:sua|esta)\s+(?:unica\s+)?funcao\s+e\b|"
		r"\b(?:unica|unico)\s+funcao\b|"
		r"\bvoce\s+e\s+(?:um|uma|o|a)\s+(?:classificador|assistente|bot|"
		r"agente|especialista)\b|"
		r"\b(?:responda|responder|reply|answer)\s+(?:apenas|somente|so)\b"
		r".{0,40}\b(?:palavra|word|motivo)\b",
	),
	(
		"internal_reasoning",
		r"\b(?:mostre|revele|exiba|diga|imprima)\b.{0,50}\b(?:cadeia|"
		r"raciocinio|pensamento)\b.{0,35}\b(?:interno|oculto|privado|passo\s+a\s+passo)\b",
	),
	(
		"disable_restrictions",
		r"\b(?:desative|remova|contorne|bypass|ignore)\b.{0,40}\b(?:restricoes|"
		r"filtros|guardrails|limites|seguranca)\b",
	),
)


def _normalize(text: str) -> str:
	text = unicodedata.normalize("NFKD", text)
	text = "".join(char for char in text if not unicodedata.combining(char))
	return re.sub(r"\s+", " ", text.casefold()).strip()


def moderate(text: str) -> GuardrailResult:
	normalized = _normalize(text)
	for category, pattern in _INJECTION_PATTERNS:
		if re.search(pattern, normalized):
			return GuardrailResult(False, "tentativa de manipulacao detectada", category)

	return GuardrailResult(True, "nenhuma manipulacao detectada", "allowed")
