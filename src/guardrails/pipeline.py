from typing import Any

from src.chain.memoria import build_structured_memory_chain
from src.guardrails.moderation import moderate
from src.guardrails.scope_validator import GuardrailResult, validate_scope


_LOCAL_RESPONSES = {
	"ignore_rules": "Não posso ignorar as regras do assistente.",
	"replace_system_prompt": "Não posso alterar as instruções do assistente.",
	"reveal_internal_instructions": "Não posso revelar instruções internas.",
	"privilege_impersonation": "Não posso alterar privilégios ou função do assistente.",
	"scope_redefinition": "Não posso alterar o escopo do assistente.",
	"internal_reasoning": "Não posso revelar raciocínio interno.",
	"disable_restrictions": "Não posso desativar restrições de segurança.",
	"out_of_scope": "Essa pergunta está fora do escopo de recarga e infraestrutura elétrica em condomínios.",
	"unsupported_product_detail": "Não há dados disponíveis para confirmar essa especificação exata.",
	"legal_advice": "Não posso orientar decisões jurídicas. Consulte um profissional habilitado.",
	"financial_advice": "Não posso orientar decisões financeiras. Consulte um profissional habilitado.",
	"electrical_safety_advice": "Não posso garantir segurança elétrica. Consulte um engenheiro eletricista.",
}


def _blocked_response(result: GuardrailResult) -> str:
	return _LOCAL_RESPONSES.get(
		result.category,
		"Não posso processar essa solicitação.",
	)


def process_user_input(
	user_input: str,
	session_id: str,
	chain: Any = None,
) -> Any:
	moderation_result = moderate(user_input)
	if not moderation_result.allowed:
		return _blocked_response(moderation_result)

	scope_result = validate_scope(user_input)
	if not scope_result.allowed:
		return _blocked_response(scope_result)

	if chain is None:
		chain = build_structured_memory_chain()

	return chain.invoke(
		{"input": user_input},
		config={"configurable": {"session_id": session_id}},
	)