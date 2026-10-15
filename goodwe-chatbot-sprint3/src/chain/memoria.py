from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chain.builder import build_chain
from src.schemas.consulta_recarga import ConsultaRecarga


DEFAULT_MAX_HISTORY_TOKENS = 1000
_SESSION_HISTORIES: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
	if session_id not in _SESSION_HISTORIES:
		_SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()
	return _SESSION_HISTORIES[session_id]


def _trim_session_history(session_id: str, max_history_tokens: int) -> None:
	history = get_session_history(session_id)
	history.messages = list(
		trim_messages(
			history.messages,
			max_tokens=max_history_tokens,
			token_counter="approximate",
			strategy="last",
			start_on="human",
			allow_partial=False,
		)
	)


def _prepare_input(input_data: Any, config: dict, max_history_tokens: int) -> Any:
	session_id = config.get("configurable", {}).get("session_id")
	if session_id is not None:
		_trim_session_history(session_id, max_history_tokens)
	return input_data


def _trim_and_return(output: Any, config: dict, max_history_tokens: int) -> Any:
	session_id = config.get("configurable", {}).get("session_id")
	if session_id is not None:
		_trim_session_history(session_id, max_history_tokens)
	return output


def build_memory_chain(
	max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
	chain=None,
	output_parser=None,
):
	if max_history_tokens <= 0:
		raise ValueError("max_history_tokens deve ser maior que zero")

	base_chain = chain if chain is not None else build_chain()
	chain_with_history = RunnableWithMessageHistory(
		base_chain,
		get_session_history,
		input_messages_key="input",
		history_messages_key="history",
	)

	memory_chain = RunnableLambda(
		lambda input_data, config: _prepare_input(
			input_data, config, max_history_tokens
		)
	) | chain_with_history | RunnableLambda(
		lambda output, config: _trim_and_return(output, config, max_history_tokens)
	)
	if output_parser is not None:
		return memory_chain | output_parser
	return memory_chain


def build_structured_memory_chain(
	max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
	model=None,
):
	parser = PydanticOutputParser(pydantic_object=ConsultaRecarga)
	raw_chain = build_chain(
		model=model,
		format_instructions=parser.get_format_instructions(),
	)
	return build_memory_chain(
		max_history_tokens=max_history_tokens,
		chain=raw_chain,
		output_parser=parser,
	)
