import json
import time
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.chain.memoria import build_memory_chain, get_session_history
from src.guardrails.moderation import moderate
from src.guardrails.scope_validator import validate_scope
from src.schemas.consulta_recarga import ConsultaRecarga


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "sprint3_cases.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "sprint3_results.json"

VALID_JSON = (
	'{"resposta":"Resposta fake para avaliação.",'
	'"energia_kwh":12.5,"custo_estimado_reais":10.0,'
	'"tempo_horas":0.57,"dados_faltantes":[],'
	'"requer_validacao_engenheiro":true}'
)
_DEFAULT_METADATA = object()


class FakeModel:
	def __init__(
		self,
		response_text: str = VALID_JSON,
		error: Exception | None = None,
		response_metadata: Any = _DEFAULT_METADATA,
		usage_metadata: Any = _DEFAULT_METADATA,
	):
		self.calls = 0
		self.response_text = response_text
		self.error = error
		self.response_metadata = response_metadata
		self.usage_metadata = usage_metadata

	def invoke(self, _prompt_value: Any) -> AIMessage:
		self.calls += 1
		if self.error is not None:
			raise self.error
		response_metadata = self.response_metadata
		if response_metadata is _DEFAULT_METADATA:
			response_metadata = {
				"model": "fake-model",
				"total_duration": 1000000,
				"prompt_eval_count": 11,
				"eval_count": 7,
			}
		usage_metadata = self.usage_metadata
		if usage_metadata is _DEFAULT_METADATA:
			usage_metadata = {
				"input_tokens": 10,
				"output_tokens": 8,
				"total_tokens": 18,
			}
		return AIMessage(
			content=self.response_text,
			response_metadata=response_metadata,
			usage_metadata=usage_metadata,
		)

	def get_num_tokens_from_messages(self, messages: list[Any]) -> int:
		return sum(self.get_num_tokens(message.content) for message in messages)

	def get_num_tokens(self, text: str) -> int:
		return len(text.split())


class InstrumentedModel:
	def __init__(self, model: Any):
		self.model = model
		self.calls = 0
		self.model_latency_ms: float | None = None
		self.response_metadata: dict[str, Any] = {}
		self.usage_metadata: dict[str, Any] = {}
		self.token_metrics = {
			"input_tokens": 0,
			"output_tokens": 0,
			"turn_tokens": 0,
			"source": "estimated",
		}

	def invoke(self, prompt_value: Any) -> AIMessage:
		self.calls += 1
		started_ns = time.perf_counter_ns()
		try:
			message = self.model.invoke(prompt_value)
		finally:
			self.model_latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

		if not isinstance(message, AIMessage):
			raise TypeError("O modelo instrumentado deve retornar AIMessage")

		self.response_metadata = dict(message.response_metadata or {})
		self.usage_metadata = dict(message.usage_metadata or {})
		self.token_metrics = _extract_token_metrics(
			prompt_value,
			message,
			self.model,
		)
		return message


def _extract_token_metrics(prompt_value: Any, message: AIMessage, model: Any) -> dict[str, Any]:
	usage = message.usage_metadata or {}
	if all(key in usage for key in ("input_tokens", "output_tokens")):
		input_tokens = usage["input_tokens"]
		output_tokens = usage["output_tokens"]
		return {
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"turn_tokens": usage.get("total_tokens", input_tokens + output_tokens),
			"source": "usage_metadata",
		}

	metadata = message.response_metadata or {}
	if "prompt_eval_count" in metadata and "eval_count" in metadata:
		input_tokens = metadata["prompt_eval_count"]
		output_tokens = metadata["eval_count"]
		return {
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"turn_tokens": input_tokens + output_tokens,
			"source": "ollama_metadata",
		}

	messages = getattr(prompt_value, "messages", [])
	input_tokens = model.get_num_tokens_from_messages(messages)
	output_tokens = model.get_num_tokens(message.content)
	return {
		"input_tokens": input_tokens,
		"output_tokens": output_tokens,
		"turn_tokens": input_tokens + output_tokens,
		"source": "estimated",
	}


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
	return json.loads(path.read_text(encoding="utf-8"))


def build_fake_structured_chain(instrumented_model: InstrumentedModel):
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", "Resposta de teste."),
			MessagesPlaceholder(variable_name="history", optional=True),
			("human", "{input}"),
		]
	)
	raw_chain = prompt | RunnableLambda(instrumented_model.invoke) | StrOutputParser()
	parser = PydanticOutputParser(pydantic_object=ConsultaRecarga)
	return build_memory_chain(
		chain=raw_chain,
		output_parser=parser,
	)


def get_guardrail_result(user_input: str) -> dict[str, Any]:
	moderation_result = moderate(user_input)
	if not moderation_result.allowed:
		return {
			"allowed": moderation_result.allowed,
			"reason": moderation_result.reason,
			"category": moderation_result.category,
		}

	scope_result = validate_scope(user_input)
	return {
		"allowed": scope_result.allowed,
		"reason": scope_result.reason,
		"category": scope_result.category,
	}


def run_cases(
	cases: list[dict[str, Any]] | None = None,
	fake_model: FakeModel | None = None,
) -> list[dict[str, Any]]:
	cases = load_cases() if cases is None else cases
	fake_model = FakeModel() if fake_model is None else fake_model
	instrumented_model = InstrumentedModel(fake_model)
	chain = build_fake_structured_chain(instrumented_model)
	results = []

	for case in cases:
		session_id = f"eval-{case['id']}"
		get_session_history(session_id).clear()
		calls_before = instrumented_model.calls
		result: Any = None
		error = None
		error_exception: Exception | None = None

		guardrail_result = get_guardrail_result(case["input"])
		started_ns = time.perf_counter_ns()
		try:
			from src.guardrails.pipeline import process_user_input

			result = process_user_input(case["input"], session_id, chain)
		except Exception as exc:
			error_exception = exc
			error = f"{type(exc).__name__}: {exc}"
		total_latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

		model_called = instrumented_model.calls > calls_before
		output_type = type(result).__name__ if result is not None else None
		blocked = not guardrail_result["allowed"]
		parse_error = isinstance(error_exception, (OutputParserException, ValidationError))
		error_stage = None
		if blocked:
			error_stage = "guardrail_block"
		elif parse_error:
			error_stage = "structured_output_parsing"
		elif error is not None:
			error_stage = "model_error"
		checks = {
			"allowed": guardrail_result["allowed"] == case["expected_allowed"],
			"category": guardrail_result["category"] == case["expected_category"],
			"model_call": model_called == case["expected_model_call"],
			"output_type": output_type == case["expected_output_type"],
		}
		if case["expected_allowed"]:
			checks["pydantic_output"] = isinstance(result, ConsultaRecarga)
		else:
			checks["blocked_without_history"] = get_session_history(session_id).messages == []

		results.append(
			{
				"id": case["id"],
				"input": case["input"],
				"guardrail_result": guardrail_result,
				"model_called": model_called,
				"output_type": output_type,
				"blocked": blocked,
				"parsing_attempted": model_called and not blocked and error_stage != "model_error",
				"structured_output_valid": isinstance(result, ConsultaRecarga),
				"parse_error": parse_error,
				"error_stage": error_stage,
				"total_latency_ms": total_latency_ms,
				"model_latency_ms": 0 if blocked else instrumented_model.model_latency_ms,
				"response_metadata": {} if blocked else instrumented_model.response_metadata,
				"usage_metadata": {} if blocked else instrumented_model.usage_metadata,
				"tokens": {
					"input_tokens": 0 if blocked else instrumented_model.token_metrics["input_tokens"],
					"output_tokens": 0 if blocked else instrumented_model.token_metrics["output_tokens"],
					"turn_tokens": 0 if blocked else instrumented_model.token_metrics["turn_tokens"],
					"source": "none" if blocked else instrumented_model.token_metrics["source"],
				},
				"success": error is None and all(checks.values()),
				"checks": checks,
				"error": error,
			}
		)

	return results


def save_results(
	results: list[dict[str, Any]],
	path: Path = RESULTS_PATH,
) -> Path:
	payload = {
		"summary": {
			"total": len(results),
			"passed": sum(1 for result in results if result["success"]),
			"failed": sum(1 for result in results if not result["success"]),
		},
		"results": [
			{
				"id": result["id"],
				"pergunta": result["input"],
				"status": "permitido" if not result["blocked"] else "bloqueado",
				"categoria": result["guardrail_result"]["category"],
				"validacao_schema": result["structured_output_valid"],
				"passou_criterio": result["success"],
				"checks": result["checks"],
			}
			for result in results
		],
	}
	path.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	return path


def main() -> None:
	results = run_cases()
	saved_path = save_results(results)
	print(json.dumps(results, ensure_ascii=False, indent=2))
	print(f"\nResultados salvos em: {saved_path}")


if __name__ == "__main__":
	main()