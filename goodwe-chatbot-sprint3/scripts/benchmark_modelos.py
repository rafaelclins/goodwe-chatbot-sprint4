"""Benchmark de modelos via Ollama Cloud / LangChain.

Compara 2 modelos generativos executando as mesmas perguntas de
`evals/sprint3_cases.json`, com hiperparâmetros de geração fixos, e
apresenta métricas de latência, tokens, aderência ao schema Pydantic v2
(ConsultaRecarga) e coerência no domínio GoodWe.

Uso:
    python -m scripts.benchmark_modelos
"""

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.consulta_recarga import ConsultaRecarga


CASES_PATH = PROJECT_ROOT / "evals" / "sprint3_cases.json"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt_v2.2.txt"
RESULTS_PATH = PROJECT_ROOT / "evals" / "benchmark_modelos_results.json"

HYPERPARAMETROS: dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 500,
}

MODELS = (
    "gemma4:cloud",
    "gpt-oss:20b",
)

NUM_QUESTIONS = 5
QUESTION_IDS = (
    "happy_kwh_concept",
    "calculation_charge_cost",
    "missing_calculation_data",
    "conceptual_dr",
    "ignore_false_positive",
)

_DOMAIN_KEYWORDS = (
    "recarga",
    "carregador",
    "carregamento",
    "ev",
    "veiculo",
    "eletrico",
    "kwh",
    "kw",
    "energia",
    "bateria",
    "infraestrutura",
    "eletrica",
    "disjuntor",
    "dr",
    "aterramento",
    "demanda",
    "quadro",
    "proteção",
    "protecao",
    "custo",
    "tarifa",
    "rateio",
    "potencia",
    "potência",
    "corrente",
    "condominio",
    "condomínio",
    "engenheiro",
    "eletricista",
    "instalação",
    "instalacao",
)

_MISSING_DATA_KEYWORDS = ("informe", "forneça", "forneca", "dado", "capacidade", "percentual", "necessário", "necessario")

_ACCENT_FREE = {"á", "ã", "é", "ê", "í", "ó", "ô", "õ", "ú", "ç"}


def _normalize(text: str) -> str:
    return text.casefold()


def _is_portuguese(text: str) -> bool:
    tokens = re.findall(r"[a-zà-úâãçêôõ]{3,}", _normalize(text))
    if not tokens:
        return False
    return sum(1 for token in tokens if any(char in _ACCENT_FREE for char in token)) / len(tokens) >= 0.03


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def select_questions() -> list[dict[str, Any]]:
    cases = {case["id"]: case for case in load_cases()}
    selected: list[dict[str, Any]] = []
    for question_id in QUESTION_IDS:
        case = cases[question_id]
        if not case.get("expected_allowed", False):
            continue
        selected.append(case)
    return selected


def count_tokens(text: str, encoding: Any) -> int:
    return len(encoding.encode(text))


def build_chat_model(model_name: str) -> ChatOllama:
    load_dotenv()
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY não configurada no .env")

    return ChatOllama(
        model=model_name,
        base_url="https://ollama.com",
        temperature=HYPERPARAMETROS["temperature"],
        top_p=HYPERPARAMETROS["top_p"],
        num_predict=HYPERPARAMETROS["max_tokens"],
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {api_key}",
            }
        },
    )


def build_prompt_messages(question: str, encoding: Any) -> list[BaseMessage]:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    parser = PydanticOutputParser(pydantic_object=ConsultaRecarga)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}\n\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt.format_messages(input=question)


def extract_json_object(content: str) -> dict[str, Any] | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = content[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def parse_schema(content: str) -> dict[str, Any]:
    parsed = extract_json_object(content)
    if parsed is None:
        return {"valid": False, "error": "JSON não encontrado na resposta"}
    try:
        ConsultaRecarga.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"valid": True, "error": None}


def domain_coherence(question: str, content: str) -> dict[str, Any]:
    response = content.strip()
    if not response:
        return {"score": 0.0, "label": "vazia", "detalhes": "resposta vazia"}

    normalized_response = _normalize(response)
    normalized_question = _normalize(question)

    score = 0.0
    score += 2.0 if _is_portuguese(response) else 0.0

    keyword_hits = sum(
        1 for keyword in _DOMAIN_KEYWORDS if keyword in normalized_response
    )
    score += min(keyword_hits, 5) * 1.2

    json_present = extract_json_object(response) is not None
    if json_present:
        score += 0.5

    requires_data = any(kw in normalized_question for kw in _MISSING_DATA_KEYWORDS)
    if requires_data and any(kw in normalized_response for kw in _MISSING_DATA_KEYWORDS):
        score += 1.0

    if "advogado" in normalized_response or "juríd" in normalized_response:
        score -= 2.0

    score = max(0.0, min(round(score, 1), 10.0))

    if score >= 7.0:
        label = "coerente"
    elif score >= 4.0:
        label = "parcial"
    else:
        label = "fraca"

    return {
        "score": score,
        "label": label,
        "detalhes": (
            f"idioma_portugues={_is_portuguese(response)}; "
            f"termos_dominio={keyword_hits}; json_presente={json_present}"
        ),
    }


def run_single_question(
    model: ChatOllama,
    question: str,
    encoding: Any,
) -> dict[str, Any]:
    messages = build_prompt_messages(question, encoding)
    input_tokens = sum(count_tokens(message.content, encoding) for message in messages)

    started_ns = time.perf_counter_ns()
    try:
        response: BaseMessage = model.invoke(messages)
        error = None
    except Exception as exc:  # noqa: BLE001
        response = None
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

    if response is None or not response.content:
        return {
            "pergunta": question,
            "latencia_s": latency_ms / 1000,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "error": error,
            "json_schema": {"valid": False, "error": "sem resposta"},
            "coerencia": {"score": 0.0, "label": "vazia", "detalhes": "sem resposta"},
        }

    content = response.content
    output_tokens = count_tokens(content, encoding)

    metadata = response.response_metadata or {}
    return {
        "pergunta": question,
        "latencia_s": latency_ms / 1000,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "eval_count": metadata.get("eval_count"),
        "error": error,
        "resposta_previa": content[:200],
        "json_schema": parse_schema(content),
        "coerencia": domain_coherence(question, content),
    }


def run_benchmark() -> dict[str, Any]:
    encoding = tiktoken.get_encoding("cl100k_base")
    questions = select_questions()

    per_model: dict[str, list[dict[str, Any]]] = {}
    for model_name in MODELS:
        print(f"=== Rodando modelo: {model_name} ===")
        model = build_chat_model(model_name)
        runs = []
        for question in questions:
            run = run_single_question(model, question["input"], encoding)
            runs.append(run)
            status = "OK" if run["json_schema"]["valid"] else "JSON inválido"
            print(
                f"  [{question['id']}] {question['input'][:60]}... "
                f"lat={run['latencia_s']:.1f}s tokens={(run['input_tokens'], run['output_tokens'])} "
                f"{status}"
            )
        per_model[model_name] = runs
        print()

    return aggregate_results(per_model)


def aggregate_results(per_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    aggregated: dict[str, dict[str, Any]] = {}
    for model_name, runs in per_model.items():
        latencies = [run["latencia_s"] for run in runs]
        input_tokens = [run["input_tokens"] for run in runs]
        output_tokens = [run["output_tokens"] for run in runs]
        schema_valid = sum(1 for run in runs if run["json_schema"]["valid"])
        scores = [run["coerencia"]["score"] for run in runs]
        errors = [run["error"] for run in runs if run.get("error")]

        aggregated[model_name] = {
            "hiperparametros": HYPERPARAMETROS,
            "n_perguntas": len(runs),
            "media_latencia_s": round(statistics.mean(latencies), 2),
            "media_tokens_entrada": round(statistics.mean(input_tokens), 1),
            "media_tokens_saida": round(statistics.mean(output_tokens), 1),
            "total_tokens_entrada": sum(input_tokens),
            "total_tokens_saida": sum(output_tokens),
            "acuracia_json_pct": round(schema_valid / len(runs) * 100, 1),
            "media_coerencia": round(statistics.mean(scores), 1),
            "erros": errors,
            "execucoes": runs,
        }
    return {"hiperparametros": HYPERPARAMETROS, "modelos": aggregated}


def display(results: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("RESULTADOS DO BENCHMARK - OLLAMA CLOUD / LANGCHAIN")
    print("=" * 78)
    print(f"Hiperparâmetros de geração: {results['hiperparametros']}")
    print()

    header = (
        f"{'Modelo':<16} {'Lat(ms)':>8} {'Tok in':>8} {'Tok out':>8} "
        f"{'JSON%':>7} {'Coerência':>10}"
    )
    print(header)
    print("-" * len(header))
    for model_name, metrics in results["modelos"].items():
        print(
            f"{model_name:<16} "
            f"{metrics['media_latencia_s'] * 1000:>8.0f} "
            f"{metrics['media_tokens_entrada']:>8.1f} "
            f"{metrics['media_tokens_saida']:>8.1f} "
            f"{metrics['acuracia_json_pct']:>7.1f} "
            f"{metrics['media_coerencia']:>8.1f}/10"
        )
    print()

    for model_name, metrics in results["modelos"].items():
        print(f"--- {model_name} ---")
        for run in metrics["execucoes"]:
            print(
                f"  pergunta: {run['pergunta'][:70]}"
            )
            schema = run["json_schema"]
            schema_text = "válido no schema" if schema["valid"] else f"inválido: {schema['error']}"
            coherence = run["coerencia"]
            print(
                f"    lat={run['latencia_s']:.1f}s in={run['input_tokens']} out={run['output_tokens']} "
                f"| schema: {schema_text} | coerência: {coherence['score']}/10 ({coherence['label']})"
            )


def main() -> None:
    results = run_benchmark()
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    display(results)
    print(f"\nResultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()