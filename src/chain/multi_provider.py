"""Demonstração opcional de chamada multi-provider (bônus).

Consulta o domínio de recarga de veículos elétricos usando mais de um modelo e
mais de uma versão de system prompt, comparando as respostas e retornando a
melhor resposta consolidada junto da comparação entre as execuções.

Uso pelo CLI (chamadas reais ao Ollama):

    python -m src.chain.multi_provider
    python -m src.chain.multi_provider --question "Quanto custa carregar 40 kWh?" \
        --models gpt-oss:120b qwen3:8b --prompts v2.1 v2.2 --mode paralelo --output evals/multi_provider_results.json

Se OLLAMA_API_KEY estiver no .env, as chamadas usam o Ollama Cloud; caso
contrário, tentam o Ollama local (http://localhost:11434).
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import tiktoken
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.schemas.consulta_recarga import ConsultaRecarga

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DEFAULT_MODELS = (
    "gpt-oss:120b",
    "gpt-oss:20b",
)
DEFAULT_PROMPT_VERSIONS = (
    "v2.1",
    "v2.2",
)
DEFAULT_QUESTION = "Quanto tempo leva para carregar uma bateria de 40 kWh em um wallbox de 7 kW?"

HYPERPARAMETROS: dict[str, float | int] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 500,
}

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

_ACCENT_FREE = {"á", "ã", "é", "ê", "í", "ó", "ô", "õ", "ú", "ç"}


def _normalize(text: str) -> str:
    return text.casefold()


def _is_portuguese(text: str) -> bool:
    tokens = re.findall(r"[a-zà-úâãçêôõ]{3,}", _normalize(text))
    if not tokens:
        return False
    return sum(1 for token in tokens if any(char in _ACCENT_FREE for char in token)) / len(tokens) >= 0.03


def _prompt_identifier(version: str) -> str:
    cleaned = version.strip()
    if cleaned.endswith(".txt") or cleaned.startswith("system_prompt"):
        return cleaned
    return f"system_prompt_{cleaned}.txt"


def load_prompt(version: str = "v2.2") -> str:
    """Carrega um system prompt versionado da pasta prompts/ (ex.: v2.1, v2.2)."""
    path = PROMPTS_DIR / _prompt_identifier(version)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt não encontrado: {path}")
    return path.read_text(encoding="utf-8")


def build_chat_model(model_name: str) -> ChatOllama:
    """Constrói um ChatOllama para o modelo informado (Cloud ou local)."""
    load_dotenv()
    api_key = os.environ.get("OLLAMA_API_KEY")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": HYPERPARAMETROS["temperature"],
        "top_p": HYPERPARAMETROS["top_p"],
        "num_predict": HYPERPARAMETROS["max_tokens"],
    }
    if api_key:
        kwargs["base_url"] = "https://ollama.com"
        kwargs["client_kwargs"] = {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            }
        }
    return ChatOllama(**kwargs)


def extract_json_object(content: str) -> dict[str, Any] | None:
    """Extrai o primeiro objeto JSON balanceado do texto retornado pelo modelo."""
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
    except (ValidationError, ValueError) as exc:  # noqa: BLE001
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"valid": True, "error": None}


def score_resposta(pergunta: str, conteudo: str) -> dict[str, Any]:
    """Retorna um score 0-10 para uma resposta (schema + fluência + vocabulário do domínio)."""
    resposta = conteudo.strip()
    if not resposta:
        return {
            "score": 0.0,
            "rotulo": "vazia",
            "schema_valid": False,
            "detalhes": "resposta vazia",
        }

    normalizada = _normalize(resposta)
    score = 0.0

    if _is_portuguese(resposta):
        score += 2.0

    parsed = extract_json_object(resposta)
    schema_valid = False
    if parsed is not None:
        try:
            ConsultaRecarga.model_validate(parsed)
            schema_valid = True
        except (ValidationError, ValueError):  # noqa: BLE001
            schema_valid = False
    if schema_valid:
        score += 5.0

    keyword_hits = sum(
        1 for keyword in _DOMAIN_KEYWORDS if keyword in normalizada
    )
    score += min(keyword_hits, 5) * 0.6

    score = max(0.0, min(round(score, 1), 10.0))

    if score >= 8.0:
        rotulo = "boa"
    elif score >= 5.0:
        rotulo = "mediana"
    else:
        rotulo = "fraca"

    return {
        "score": score,
        "rotulo": rotulo,
        "schema_valid": schema_valid,
        "detalhes": (
            f"idioma_portugues={_is_portuguese(resposta)}; "
            f"schema_valido={schema_valid}; termos_dominio={keyword_hits}"
        ),
    }


@dataclass
class Execucao:
    modelo: str
    prompt: str
    resposta: str
    score: float
    rotulo: str
    schema_valid: bool
    latencia_s: float
    erro: str | None
    tokens_entrada: int
    tokens_saida: int
    detalhes: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelo": self.modelo,
            "prompt": self.prompt,
            "resposta": self.resposta,
            "score": self.score,
            "rotulo": self.rotulo,
            "schema_valid": self.schema_valid,
            "latencia_s": round(self.latencia_s, 3),
            "erro": self.erro,
            "tokens_entrada": self.tokens_entrada,
            "tokens_saida": self.tokens_saida,
            "detalhes_avaliacao": self.detalhes,
        }


class MultiProviderDemo:
    """Consulta o mesmo domínio com mais de um modelo e mais de um prompt versionado.

    `models` aceita nomes de modelos (str) ou objetos com `.invoke(...)` (para
    testes e modelos injetáveis). `prompt_versions` aceita identificadores de
    versão (ex.: "v2.1", "v2.2") ou caminhos de arquivo da pasta prompts/.
    """

    def __init__(
        self,
        models: Iterable[Any] = DEFAULT_MODELS,
        prompt_versions: Iterable[str] = DEFAULT_PROMPT_VERSIONS,
    ):
        self._pares: list[tuple[str, Any]] = []
        for item in models:
            if isinstance(item, str):
                self._pares.append((item, build_chat_model(item)))
            elif hasattr(item, "invoke"):
                nome = getattr(item, "model", None) or type(item).__name__
                self._pares.append((nome, item))
            else:
                raise TypeError(
                    f"Modelo deve ser um nome (str) ou ter .invoke(...): {item!r}"
                )
        self.versoes_prompt = list(prompt_versions)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _build_mensagens(self, pergunta: str, versao: str) -> list[Any]:
        system_prompt = load_prompt(versao)
        parser = PydanticOutputParser(pydantic_object=ConsultaRecarga)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}\n\n{format_instructions}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        return prompt.format_messages(input=pergunta)

    def _executar_combo(
        self,
        pergunta: str,
        nome_modelo: str,
        model: Any,
        versao: str,
    ) -> Execucao:
        try:
            mensagens = self._build_mensagens(pergunta, versao)
        except Exception as exc:  # noqa: BLE001
            return Execucao(
                modelo=nome_modelo,
                prompt=versao,
                resposta="",
                score=0.0,
                rotulo="erro",
                schema_valid=False,
                latencia_s=0.0,
                erro=f"{type(exc).__name__}: {exc}",
                tokens_entrada=0,
                tokens_saida=0,
            )

        tokens_entrada = sum(len(self._encoding.encode(message.content)) for message in mensagens)

        started_ns = time.perf_counter_ns()
        error = None
        resposta = ""
        try:
            resposta = model.invoke(mensagens).content
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            resposta = ""
        latencia_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000

        tokens_saida = len(self._encoding.encode(resposta)) if resposta else 0
        avaliacao = score_resposta(pergunta, resposta)

        return Execucao(
            modelo=nome_modelo,
            prompt=versao,
            resposta=resposta,
            score=avaliacao["score"],
            rotulo=avaliacao["rotulo"] if not error else "erro",
            schema_valid=avaliacao["schema_valid"],
            latencia_s=latencia_s,
            erro=error,
            tokens_entrada=tokens_entrada,
            tokens_saida=tokens_saida,
            detalhes=avaliacao["detalhes"],
        )

    def executar(
        self,
        pergunta: str,
        versoes_prompt: Iterable[str] | None = None,
        modo: str = "paralelo",
    ) -> dict[str, Any]:
        """Executa todas as combinações modelo x prompt e consolida a melhor resposta.

        `modo` aceita "paralelo" (ThreadPoolExecutor) ou "alternado" (sequencial).
        """
        modo_normalizado = modo.strip().lower()
        if modo_normalizado not in ("paralelo", "alternado"):
            raise ValueError('modo deve ser "paralelo" ou "alternado"')

        versoes = list(versoes_prompt or self.versoes_prompt)
        combos = [
            (nome_modelo, model, versao)
            for nome_modelo, model in self._pares
            for versao in versoes
        ]

        if modo_normalizado == "paralelo":
            with ThreadPoolExecutor(max_workers=len(combos)) as executor:
                futures = [
                    executor.submit(self._executar_combo, pergunta, nome, model, versao)
                    for nome, model, versao in combos
                ]
                execucoes = [future.result() for future in futures]
        else:
            execucoes = [
                self._executar_combo(pergunta, nome, model, versao)
                for nome, model, versao in combos
            ]

        melhor = max(execucoes, key=lambda execucao: execucao.score)
        resposta_consolidada = (
            melhor.resposta.strip()
            if melhor.resposta.strip()
            else "Nenhuma combinação retornou uma resposta válida."
        )

        return {
            "pergunta": pergunta,
            "modo": modo_normalizado,
            "modelos": [nome for nome, _ in self._pares],
            "versoes_prompt": versoes,
            "execucoes": [execucao.to_dict() for execucao in execucoes],
            "melhor": melhor.to_dict(),
            "resposta_consolidada": resposta_consolidada,
            "resumo": {
                "total_execucoes": len(execucoes),
                "media_latencia_s": round(
                    statistics.mean(execucao.latencia_s for execucao in execucoes),
                    3,
                ),
                "melhor_score": melhor.score,
            },
        }


def _exibir(resultado: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("DEMONSTRAÇÃO MULTI-PROVIDER")
    print("=" * 78)
    print(f"Pergunta: {resultado['pergunta']}")
    print(f"Modo: {resultado['modo']} | Modelos: {', '.join(resultado['modelos'])}")
    print(f"Prompts: {', '.join(resultado['versoes_prompt'])}")
    print(f"Resumo: {resultado['resumo']}")
    print()

    header = f"{'Modelo':<18} {'Prompt':<8} {'Score':>6} {'Lat(s)':>8} {'Tok in/out':>14} {'Schema':>8}   Resposta/Erro"
    print(header)
    print("-" * len(header))
    for execucao in resultado["execucoes"]:
        schema = "válido" if execucao["schema_valid"] else "não"
        if execucao["erro"]:
            preview = f"ERRO: {execucao['erro']}"
        elif execucao["resposta"]:
            preview = execucao["resposta"][:60].replace("\n", " ") + "…"
        else:
            preview = "SEM RESPOSTA"
        print(
            f"{execucao['modelo']:<18} "
            f"{execucao['prompt']:<8} "
            f"{execucao['score']:>6.1f} "
            f"{execucao['latencia_s']:>8.2f} "
            f"{execucao['tokens_entrada']}/{execucao['tokens_saida']:>8} "
            f"{schema:>8}   {preview}"
        )

    melhores = [e for e in resultado["execucoes"] if e["score"] == resultado["melhor"]["score"]]
    print("\nMelhor combinação:")
    for melhor in melhores:
        print(f"  -> {melhor['modelo']} com prompt {melhor['prompt']} (score {melhor['score']}/10)")
    print("\n" + "=" * 78)
    print("RESPOSTA CONSOLIDADA")
    print("=" * 78)
    print(resultado["resposta_consolidada"])
    print()


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Demonstração multi-provider: compara modelos e versões de prompt."
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPT_VERSIONS))
    parser.add_argument("--mode", default="paralelo", choices=("paralelo", "alternado"))
    parser.add_argument("--output", default=None, help="Caminho para salvar o JSON do resultado")
    args = parser.parse_args(argv)

    demo = MultiProviderDemo(models=args.models, prompt_versions=args.prompts)
    resultado = demo.executar(args.question, modo=args.mode)
    _exibir(resultado)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Resultado salvo em: {output_path}")


if __name__ == "__main__":
    main()