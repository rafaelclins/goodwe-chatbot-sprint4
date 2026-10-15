"""
Benchmark de modelos no caminho real do RAG, para alimentar
`docs/relatorio_modelos.md`.

O que este script mede, e por que cada número importa
======================================================

`evals/run_rag_eval.py` responde "o modelo é bom o bastante para sustentar as
regras de grounding?". Ele não responde "quanto custa responder", e as duas
perguntas levam a decisões diferentes: um modelo pode ter fidelidade alta e
latência de 40 s por turno, o que é inviável para alguém esperando a resposta
na tela.

As três grandezas medidas aqui:

  1. LATÊNCIA POR TURNO
     Tempo do `responder()` completo — busca vetorial inclusa, porque é ela
     que o usuário espera. Medir só a chamada ao modelo esconderia o custo da
     recuperação.

  2. FIDELIDADE DE CITACAO
     Fração de respostas com ao menos uma citação no formato exato
     `[arquivo.pdf, p. N]`, e fração cujo arquivo citado existe nos rótulos
     do contexto recuperado.
     A segunda é a que separa um modelo que cita de um modelo que decora o
     formato: citar `p. 7` quando o contexto só tem `p. 1` é o modo de falha
     mais difícil de ver a olho, e é exatamente o que a rubrica de
     `evals/rubrica_manual.md` penaliza como fidelidade `0.5`.

  3. TAXA DE RECUSA
     Fração de respostas que recusam. Serve de contrapeso à fidelidade: um
     modelo que recusa tudo tem fidelidade alta e respondibilidade zero.

A escolha de parâmetros é fixa e igual para todos os modelos (`temperature=0`,
os mesmos que `src/rag/prompt_rag.py` usa), porque comparar modelos com
temperaturas diferentes mede a temperatura, não o modelo.

Uso (a partir da raiz do projeto):

    python scripts/benchmark_modelos.py
    python scripts/benchmark_modelos.py --modelos gemma4:cloud gpt-oss:120b-cloud
    python scripts/benchmark_modelos.py --repeticoes 3
    python scripts/benchmark_modelos.py --saida docs/benchmark_modelos.json

Os nomes de modelo da nuvem levam o sufixo `-cloud` quando o Ollama local já
está autenticado (`gemma4:cloud`, `gpt-oss:120b-cloud`). O relatório
`docs/relatorio_modelos.md` registra qual foi usado em cada execução.

O script mede o pipeline local. Ele não chama a API do RAGAS: as notas de
fidelidade do RAGAS vêm de `evals/run_rag_eval.py`, e misturar as duas fontes
numa tabela só tornaria ambíguo de onde veio cada número.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
for _raiz in (RAIZ_PROJETO, RAIZ_PROJETO / "goodwe-chatbot-sprint3"):
    if str(_raiz) not in sys.path:
        sys.path.insert(0, str(_raiz))

# Só o rótulo de citação que `src/rag/retriever.py::format_docs` produz e que o
# prompt exige. O número de página é obrigatório: `[manual.pdf]` sem `p. N` é
# o formato que o V3.0 lista como proibido, e contá-lo como acerto mediria um
# formato que o modelo erra.
ROTULO_CITACAO = re.compile(r"\[([^\[\]]+?),\s*p\.\s*([^\]]+?)\]")

# Perguntas de `evals/rag_cases.json` com `expected_sources` não vazio: a base
# responde a elas, então citação é obrigatória e recusa é erro. As duas últimas
# linhas seriam testes negativos, mas aqui não servem — para medir taxa de
# recusa elas distorcem a média, porque recusar é o comportamento correto.
PERGUNTAS = [
    "O que é um DR em uma instalação de recarga?",
    "Como fazer o rateio do custo de energia da recarga entre moradores?",
    "Quanto custa carregar uma bateria de 60 kWh de 20% a 80%?",
    "O que é kWh?",
    "Como avaliar a infraestrutura elétrica antes de instalar carregadores?",
    "Qual cuidado devo tomar na instalação elétrica?",
]

MODELOS_PADRAO = ["gemma4:cloud", "gpt-oss:120b-cloud"]

# `RESPONDER_ERRO` de `src/rag/pipeline.py` marca falha de infraestrutura: o
# pipeline devolveu a mensagem de erro e o LLM nunca respondeu. Essa resposta
# não tem citação e não é recusa — contá-la em qualquer uma das duas taxas
# desinforma a leitura do relatório, então ela é separada.
PREFIXO_ERRO = "Não consegui consultar a base de conhecimento"

FRASE_RECUSA = "Não encontrei essa informação na base de conhecimento"


def _limpar_texto(valor: str) -> str:
    """Normaliza a resposta antes das contagens.

    O `.env` e a interface usam acentos; o texto observado no terminal deste
    projeto às vezes chega sem eles, por causa da página de codificação do
    console. Comparar `"Não encontrei"` contra `"N?o encontrei"` daria taxa de
    recusa zero sem que nenhum modelo tivesse mudado.
    """
    import unicodedata

    sem_acento = unicodedata.normalize("NFKD", valor)
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


def _versao_prompt() -> str:
    """Versão do prompt em uso, para que a medição seja atribuível."""
    try:
        from src.rag.prompt_rag import get_versao_prompt

        return get_versao_prompt()
    except Exception:  # noqa: BLE001
        return "indisponivel"


def _fixar_modelo(modelo: str):
    """Devolve um contexto que troca o modelo da chain RAG durante a medição.

    Mesma técnica de `evals/run_rag_eval.py`: `responder()` chama
    `build_rag_chain()` sem argumento e o padrão do projeto é `gpt-oss:120b`,
    então o modelo é fixado por substituição temporária do símbolo no módulo.

    Args:
        modelo: Nome do modelo a medir.

    Returns:
        ContextManager: Envolve a medição e restaura a função original.
    """
    from contextlib import contextmanager

    @contextmanager
    def _fixo():
        import src.rag.prompt_rag as prompt_rag

        original = prompt_rag.build_rag_chain
        prompt_rag.build_rag_chain = lambda model=modelo: original(modelo)

        try:
            yield
        finally:
            prompt_rag.build_rag_chain = original

    return _fixo()


def medir_modelo(modelo: str, repeticoes: int) -> dict[str, Any]:
    """Roda o pipeline real com um modelo e resume as três grandezas.

    Args:
        modelo: Nome do modelo no Ollama.
        repeticoes: Vezes que cada pergunta é executada. Serve para separar o
            efeito do modelo do ruído de amostragem; com `temperature=0` a
            variabilidade é baixa, mas a rede da nuvem introduz a sua.

    Returns:
        dict: Tempos, contagens e taxas. As chaves que descrevem as
        perguntas ficam dentro de `por_pergunta`; o resumo é o que entra na
        tabela do relatório.
    """
    from src.rag.pipeline import responder
    from src.rag.retriever import buscar_com_scores

    latencias: list[float] = []
    por_pergunta: list[dict[str, Any]] = []
    citou = citado_com_fonte_valida = recusou = vazio = erro = total = 0

    with _fixar_modelo(modelo):
        for pergunta in PERGUNTAS:
            # Os rótulos recuperados são a lista contra a qual a citação da
            # resposta é conferida. Sem isso, "citou corretamente" seria apenas
            # "citou alguma coisa".
            try:
                disponiveis = {
                    (doc.metadata.get("source"), str(doc.metadata.get("page")))
                    for doc, _ in buscar_com_scores(pergunta)
                }
            except Exception as erro_busca:  # noqa: BLE001
                print(f"  ⚠️  busca falhou para {pergunta[:40]}: {erro_busca}")
                disponiveis = set()

            for _ in range(repeticoes):
                inicio = time.perf_counter()
                resultado = responder(pergunta)
                latencias.append(time.perf_counter() - inicio)

                resposta = resultado["resposta"]
                normalizada = _limpar_texto(resposta)
                total += 1

                if resposta.startswith(PREFIXO_ERRO):
                    erro += 1
                    continue

                if not resposta.strip():
                    vazio += 1
                    continue

                achados = ROTULO_CITACAO.findall(resposta)

                if achados:
                    citou += 1
                    if any(
                        (arquivo.strip(), pagina.strip()) in disponiveis
                        for arquivo, pagina in achados
                    ):
                        citado_com_fonte_valida += 1

                if FRASE_RECUSA in normalizada:
                    recusou += 1

            por_pergunta.append(
                {
                    "pergunta": pergunta,
                    # Ordenado por string, não pelo par: os dois elementos são
                    # strings, mas `sorted` de um set de tuplas funciona; o
                    # erro real vinha de tentar ordenar dicionários aqui.
                    "fontes_recuperadas": sorted(
                        f"{a} | p. {p}" for a, p in disponiveis
                    ),
                }
            )

    respondidas = total - erro - vazio

    return {
        "modelo": modelo,
        "execucoes": total,
        "latencia_media_s": round(mean(latencias), 2) if latencias else None,
        "latencia_mediana_s": round(median(latencias), 2) if latencias else None,
        "latencia_min_s": round(min(latencias), 2) if latencias else None,
        "latencia_max_s": round(max(latencias), 2) if latencias else None,
        "respostas_vazias": vazio,
        "erros_infraestrutura": erro,
        "com_citacao": citou,
        "citacao_com_fonte_valida": citado_com_fonte_valida,
        "recusas": recusou,
        "taxa_citacao": round(citou / respondidas, 4) if respondidas else None,
        "taxa_citacao_valida": (
            round(citado_com_fonte_valida / respondidas, 4) if respondidas else None
        ),
        "taxa_recusa": round(recusou / respondidas, 4) if respondidas else None,
        "por_pergunta": por_pergunta,
    }


def main(argv: list[str] | None = None) -> int:
    """Executa o benchmark e imprime a tabela que vai para o relatório."""
    parser = argparse.ArgumentParser(
        description="Mede latência e fidelidade de citação por modelo no RAG.",
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        default=MODELOS_PADRAO,
        help=f"modelos a medir (padrão: {' '.join(MODELOS_PADRAO)})",
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=1,
        help="execuções por pergunta (padrão: 1)",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="grava o resultado em JSON nesse caminho",
    )

    args = parser.parse_args(argv)

    _configurar_stdout()

    print("=" * 78)
    print("  BENCHMARK DE MODELOS — pipeline RAG")
    print(f"  Prompt RAG: v{_versao_prompt()}")
    print("=" * 78)

    resultados = []

    for modelo in args.modelos:
        print(f"\n>> {modelo} ({args.repeticoes}x por pergunta)")

        try:
            resultado = medir_modelo(modelo, args.repeticoes)
        except Exception as erro:  # noqa: BLE001
            print(f"  ✗ falhou: {type(erro).__name__}: {erro}")
            resultados.append({"modelo": modelo, "erro": f"{type(erro).__name__}: {erro}"})
            continue

        resultados.append(resultado)

        print(f"   latência média ....... {resultado['latencia_media_s']}s")
        print(f"   com citação .......... {_pct(resultado['taxa_citacao'])}")
        print(f"   citação válida ....... {_pct(resultado['taxa_citacao_valida'])}")
        print(f"   recusas .............. {_pct(resultado['taxa_recusa'])}")

        if resultado["erros_infraestrutura"]:
            print(f"   ⚠️  erros de infra: {resultado['erros_infraestrutura']}")
        if resultado["respostas_vazias"]:
            print(f"   ⚠️  respostas vazias: {resultado['respostas_vazias']}")

    print("\n" + "=" * 78)
    print("  TABELA")
    print("=" * 78)
    print(f"{'modelo':<22}{'latência':>10}{'cita':>8}{'c.válida':>10}{'recusa':>9}")

    for resultado in resultados:
        if "erro" in resultado:
            print(f"{resultado['modelo']:<22}{'erro':>10}")
            continue
        print(
            f"{resultado['modelo']:<22}"
            f"{str(resultado['latencia_media_s']) + 's':>10}"
            f"{_pct(resultado['taxa_citacao']):>8}"
            f"{_pct(resultado['taxa_citacao_valida']):>10}"
            f"{_pct(resultado['taxa_recusa']):>9}"
        )

    if args.saida:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prompt_rag_versao": _versao_prompt(),
            "repeticoes": args.repeticoes,
            "perguntas": PERGUNTAS,
            "resultados": resultados,
        }
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n✓ Salvo em: {args.saida}")

    return 0


def _pct(valor: float | None) -> str:
    """Formata uma taxa, tolerando `None` quando não houve resposta."""
    return "n/d" if valor is None else f"{valor * 100:.0f}%"


def _configurar_stdout() -> None:
    """Força UTF-8 no stdout, como `evals/run_rag_eval.py` faz.

    O console padrão do Python no Windows é `cp1252`, que não tem `⚠️` nem `✓`.
    Sem isto, redirecionar a saída derruba o script no primeiro print.
    """
    for fluxo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(fluxo, "reconfigure", None)
        if reconfigurar is not None:
            reconfigurar(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())