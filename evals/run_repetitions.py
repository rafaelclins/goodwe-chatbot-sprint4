"""
Repetições do eval RAG: N execuções por versão de prompt, com média e desvio.

POR QUE ESTE SCRIPT EXISTE
=========================

Uma execução única não separa efeito de prompt de ruído. Medido nesta base com
`gemma4:cloud` como modelo e `gpt-oss:120b-cloud` como judge:

    v3.0:  faithfulness 0,6085   answer_relevancy 0,6107
    v3.1:  faithfulness 0,5852   answer_relevancy 0,5657

Lido direto, isso diz que o v3.1 piorou. Mas o caso `conceptual_demand` sozinho
responde por quase todo o delta de relevância, e nele **os dois prompts se
comportam igual** — ambos recusam corretamente. O judge deu 0,81 e 0,00 para a
mesma decisão, por causa da redação do segundo parágrafo da recusa. Um número
que muda com a formulação é ruído, não efeito.

Com 9 casos, uma diferença de 0,045 na média equivale a meio caso. Repetir é o
que permite dizer se 0,6085 × 0,5857 é sinal ou amostra.

O QUE ESTE SCRIPT FAZ
=====================

Para cada versão em `--versions`, executa o eval inteiro `--repeticoes` vezes e
reporta, por bloco:

    media, desvio_padrao, minimo, maximo, n

O desvio padrão é o dado novo: sem ele, uma média de 0,59 não diz se o prompt
produz 0,59 sempre ou 0,45/0,73 alternadamente.

O script **reaproveita `run_rag_eval.py`** em vez de duplicar a lógica — o
split answerable/refusal_expected, a rubrica de recusa e o shim do RAGAS vivem
lá, e uma cópia divergiria silenciosamente.

COMPARAÇÃO ENTRE VERSÕES
========================

`--comparar` emite o delta v(B) − v(A) com o intervalo de sobreposição. Duas
execuções com intervalos que se sobrepõem não sustentam afirmação de diferença;
o script marca isso explicitamente em vez de deixar o leitor concluir sozinho.

USO
===

    python evals/run_repetitions.py --versions 3.0 3.1 --repeticoes 3
    python evals/run_repetitions.py --versions 3.0 3.1 3.2 --repeticoes 5
    python evals/run_repetitions.py --versions 3.1 --repeticoes 3 --saida evals/rep_v31.json

    # smoke, sem gastar chamada de LLM em tudo
    python evals/run_repetitions.py --versions 3.0 3.1 --repeticoes 2 --limit 3

Cada execução grava o JSON completo em `evals/repetitions/`, para que uma
repetição possa ser auditada sem rerodar o conjunto. O `--saida` aponta para o
resumo agregado.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Mesma inserção de `sys.path` de `run_rag_eval.py`: os dois `src` são namespace
# packages (PEP 420) e precisam estar no caminho antes de qualquer import de
# `src.rag`.
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
for _raiz in (RAIZ_PROJETO, RAIZ_PROJETO / "goodwe-chatbot-sprint3"):
    if str(_raiz) not in sys.path:
        sys.path.insert(0, str(_raiz))

sys.path.insert(0, str(RAIZ_PROJETO))

# Importado depois do `sys.path`: `run_rag_eval` ajusta o caminho no próprio
# import, mas depender disso seria frágil se ele mudar.
from evals.run_rag_eval import (  # noqa: E402
    ARQUIVO_CASOS,
    METRICAS,
    MODELO_JUDGE_PADRAO,
    avaliar_com_ragas,
    avaliar_recusas,
    avaliar_manualmente,
    carregar_casos,
    classificar_recusa,
    executar_caso,
    resumir,
    salvar,
)

PASTA_REPETICOES = RAIZ_PROJETO / "evals" / "repetitions"
MODELO_RAG_PADRAO = "gemma4:cloud"


def _configurar_stdout() -> None:
    """Força UTF-8 no stdout — `cp1252` não tem `✓`, `⚠️` nem `→`."""
    for fluxo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(fluxo, "reconfigure", None)
        if reconfigurar is not None:
            reconfigurar(encoding="utf-8", errors="replace")


def executar_uma_vez(
    casos: list[dict[str, Any]],
    versao: str,
    modelo_rag: str,
    modelo_judge: str,
    manual: bool,
) -> dict[str, Any]:
    """Roda o conjunto inteiro uma vez, com o prompt da versão pedida.

    A versão é fixada por variável de ambiente, e não por argumento: `get_rag_prompt()`
    lê `PROMPT_RAG_VERSION` a cada chamada, e é essa leitura por chamada que
    permite trocar de versão sem recarregar o módulo. Sem essa propriedade,
    repetir exigiria um processo novo por versão.

    Args:
        casos: Casos de `evals/rag_cases.json`.
        versao: Versão do prompt, ex.: `3.0`.
        modelo_rag: Modelo de chat da chain RAG.
        modelo_judge: Modelo de judge do RAGAS.
        manual: Se verdadeiro, usa a rubrica manual em vez do RAGAS.

    Returns:
        dict: O payload de `salvar()`, com `resumo` já nos dois blocos.
    """
    os.environ["PROMPT_RAG_VERSION"] = versao

    registros: list[dict[str, Any]] = []

    for caso in casos:
        try:
            registros.append(executar_caso(caso, modelo_rag=modelo_rag))
        except Exception as erro:  # noqa: BLE001
            # Um caso que derruba a execução inteira esconde os demais. O registro
            # espelha o tratamento de `run_rag_eval.main()`: a falha fica
            # registrada e a repetição continua.
            registros.append(
                {
                    "id": caso.get("id"),
                    "tipo": caso.get("tipo", "answerable"),
                    "pergunta": caso["input"],
                    "expected_sources": caso.get("expected_sources", []),
                    "resposta": "",
                    "contextos": [],
                    "fontes": [],
                    "bloqueado": True,
                    "recusa_guardrail": False,
                    "erro_infraestrutura": True,
                    "erro_pipeline": f"{type(erro).__name__}: {erro}",
                    "faithfulness": None,
                    "answer_relevancy": None,
                }
            )

    metodo = "rubrica_manual"

    if manual:
        avaliar_manualmente(registros)
    else:
        avaliar_com_ragas(registros, modelo_judge)
        metodo = "ragas"

    avaliar_recusas(registros)

    return {
        "metodo": metodo,
        "resumo": resumir(registros, metodo),
        "registros": registros,
    }


def _estatisticas(valores: list[float]) -> dict[str, Any]:
    """Média, desvio padrão e extremos de uma série de notas.

    O desvio padrão é de *amostra* (`statistics.stdev`), não populacional: as
    execuções são uma amostra das respostas que o modelo pode dar, e é a
    dispersão dessa amostra que limita o que se pode concluir.

    Com um único valor, `stdev` não é definido — e a ausência de desvio é
    justamente a informação que impede concluir que duas execuções se
   parecem. O campo vira `None`, nunca `0.0`.

    Args:
        valores: Notas das execuções com resultado válido.

    Returns:
        dict: `media`, `desvio_padrao`, `minimo`, `maximo` e `n`.
    """
    if not valores:
        return {"media": None, "desvio_padrao": None, "minimo": None, "maximo": None, "n": 0}

    return {
        "media": round(statistics.mean(valores), 4),
        # Uma execução não informa variabilidade. `0.0` seria uma afirmação
        # falsa de determinismo.
        "desvio_padrao": round(statistics.stdev(valores), 4) if len(valores) > 1 else None,
        "minimo": round(min(valores), 4),
        "maximo": round(max(valores), 4),
        "n": len(valores),
    }


def agregar(execucoes: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduz as execuções de uma versão a médias e desvios por bloco e por caso.

    Além da média geral, calcula **por caso**: sem isso, uma versão que melhora
    três casos e piora cinco aparece como "caiu", e a informação que importa —
    quais casos moveram a média — fica no detalhe dos JSONs.

    Args:
        execucoes: Payloads de `executar_uma_vez()`.

    Returns:
        dict: Estatísticas do bloco `answerable`, do bloco `refusal_expected`, e
        por caso.
    """
    registros = [r for execucao in execucoes for r in execucao["registros"]]

    answerable = [r for r in registros if r.get("tipo") != "refusal_expected"]
    recusa = [r for r in registros if r.get("tipo") == "refusal_expected"]

    por_caso: dict[str, Any] = {}

    for registro in registros:
        id_caso = registro.get("id")

        if id_caso not in por_caso:
            por_caso[id_caso] = {"tipo": registro.get("tipo"), "execucoes": []}

        if registro.get("tipo") == "refusal_expected":
            por_caso[id_caso]["execucoes"].append(registro.get("recusa_nota"))
        else:
            por_caso[id_caso]["execucoes"].append(
                {
                    "faithfulness": registro.get("faithfulness"),
                    "answer_relevancy": registro.get("answer_relevancy"),
                }
            )

    por_caso_resumido: dict[str, Any] = {}

    for id_caso, dados in por_caso.items():
        if dados["tipo"] == "refusal_expected":
            por_caso_resumido[id_caso] = {
                "tipo": dados["tipo"],
                "recusa_nota": _estatisticas(
                    [v for v in dados["execucoes"] if v is not None]
                ),
            }
        else:
            por_caso_resumido[id_caso] = {
                "tipo": dados["tipo"],
                **{
                    metrica: _estatisticas(
                        [
                            e[metrica]
                            for e in dados["execucoes"]
                            if isinstance(e, dict) and e.get(metrica) is not None
                        ]
                    )
                    for metrica in METRICAS
                },
            }

    # Falhas de métrica somem do agregado: um caso cujo judge falhou fica com
    # `None` e não entra na média, sem sinal. Sem esta contagem, um `n` menor na
    # média parece um `n` menor por escolha do subconjunto — e o número que
    # depende disso é justamente a dispersão, que já é frágil.
    falhas = [
        {
            "execucao": i + 1,
            "id": r.get("id"),
            "metrica": campo,
            "erro": r.get(campo),
        }
        for i, execucao in enumerate(execucoes)
        for r in execucao["registros"]
        for campo in ("erro_faithfulness", "erro_answer_relevancy")
        if r.get(campo)
    ]

    return {
        "answerable": {
            metrica: _estatisticas(
                [r[metrica] for r in answerable if r.get(metrica) is not None]
            )
            for metrica in METRICAS
        },
        "refusal_expected": {
            "recusa_nota": _estatisticas_recusa(recusa),
        },
        "falhas_de_metrica": falhas,
        "n_execucoes_com_falha": len({(f["execucao"], f["id"]) for f in falhas}),
        "por_caso": por_caso_resumido,
    }


def _estatisticas_recusa(registros: list[dict[str, Any]]) -> dict[str, Any]:
    """Estatísticas da rubrica binária de recusa.

    Args:
        registros: Casos `refusal_expected` de todas as execuções.

    Returns:
        dict: Estatísticas de `recusa_nota`, mais a contagem por desfecho — que é
        o que distingue "recusou pelo guardrail" de "recusou porque o modelo
        entendeu", dois comportamentos com o mesmo 1 e origens diferentes.
    """
    valores = [r["recusa_nota"] for r in registros if r.get("recusa_nota") is not None]

    desfechos: dict[str, int] = {}
    for registro in registros:
        chave = registro.get("recusa_resultado") or "nao_avaliado"
        desfechos[chave] = desfechos.get(chave, 0) + 1

    return {**_estatisticas(valores), "desfechos": desfechos}


def comparar(versoes: dict[str, Any]) -> dict[str, Any]:
    """Compara versões duas a duas, avisando quando os intervalos se sobrepõem.

    Dois intervalos `[min, max]` que se sobrepõem não sustentam a afirmação de
    que um prompt é melhor que o outro. O script não decide quem ganhou: marca
    o que os dados sustentam e deixa a conclusão explícita.

    Args:
        versoes: Agregado por versão, de `agregar()` mais o rótulo `versao`.

    Returns:
        dict: Um item por par de versões, na ordem em que foram pedidas.
    """
    rotulos = list(versoes)
    comparacoes = []

    for i, va in enumerate(rotulos):
        for vb in rotulos[i + 1 :]:
            itens = []
            sobrepostos = []

            for bloco in ("answerable", "refusal_expected"):
                for metrica in (
                    METRICAS if bloco == "answerable" else ("recusa_nota",)
                ):
                    a = versoes[va]["agregado"][bloco][metrica]
                    b = versoes[vb]["agregado"][bloco][metrica]

                    if not a["media"] or not b["media"]:
                        continue

                    # Desvio ausente (uma execução só) não dá intervalo: sem ele,
                    # a comparação é de ponto único e não se sustenta.
                    if a["desvio_padrao"] is None or b["desvio_padrao"] is None:
                        sobrepostos.append(None)
                        itens.append(
                            {
                                "bloco": bloco,
                                "metrica": metrica,
                                **{f"{va}": a["media"], f"{vb}": b["media"]},
                                "delta": round(b["media"] - a["media"], 4),
                                "sustentavel": None,
                                "motivo": "uma execução só: sem desvio padrão",
                            }
                        )
                        continue

                    intervalos_separados = (
                        a["minimo"] > b["maximo"] or b["minimo"] > a["maximo"]
                    )

                    if intervalos_separados:
                        sobrepostos.append(False)
                    else:
                        sobrepostos.append(True)

                    itens.append(
                        {
                            "bloco": bloco,
                            "metrica": metrica,
                            f"{va}_media": a["media"],
                            f"{vb}_media": b["media"],
                            f"{va}_desvio": a["desvio_padrao"],
                            f"{vb}_desvio": b["desvio_padrao"],
                            "delta": round(b["media"] - a["media"], 4),
                            "intervalos_separados": intervalos_separados,
                            # Só `True` autoriza dizer que a diferença é real.
                            "sustentavel": intervalos_separados,
                        }
                    )

            comparáveis = [s for s in sobrepostos if s is not None]

            comparacoes.append(
                {
                    "de": va,
                    "para": vb,
                    "metricas": itens,
                    "alguma_metrica_sustentavel": any(sobrepostos)
                    if comparáveis
                    else None,
                    "leitura": (
                        "Nenhuma métrica tem intervalos separados: as diferenças "
                        "observadas estão dentro da dispersão das execuções."
                        if comparáveis and not any(sobrepostos)
                        else (
                            "Há métrica com intervalos separados."
                            if any(sobrepostos)
                            else "Dados insuficientes para comparar."
                        )
                    ),
                }
            )

    return {"pares": comparacoes}


def main(argv: list[str] | None = None) -> int:
    """Executa as repetições e grava o resumo agregado."""
    parser = argparse.ArgumentParser(
        description="Repete o eval RAG por versão de prompt e agrega por média e desvio.",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["3.0", "3.1"],
        help="versões de prompt a comparar (padrão: 3.0 3.1)",
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=3,
        help="execuções por versão (padrão: 3)",
    )
    parser.add_argument(
        "--rag-model",
        default=MODELO_RAG_PADRAO,
        help=f"modelo da chain RAG (padrão: {MODELO_RAG_PADRAO})",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-oss:120b-cloud",
        help="modelo de judge do RAGAS",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ARQUIVO_CASOS,
        help="arquivo de casos (padrão: evals/rag_cases.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="apenas os N primeiros casos de cada execução",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="usa a rubrica manual em vez do RAGAS",
    )
    parser.add_argument(
        "--nao-pular-ragas",
        action="store_true",
        help="não cai para a rubrica manual se o RAGAS falhar",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=RAIZ_PROJETO / "evals" / "repetitions_summary.json",
        help="caminho do resumo agregado",
    )
    parser.add_argument(
        "--sem-detalhe",
        action="store_true",
        help="não grava os JSONs individuais em evals/repetitions/",
    )

    args = parser.parse_args(argv)

    _configurar_stdout()

    casos = carregar_casos(args.cases)

    if args.limit is not None:
        casos = casos[: args.limit]

    if not casos:
        print("✗ Nenhum caso para avaliar.")
        return 1

    print("=" * 78)
    print(f"  REPETIÇÕES DO EVAL RAG — {len(args.versions)} versão(ões) "
          f"× {args.repeticoes} repetição(ões) × {len(casos)} caso(s)")
    print(f"  modelo RAG: {args.rag_model}    judge: {args.judge_model}")
    print("=" * 78)

    versoes: dict[str, Any] = {}
    total_execucoes = len(args.versions) * args.repeticoes
    contador = 0

    for versao in args.versions:
        execucoes: list[dict[str, Any]] = []

        for repeticao in range(1, args.repeticoes + 1):
            contador += 1
            print(f"\n[{contador}/{total_execucoes}] prompt v{versao} "
                  f"— repetição {repeticao}/{args.repeticoes}")

            execucao = executar_uma_vez(
                casos, versao, args.rag_model, args.judge_model, args.manual
            )
            execucao["repeticao"] = repeticao
            execucao["prompt_rag_versao"] = versao
            execucoes.append(execucao)

            bloco = execucao["resumo"]["answerable"]
            recusa = execucao["resumo"]["refusal_expected"]
            # O `n` exibido é o de cada métrica, não o total de casos: uma
            # falha pontual do judge deixa `faithfulness_n` menor que
            # `total_casos`, e mostrar o total esconderia exatamente isso.
            print(
                f"   answerable: f={_nota(bloco['faithfulness_media'])} "
                f"(n={bloco['faithfulness_n']})  "
                f"r={_nota(bloco['answer_relevancy_media'])} "
                f"(n={bloco['answer_relevancy_n']})"
            )
            print(
                f"   refusal_expected: recusa="
                f"{_nota(recusa['taxa_recusa_correta'])} "
                f"(n={recusa['recusa_nota_n']})"
            )

            if not args.sem_detalhe:
                caminho = PASTA_REPETICOES / f"v{versao}_rep{repeticao}.json"
                salvar(
                    execucao["registros"],
                    caminho,
                    execucao["metodo"],
                    args.judge_model,
                    args.rag_model,
                    versao,
                )

        versoes[versao] = {
            "versao": versao,
            "repeticoes": args.repeticoes,
            "agregado": agregar(execucoes),
        }

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metodologia": {
            "split": "tipo=answerable -> RAGAS; tipo=refusal_expected -> rubrica de recusa",
            "desvio_padrao": "amostral (statistics.stdev); None com menos de 2 execuções",
            "comparacao": (
                "intervalos [min, max] separados sustentam a diferença; "
                "sobrepostos, não sustentam"
            ),
        },
        "modelo_rag": args.rag_model,
        "modelo_judge": args.judge_model if not args.manual else None,
        "metodo": "rubrica_manual" if args.manual else "ragas",
        "repeticoes_por_versao": args.repeticoes,
        "total_casos": len(casos),
        "versoes": versoes,
        "comparacao": comparar(versoes),
    }

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _imprimir_tabela(payload)

    print(f"\n✓ Resumo em: {args.saida}")

    if not args.sem_detalhe:
        print(f"✓ Execuções individuais em: {PASTA_REPETICOES}")

    return 0


def _nota(valor: float | None) -> str:
    """Formata uma nota, tolerando `None`."""
    return "n/d" if valor is None else f"{valor:.4f}"


def _imprimir_tabela(payload: dict[str, Any]) -> None:
    """Imprime a tabela consolidada no terminal.

    Mostra média ± desvio entre parênteses, e `n/d` quando o desvio não existe
    (uma execução só). O desvio entre parênteses é deliberado: sem ele, duas
    colunas com médias próximas parecem equivalentes quando não são.
    """
    print("\n" + "=" * 78)
    print("  MÉDIA ± DESVIO POR VERSÃO")
    print("=" * 78)

    rotulos = list(payload["versoes"])

    print(f"{'versão':<8}{'faithfulness':<22}{'answer_relev.':<22}{'recusa_ok':<16}")
    print("-" * 78)

    for versao in rotulos:
        agg = payload["versoes"][versao]["agregado"]
        f = agg["answerable"]["faithfulness"]
        r = agg["answerable"]["answer_relevancy"]
        rec = agg["refusal_expected"]["recusa_nota"]

        print(
            f"{'v' + versao:<8}"
            f"{_com_desvio(f):<22}{_com_desvio(r):<22}{_com_desvio(rec):<16}"
        )

    print("-" * 78)
    print(f"{'n/caso':<8}{payload['total_casos']}")

    if len(rotulos) > 1:
        print("\n" + "=" * 78)
        print("  COMPARAÇÃO")
        print("=" * 78)

        for par in payload["comparacao"]["pares"]:
            print(f"\nv{par['de']} → v{par['para']}: {par['leitura']}")

            for item in par["metricas"]:
                marca = (
                    "sustentada" if item.get("sustentavel")
                    else ("não sustentada" if item.get("sustentavel") is False else "indecidível")
                )
                print(
                    f"   {item['metrica']:<18} delta={item['delta']:+.4f}   {marca}"
                )


def _com_desvio(estat: dict[str, Any]) -> str:
    """Formata `média ± desvio` para a tabela."""
    if estat.get("media") is None:
        return "n/d"

    if estat.get("desvio_padrao") is None:
        return f"{estat['media']:.4f} (±? n=1)"

    return f"{estat['media']:.4f} ±{estat['desvio_padrao']:.4f}"


if __name__ == "__main__":
    sys.exit(main())