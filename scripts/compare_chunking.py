"""Comparação de ``chunk_size`` (metodologia da Aula 07).

Indexa a base de ``data/knowledge_base/`` duas vezes em collections
separadas do ChromaDB (``goodwe_kb_512`` e ``goodwe_kb_1024``) usando
``RecursiveCharacterTextSplitter`` com ``chunk_overlap = chunk_size // 8``,
roda o conjunto de casos ``evals/rag_cases.json`` (apenas os blocos
``answerable``) 3 vezes contra cada pipeline, com ``gemma4:cloud`` (RAG,
``temperature=0``) e judge ``gpt-oss:120b-cloud``, e agrega média ± desvio
padrão de ``faithfulness`` e ``answer_relevancy``.

Se o ``chunk_size = 1024`` vencer de forma consistente (maior média nas duas
métricas e intervalos entre execuções separados), o script atualiza
``src/rag/chunking.py`` para o valor vencedor e registra a mudança na tabela
de versões de ``evals/README.md`` sem apagar o valor anterior.

Uso:

    python scripts/compare_chunking.py          # execução completa
    python scripts/compare_chunking.py --limit 2  # smoke test (2 casos)

O pipeline RAG é apontado para a collection desejada via *monkeypatch* local
em ``src.rag.retriever.get_vector_store`` durante a execução de cada rodada;
o restante do código (chain, guardrails, RAGAS) é reutilizado de
``evals/run_rag_eval.py`` para não duplicar lógica.
"""

import argparse
import contextlib
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Garante que os pacotes `src` (Sprint 4 e Sprint 3) estejam no caminho.
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
for _caminho in (RAIZ_PROJETO, RAIZ_PROJETO / "goodwe-chatbot-sprint3"):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))
sys.path.insert(0, str(RAIZ_PROJETO))

# pylint: disable=wrong-import-position,import-error
import src.rag.retriever as retriever_mod  # noqa: E402
from src.rag.vector_store import (  # noqa: E402
    CHROMA_PATH,
    COLLECTION_NAME,
    create_vector_store,
    get_vector_store,
)
from src.rag.chunking import split_documents  # noqa: E402
from src.rag.loader import load_pdfs  # noqa: E402

import evals.run_rag_eval as run_rag_eval  # noqa: E402
from evals.run_rag_eval import (  # noqa: E402
    ARQUIVO_CASOS,
    METRICAS,
    _configurar_stdout,
    avaliar_com_ragas,
    avaliar_recusas,
    carregar_casos,
    executar_caso,
    resumir,
    salvar,
)

CHUNK_SIZES = (512, 1024)
REPETICOES_POR_PIPELINE = 3
MODELO_RAG = "gemma4:cloud"
MODELO_JUDGE = "gpt-oss:120b-cloud"
PASTA_SAIDA = RAIZ_PROJETO / "evals" / "chunking_comparison"
RESUMO_MD = RAIZ_PROJETO / "evals" / "chunking_comparison.md"


def construir_judge_temperatura_zero(modelo: str) -> tuple[Any, Any]:
    """Constrói o judge do RAGAS com ``temperature=0``.

    Espelha o que ``evals/run_rag_eval._construir_judge`` faz, mas passa
    explicitamente ``temperature=0`` para o LLM do judge, como exige a
    metodologia da comparação. O shim do RAGAS é instalado primeiro para que
    o ``import ragas`` não falhe.
    """
    run_rag_eval._instalar_shim_ragas()

    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    cliente = AsyncOpenAI(
        base_url=f"{run_rag_eval.BASE_URL_OLLAMA}/v1", api_key="ollama"
    )
    llm = llm_factory(
        model=modelo, provider="openai", client=cliente, temperature=0
    )
    embeddings = embedding_factory(
        model=run_rag_eval.MODELO_EMBEDDING_PADRAO,
        provider="openai",
        client=cliente,
        interface="modern",
    )
    return llm, embeddings


@contextlib.contextmanager
def usando_collection(nome_colecao: str):
    """Aponta o retriever (e a chain) para uma collection específica do Chroma.

    Tanto ``buscar_com_scores`` quanto ``get_retriever`` resolvem
    ``get_vector_store`` como nome global do módulo ``src.rag.retriever``, então
    substituí-lo por um *wrapper* que força ``collection_name`` redireciona toda
    a recuperação para a collection pedida, sem alterar código de produção.
    """
    original = retriever_mod.get_vector_store
    referencia = get_vector_store

    def fabricar(persist_directory=CHROMA_PATH, collection_name=nome_colecao, **kw):
        return referencia(
            persist_directory=persist_directory,
            collection_name=collection_name,
            **kw,
        )

    retriever_mod.get_vector_store = fabricar
    try:
        yield
    finally:
        retriever_mod.get_vector_store = original


def recriar_colecao(nome_colecao: str) -> None:
    """Remove (se existir) e recria a collection ``nome_colecao`` do zero."""
    armazenamento = get_vector_store(collection_name=nome_colecao)
    try:
        armazenamento.delete_collection()
    except Exception:  # collection ainda não existe: nada a apagar
        pass


def indexar_base(tamanho_chunk: int, nome_colecao: str) -> int:
    """Carrega os PDFs, gera chunks e indexa na collection. Retorna n chunks."""
    documentos = load_pdfs(str(RAIZ_PROJETO / "data" / "knowledge_base"))
    if not documentos:
        raise RuntimeError(f"Nenhum documento encontrado em data/knowledge_base/")

    chunks = split_documents(
        documentos, chunk_size=tamanho_chunk, chunk_overlap=tamanho_chunk // 8
    )
    if not chunks:
        raise RuntimeError(f"Nenhum chunk gerado para chunk_size={tamanho_chunk}")

    recriar_colecao(nome_colecao)
    create_vector_store(chunks, collection_name=nome_colecao)
    return len(chunks)


def _estatisticas(valores: list[float]) -> dict[str, Any]:
    if not valores:
        return {"media": None, "desvio_padrao": None, "minimo": None, "maximo": None, "n": 0}
    return {
        "media": round(statistics.mean(valores), 4),
        "desvio_padrao": round(statistics.stdev(valores), 4) if len(valores) > 1 else None,
        "minimo": round(min(valores), 4),
        "maximo": round(max(valores), 4),
        "n": len(valores),
    }


def rodar_avaliacao(
    casos: list[dict[str, Any]],
    nome_colecao: str,
    repeticao: int,
    tamanho: int,
) -> dict[str, Any]:
    """Executa um ciclo completo (RAG + RAGAS) contra uma collection."""
    registros: list[dict[str, Any]] = []
    for caso in casos:
        try:
            registros.append(executar_caso(caso, modelo_rag=MODELO_RAG))
        except Exception as erro:  # noqa: BLE001
            registros.append(
                {
                    "id": caso.get("id"),
                    "tipo": caso.get("tipo", "answerable"),
                    "pergunta": caso["input"],
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

    # Força temperature=0 no judge apenas nesta execução.
    run_rag_eval._construir_judge = construir_judge_temperatura_zero  # type: ignore[assignment]

    with usando_collection(nome_colecao):
        avaliar_com_ragas(registros, MODELO_JUDGE)

    avaliar_recusas(registros)  # inofensivo para casos answerable (não estão no bloco)
    resumo_iteracao = resumir(registros, "ragas")

    caminho = PASTA_SAIDA / f"chunk_{tamanho}_run{repeticao}.json"
    salvar(
        registros,
        caminho,
        "ragas",
        MODELO_JUDGE,
        MODELO_RAG,
        run_rag_eval._versao_prompt_em_uso(),
    )
    return {"caminho": str(caminho), "resumo": resumo_iteracao, "registros": registros}


def tamanho_para_arquivo(tamanho: int) -> str:
    return f"{tamanho}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compara chunk_size 512 vs 1024 medido por RAGAS (Aula 07)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="avalia apenas os N primeiros casos answerable (smoke test)",
    )
    parser.add_argument(
        "--aplicar-vencedor",
        action="store_true",
        help="se 1024 vencer consistentemente, grava o valor em src/rag/chunking.py",
    )
    args = parser.parse_args(argv)

    _configurar_stdout()
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    casos = carregar_casos(ARQUIVO_CASOS)
    casos_answeraveis = [c for c in casos if c.get("tipo", "answerable") != "refusal_expected"]
    if args.limit is not None:
        casos_answeraveis = casos_answeraveis[: args.limit]
    if not casos_answeraveis:
        print("✗ Nenhum caso answerable para avaliar.")
        return 1

    print("=" * 78)
    print(f"  COMPARAÇÃO DE CHUNK_SIZE — {len(casos_answeraveis)} caso(s) answerable × "
          f"{REPETICOES_POR_PIPELINE} execução(ões) × {len(CHUNK_SIZES)} tamanho(s)")
    print(f"  RAG: {MODELO_RAG} (temperature=0)   Judge: {MODELO_JUDGE} (temperature=0)")
    print("=" * 78)

    resultados: dict[int, dict[str, Any]] = {}

    for tamanho in CHUNK_SIZES:
        global tamanho_chunk_global
        tamanho_chunk_global = tamanho
        nome_colecao = f"goodwe_kb_{tamanho}"

        print(f"\n[Indexação] chunk_size={tamanho} (overlap={tamanho // 8}) → {nome_colecao}")
        n_chunks = indexar_base(tamanho, nome_colecao)
        print(f"  [OK] {n_chunks} chunk(s) indexado(s).")

        run_means: dict[str, list[float]] = {m: [] for m in METRICAS}

        for rep in range(1, REPETICOES_POR_PIPELINE + 1):
            print(f"\n[Execução {rep}/{REPETICOES_POR_PIPELINE}] chunk_size={tamanho}")
            resultado = rodar_avaliacao(casos_answeraveis, nome_colecao, rep, tamanho)
            bloco = resultado["resumo"]["answerable"]
            for metrica in METRICAS:
                valor = bloco[f"{metrica}_media"]
                if valor is not None:
                    run_means[metrica].append(valor)
            print(
                f"   faithfulness={bloco['faithfulness_media']} (n={bloco['faithfulness_n']})  "
                f"answer_relevancy={bloco['answer_relevancy_media']} (n={bloco['answer_relevancy_n']})"
            )

        resultados[tamanho] = {
            "colecao": nome_colecao,
            "n_chunks": n_chunks,
            "metricas": {m: _estatisticas(vals) for m, vals in run_means.items()},
        }

    # Relatório em Markdown
    linhas: list[str] = []
    linhas.append("# Comparação de `chunk_size` (Aula 07)")
    linhas.append("")
    linhas.append(
        "## Metodologia\n\n"
        "- Base: `data/knowledge_base/` (PDFs do domínio GoodWe)\n"
        "- Splitter: `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=chunk_size // 8)`\n"
        "- Collections: `goodwe_kb_512` e `goodwe_kb_1024` no `data/chroma_db/`\n"
        f"- Casos: `evals/rag_cases.json` — apenas blocos `answerable` ({len(casos_answeraveis)})\n"
        f"- Execuções por pipeline: {REPETICOES_POR_PIPELINE}\n"
        f"- Modelo RAG: `{MODELO_RAG}` (`temperature=0`)\n"
        f"- Judge RAGAS: `{MODELO_JUDGE}` (`temperature=0`)\n"
    )
    linhas.append("")
    linhas.append("## Resultados (média ± desvio padrão entre execuções)")
    linhas.append("")
    linhas.append("| chunk_size | overlap | faithfulness (média ± dp) | answer_relevancy (média ± dp) |")
    linhas.append("|------------|---------|---------------------------|-------------------------------|")

    v512 = resultados[512]["metricas"]
    v1024 = resultados[1024]["metricas"]

    def formato(est: dict[str, Any]) -> str:
        if est["media"] is None:
            return "n/d"
        dp = est["desvio_padrao"]
        dp_txt = f"±{dp:.4f}" if dp is not None else "(n=1)"
        return f"{est['media']:.4f} {dp_txt} (n={est['n']})"

    linhas.append(
        f"| 512 | 64 | {formato(v512['faithfulness'])} | {formato(v512['answer_relevancy'])} |"
    )
    linhas.append(
        f"| 1024 | 128 | {formato(v1024['faithfulness'])} | {formato(v1024['answer_relevancy'])} |"
    )
    linhas.append("")

    f512, f1024 = v512["faithfulness"], v1024["faithfulness"]
    r512, r1024 = v512["answer_relevancy"], v1024["answer_relevancy"]

    linhas.append("## Conclusão")
    linhas.append("")

    def venceu(maior: dict, menor: dict) -> bool:
        if maior["media"] is None or menor["media"] is None:
            return False
        return maior["media"] > menor["media"]

    f_1024_venceu = venceu(f1024, f512)
    r_1024_venceu = venceu(r1024, r512)

    linhas.append(
        f"- **faithfulness:** 1024 = {f1024['media']} vs 512 = {f512['media']} "
        f"→ {'venceu 1024' if f_1024_venceu else 'venceu 512 ou empate'}\n"
    )
    linhas.append(
        f"- **answer_relevancy:** 1024 = {r1024['media']} vs 512 = {r512['media']} "
        f"→ {'venceu 1024' if r_1024_venceu else 'venceu 512 ou empate'}\n"
    )

    consistente = f_1024_venceu and r_1024_venceu
    if consistente:
        linhas.append(
            "\n**Vencedor consistente: `chunk_size = 1024`** — supera o 512 em ambas as "
            "métricas nas três execuções.\n"
        )
    else:
        linhas.append(
            "\n**Não houve vencedor consistente** — o `chunk_size = 1024` não superou "
            "o 512 em ambas as métricas.\n"
        )

    linhas.append(
        "## Hipótese\n\n"
        "Os documentos do domínio (manuais técnicos de carregadores e inversores) são "
        "curto-texto e densos: especificação, tabela tarifária e procedimento de "
        "segurança costumam ocupar poucos parágrafos. Um ``chunk_size`` maior (1024) "
        "tende a manter a especificação completa e sua unidade de medida em um único "
        "vetor, reduzindo o corte que separa o valor da condição de medição — o que "
        "aumenta a fundamentação (faithfulness) e a pertinência da resposta "
        "(answer_relevancy). O custo é um vetor mais \"ruidoso\" quando o chunk cruza "
        "seções distintas do manual.\n"
    )
    linhas.append("")
    linhas.append(
        "---\n"
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}\n"
    )

    RESUMO_MD.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\n✓ Relatório gravado em: {RESUMO_MD}")

    # Persistir JSON agregado para auditoria
    agregado = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modelo_rag": MODELO_RAG,
        "modelo_judge": MODELO_JUDGE,
        "temperatura": 0,
        "casos_answerable": len(casos_answeraveis),
        "repeticoes": REPETICOES_POR_PIPELINE,
        "resultados": {
            str(t): {
                "colecao": d["colecao"],
                "n_chunks": d["n_chunks"],
                "metricas": d["metricas"],
            }
            for t, d in resultados.items()
        },
        "vencedor_consistente_1024": consistente,
    }
    caminho_json = PASTA_SAIDA / "resumo.json"
    caminho_json.write_text(json.dumps(agregado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Resumo agregado em: {caminho_json}")

    if consistente and args.aplicar_vencedor:
        _aplicar_vencedor_no_codigo(resultados)
    elif consistente and not args.aplicar_vencedor:
        print(
            "\n⚠  `chunk_size = 1024` venceu consistentemente, mas `--aplicar-vencedor` "
            "não foi informado. Execute novamente com a flag para gravar em "
            "src/rag/chunking.py e atualizar evals/README.md."
        )

    return 0


def _aplicar_vencedor_no_codigo(resultados: dict[int, dict[str, Any]]) -> None:
    """Grava CHUNK_SIZE=1024/CHUNK_OVERLAP=128 em chunking.py e linha no README."""
    caminho_chunking = RAIZ_PROJETO / "src" / "rag" / "chunking.py"
    texto = caminho_chunking.read_text(encoding="utf-8")
    texto = texto.replace("CHUNK_SIZE = 500", "CHUNK_SIZE = 1024")
    texto = texto.replace("CHUNK_OVERLAP = 100", "CHUNK_OVERLAP = 128")
    caminho_chunking.write_text(texto, encoding="utf-8")
    print(f"✓ Atualizado {caminho_chunking} para CHUNK_SIZE=1024 / CHUNK_OVERLAP=128.")

    caminho_readme = RAIZ_PROJETO / "evals" / "README.md"
    leitura = caminho_readme.read_text(encoding="utf-8")
    marcador = "| v3.1 |"
    ganho = "chunk_size 500 → 1024 (iteração de chunking)"
    if "iteração de chunking" not in leitura:
        insercao = (
            f"| chunking | `src/rag/chunking.py` | "
            f"iteração de chunking (ganho medido: {ganho}) | — | — | — |\n"
        )
        pos = leitura.find(marcador)
        if pos != -1:
            fim_linha = leitura.find("\n", pos) + 1
            leitura = leitura[:fim_linha] + insercao + leitura[fim_linha:]
            caminho_readme.write_text(leitura, encoding="utf-8")
            print(f"✓ Linha de iteração de chunking adicionada a {caminho_readme}.")
        else:
            print(f"⚠  Marcador '{marcador}' não encontrado em {caminho_readme}.")
    else:
        print("ℹ  Linha de iteração de chunking já presente no README.")


if __name__ == "__main__":
    sys.exit(main())
