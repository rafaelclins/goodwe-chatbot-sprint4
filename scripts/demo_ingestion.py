"""
Teste manual do trecho de carga e chunking do pipeline de RAG.

Executa o caminho `load_pdfs()` -> `split_documents()` sobre um PDF real da base
e imprime a quantidade de chunks e os metadados do primeiro, que sao exatamente
os dois numeros que o relatorio de evolucao precisa registrar quando o
chunking muda.

Este script nao chama o Ollama nem o ChromaDB: ele termina antes de
`create_vector_store()`, entao roda offline e nao precisa da base ja ingerida.

Uso (a partir da raiz do projeto):

    python scripts/demo_ingestion.py
    python scripts/demo_ingestion.py data/knowledge_base/regimento_condominio.pdf
"""

import sys
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.rag.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_documents  # noqa: E402
from src.rag.loader import load_pdfs  # noqa: E402

PDF_EXEMPLO = "data/knowledge_base/manual_goodwe.pdf"


def main() -> None:
    alvo = sys.argv[1] if len(sys.argv) > 1 else PDF_EXEMPLO

    print("=" * 72)
    print(f"Carga + chunking: {alvo}")
    print("=" * 72)

    # Um arquivo vira um diretorio temporario, porque load_pdfs() varre
    # diretorios. Reaproveitar a base inteira tambem funciona, mas isolar o PDF
    # deixa a saida comparável entre versoes do chunking.
    if Path(alvo).is_file():
        pasta = Path(alvo).parent
        documentos_alvo = load_pdfs(str(pasta))
        documentos_alvo = [
            d for d in documentos_alvo if d.metadata["source"] == Path(alvo).name
        ]
    else:
        documentos_alvo = load_pdfs(alvo)

    print(f"\nDocumentos carregados: {len(documentos_alvo)}")
    for doc in documentos_alvo:
        print(f"  - {doc.metadata['source']} (p. {doc.metadata['page']})")

    if not documentos_alvo:
        print("\n[ERRO] Nenhum documento carregado.")
        sys.exit(1)

    chunks = split_documents(documentos_alvo)

    print(f"\nConfiguracao: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print(f"QUANTIDADE DE CHUNKS: {len(chunks)}")
    print(f"Caracteres no total: {sum(len(c.page_content) for c in chunks)}")
    print(f"Media por chunk: {sum(len(c.page_content) for c in chunks) // len(chunks)}")

    print("\nMETADADOS DO PRIMEIRO CHUNK:")
    print(f"  {chunks[0].metadata}")

    print("\n" + "-" * 72)
    print("CONFERENCIA DE CITACAO")
    print("-" * 72)

    # Estas verificacoes nao sao decorativas: sao o contrato que o prompt de RAG
    # depende para montar "Fonte: <arquivo> (p. <n>)". Se 'page' voltar a base 0
    # ou 'source' voltar a ser o caminho completo, a citacao do chatbot passa a
    # citar uma pagina que o usuario nao consegue conferir.
    problemas = []

    for chunk in chunks:
        if set(chunk.metadata) != {"source", "page"}:
            problemas.append(f"chaves inesperadas: {sorted(chunk.metadata)}")
            break
        if "/" in chunk.metadata["source"] or "\\" in chunk.metadata["source"]:
            problemas.append(f"source nao e nome de arquivo: {chunk.metadata['source']}")
            break
        if not isinstance(chunk.metadata["page"], int) or chunk.metadata["page"] < 1:
            problemas.append(f"page fora de base 1: {chunk.metadata['page']}")
            break

    if problemas:
        for problema in problemas:
            print(f"  [FALHA] {problema}")
        sys.exit(1)

    print("  [OK] todos os chunks com source (nome) e page (base 1)")
    print(f"  [OK] {len(chunks)} chunk(s) preservaram os metadados de origem")


if __name__ == "__main__":
    main()