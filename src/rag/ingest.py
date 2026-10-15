from src.rag.loader import load_pdfs
from src.rag.chunking import split_documents
from src.rag.vector_store import create_vector_store


def run_ingestion(data_dir: str = "data/knowledge_base") -> None:
    """
    Executa o pipeline de ingestão da base de conhecimento.

    Carrega os PDFs do diretorio informado, divide os documentos em chunks e
    grava os vetores correspondentes no ChromaDB, deixando a base pronta para
    as consultas do retriever.

    Args:
        data_dir: Diretorio contendo os PDFs da base de conhecimento.

    Returns:
        None: A funcao apenas executa o pipeline e reporta o progresso.
    """
    print("Carregando PDFs...")
    documents = load_pdfs(data_dir)

    if not documents:
        print(f"[AVISO] Nenhum documento encontrado em '{data_dir}'. Ingestão cancelada.")
        return

    print(f"[OK] {len(documents)} documento(s) carregado(s).")

    print("Gerando chunks...")
    chunks = split_documents(documents)

    if not chunks:
        print("[AVISO] Nenhum chunk gerado. Ingestão cancelada.")
        return

    print(f"[OK] {len(chunks)} chunk(s) gerado(s).")

    print("Salvando no ChromaDB...")
    create_vector_store(chunks)

    print("Ingestão concluída!")


if __name__ == "__main__":
    run_ingestion()
