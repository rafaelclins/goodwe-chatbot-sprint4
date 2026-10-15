from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.rag.embeddings import get_embeddings_model

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "goodwe_kb"
K_PADRAO = 4


def get_vector_store(
    persist_directory: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Carrega a collection vetorial persistente do ChromaDB.

    Abre (ou cria, se ainda nao existir) a collection `goodwe_kb` no diretorio
    `persist_directory`, usando o mesmo modelo de embeddings da ingestion. A
    collection e persistente de verdade: os vetores ficam em disco entre
    execucoes, entao a ingestion roda uma vez e as consultas reaproveitam o
    indice sem re-embedar os PDFs.

    O modelo de embeddings precisa ser o mesmo usado na criacao, porque a
    dimensao e o espaco vetorial sao determinados por ele. Usar outro modelo
    sobre a mesma collection tornaria as comparacoes de similaridade invalidas.

    Args:
        persist_directory: Diretorio onde a collection e gravada em disco.
        collection_name: Nome da collection dentro do diretorio.

    Returns:
        Chroma: Instancia da collection vetorial persistente.
    """
    embeddings = get_embeddings_model()

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def create_vector_store(
    documents: list[Document],
    persist_directory: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Cria a collection vetorial a partir dos chunks de texto.

    Cada chunk recebido e convertido em vetor pelo modelo `nomic-embed-text` e
    gravado na collection persistente do Chroma, junto dos metadados de origem
    (arquivo e pagina), que depois permitem citar a fonte da resposta.

    Args:
        documents: Lista de chunks que serao convertidos em vetores e
            persistidos.
        persist_directory: Diretorio onde a collection sera gravada.
        collection_name: Nome da collection a ser criada.

    Returns:
        Chroma: Instancia da collection vetorial persistida.
    """
    embeddings = get_embeddings_model()

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


def add_documents(
    documents: list[Document],
    persist_directory: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Acrescenta documentos a uma collection persistente ja existente.

    Diferente de `create_vector_store`, que popula a collection do zero, esta
    funcao apenas adiciona vetores: e o caminho para ingerir um novo PDF sem
    descartar o que ja foi ingerido. Reexecutar a ingestion inteira
    sobre os mesmos arquivos criaria duplicatas, porque o Chroma nao deduplica
    chunks identicos por padrao.

    Args:
        documents: Lista de chunks a serem adicionados.
        persist_directory: Diretorio onde a collection foi persistida.
        collection_name: Nome da collection de destino.

    Returns:
        Chroma: Instancia da collection apos a inclusao dos documentos.
    """
    store = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    store.add_documents(documents)

    return store


def as_retriever(
    k: int = K_PADRAO,
    persist_directory: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreRetriever:
    """
    Expoe a collection vetorial como um retriever de busca semantica.

    Dada a pergunta do usuario, o retriever recupera os `k` chunks
    semanticamente mais proximos. Esse contexto e o que fundamenta a resposta
    do assistente, permitindo que ela cite os manuais em vez de responder
    apenas com o conhecimento do modelo.

    O padrao de `k=4` equilibra cobertura e ruido: valores menores correm o
    risco de perder a especificacao complementar em outra pagina do manual
    (por exemplo, um valor acompanhado da sua condicao de medicao), enquanto
    valores maiores arrastam trechos irrelevantes para o prompt.

    Args:
        k: Numero de chunks recuperados por consulta.
        persist_directory: Diretorio onde a collection foi persistida.
        collection_name: Nome da collection consultada.

    Returns:
        VectorStoreRetriever: Retriever pronto para ser invocado com
        `invoke()` ou integrado em uma chain do LangChain.
    """
    store = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return store.as_retriever(search_kwargs={"k": k})