import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL_PADRAO = "nomic-embed-text"
BASE_URL_PADRAO = "http://localhost:11434"


def get_embeddings_model(
    model: str = EMBEDDING_MODEL_PADRAO,
) -> OllamaEmbeddings:
    """
    Retorna o modelo de embeddings usado para vetorizar os chunks de texto.

    O modelo `nomic-embed-text` transforma cada chunk em um vetor numerico de
    alta dimensao que representa seu significado. Esses vetores sao o que torna
    possivel a busca semantica: em vez de comparar palavras literais, a base
    vetorial mede a distancia angular entre o vetor da pergunta e os vetores
    dos chunks, permitindo recuperar passagens que falam de "limite de corrente"
    mesmo quando a pergunta usa "corrente maxima admissivel".

    A conexao com o Ollama e configurada por variaveis de ambiente lidas do
    arquivo `.env`, nunca por valores fixos no codigo. `OLLAMA_BASE_URL` define
    o endpoint (com fallback para a instancia local) e `OLLAMA_API_KEY`, quando
    presente, e enviada como header `Authorization: Bearer`, o que permite
    usar o Ollama Cloud sem alterar o codigo. `OllamaEmbeddings` nao possui um
    parametro `api_key` dedicado, entao a credencial trafega por
    `client_kwargs`, que o cliente repassa para o httpx.

    O modelo de embeddings precisa ser o mesmo na ingestion e na consulta,
    porque e ele que determina a dimensao e o espaco vetorial: trocar de modelo
    invalidaria as comparacoes de similaridade sobre a collection existente.

    Args:
        model: Nome do modelo de embeddings no Ollama.

    Returns:
        OllamaEmbeddings: Instancia pronta para `embed_documents()` e
        `embed_query()`.
    """
    load_dotenv()

    base_url = os.getenv("OLLAMA_BASE_URL") or BASE_URL_PADRAO
    client_kwargs: dict = {}

    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
        client_kwargs=client_kwargs,
    )