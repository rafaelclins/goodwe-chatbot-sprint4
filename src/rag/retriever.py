from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.rag.vector_store import get_vector_store

K_PADRAO = 4


def get_retriever(k: int = K_PADRAO) -> VectorStoreRetriever:
    """Retorna um retriever de busca por similaridade sobre a base vetorial.

    Abre a collection persistente do ChromaDB e a expoe como um retriever que,
    dada a pergunta do usuario, recupera os ``k`` chunks semanticamente mais
    proximos. A busca e por similaridade de vetor (embedding da pergunta contra
    o embedding de cada chunk), e nao por correspondencia literal de palavras:
    e por isso que uma busca por "corrente maxima admissivel" encontra um chunk
    que fala em "limite de corrente".

    O padrao de ``k=4`` equilibra cobertura e ruido. Valores menores correm o
    risco de perder a especificacao complementar em outra pagina do manual (por
    exemplo, um valor acompanhado da sua condicao de medicao); valores maiores
    arrastam trechos irrelevantes para o prompt, o que dilui a precisao do
    grounding. Este valor esta no limiar entre os dois: com os parametros
    atuais de chunking sao 4 de 6 chunks, o que significa que quase toda a base
    alcanca o prompt em qualquer pergunta -- acima de k=5 a base de 6 chunks
    seria sempre devolvida por inteiro e a busca deixaria de discriminar.

    Args:
        k: Numero de chunks recuperados por consulta.

    Returns:
        VectorStoreRetriever: Retriever pronto para ``invoke()`` com a pergunta
        (str) ou integrado em uma chain LCEL.
    """
    db = get_vector_store()

    return db.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    """Monta o bloco de grounding a partir dos chunks recuperados.

    Cada trecho e apresentado com o rotulo da sua fonte entre colchetes, no
    mesmo formato exigido pelo prompt V3.0 para citacao:

        [manual_goodwe.pdf, p. 1]

    O rotulo ficar no MESMO formato da citacao e uma decisao deliberada: o
    modelo nao precisa transpor nomes e paginas de um formato para outro, ele
    copia a etiqueta. Formatos divergentes (rotulo "Fonte: X (p. N)", citacao
    pedida como "[arquivo, p. X]") concentram o erro justamente na transcricao,
    que e a etapa que o usuario precisa conferir contra o PDF.

    Os separadores ``---`` marcam a fronteira entre trechos: sem eles, o modelo
    nao distingue onde acaba um documento e comeca outro, e pode atribuir a
    citacao errada a uma afirmacao que cruza dois trechos.

    Args:
        docs: Chunks recuperados pelo retriever. Cada um deve trazer
            ``metadata["source"]`` e ``metadata["page"]``, garantidos pelo
            contrato de ``src.rag.loader``.

    Returns:
        str: Bloco de contexto com os trechos e suas fontes. Retorna string
        vazia quando ``docs`` e vazio -- nesse caso o prompt V3.0 manda o modelo
        recusar, e nao inventar.
    """
    if not docs:
        return ""

    blocos = [
        f"[{doc.metadata.get('source', 'fonte_desconhecida')}, "
        f"p. {doc.metadata.get('page', 'n/d')}]\n"
        f"{doc.page_content.strip()}"
        for doc in docs
    ]

    return "\n\n---\n\n".join(blocos)


def retrieve_with_sources(question: str, k: int = K_PADRAO) -> tuple[list[Document], str]:
    """Recupera trechos para uma pergunta e devolve os documentos e o contexto.

    Atalho que faz as duas metades do grounding em uma chamada: devolve os
    ``Document`` brutos, com os metadados intactos, e tambem o bloco de texto ja
    formatado por ``format_docs``. Os dois sao necessarios: o texto vai para o
    prompt, enquanto os documentos sao o que a avaliacao de recuperacao
    inspeciona para conferir se a busca trouxe o arquivo certo (campo
    ``expected_sources`` em ``evals/rag_cases.json``).

    Args:
        question: Pergunta do usuario, em texto puro.
        k: Numero de chunks a recuperar.

    Returns:
        tuple: ``(documentos, contexto_formatado)``. Quando nada e recuperado,
        a lista vem vazia e o contexto vem como string vazia.
    """
    docs = get_retriever(k=k).invoke(question)

    return docs, format_docs(docs)


def _espaco_da_collection(db) -> str:
    """Le a metrica de distancia configurada na collection do Chroma.

    Chroma guarda a metrica em ``hnsw.space``. Lemos direto da configuracao
    porque a escolha define como converter a distancia em similaridade, e uma
    conversao errada aqui passaria silenciosamente: nenhum valor fica fora da
    faixa, apenas todos ficam errados.

    Returns:
        str: ``"cosine"``, ``"l2"`` ou ``"ip"``. Em caso de duvida, assume
        ``"l2"``, que e o padrao do Chroma e o que a collection atual usa.
    """
    try:
        config = db._collection.configuration
    except AttributeError:
        return "l2"

    hnsw = (config or {}).get("hnsw") or {}
    spann = (config or {}).get("spann") or {}

    return hnsw.get("space") or spann.get("space") or "l2"


def similarity_from_distance(distancia: float, espaco: str) -> float:
    """Converte a distancia do Chroma em similaridade de cosseno.

    Por que nao usar ``similarity_search_with_relevance_scores`` do LangChain:
    a relevance fn dele e escolhida pela metrica da collection e, no caso de
    ``l2``, faz ``1 - dist/(4 * dimensao)``. Os embeddings do ``nomic-embed-text``
    sao normalizados (norma 1.0), entao a distancia l2 QUADRADA vive em [0, 4] e
    o denominador ``4 * 768`` empurra toda pontuacao para [0.9987, 1.0]. Um
    limiar de 0.30 nunca dispara nessa escala -- nao porque a busca seja boa,
    mas porque a escala nao tem contraste. Por isso o gate compara similaridade
    de cosseno, que e a grandeza que o modelo de embeddings de fato otimiza.

    Args:
        distancia: Distancia devolvida pelo Chroma. Em espaco ``l2`` o Chroma
            devolve a distancia JA ao quadrado.
        espaco: Metrica da collection, como lida por ``_espaco_da_collection``.

    Returns:
        float: Similaridade de cosseno em [-1, 1], onde 1 e vetores identicos.

    Nota:
        A conversao de ``l2`` para cosseno vale porque os embeddings sao de norma
        unitaria. Ela nao valeria para embeddings de norma livre, em que a
        distancia depende do comprimento dos vetores; nesse caso o jeito correto
        e mudar a collection para espaco ``cosine`` e usar
        ``1 - distancia`` diretamente.
    """
    if espaco == "cosine":
        return 1.0 - distancia

    if espaco == "ip":
        return distancia

    # l2: o Chroma devolve o quadrado da distancia. Para vetores de norma 1,
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 2 - 2*cos, logo
    # cos = 1 - ||a - b||^2 / 2.
    return 1.0 - (distancia / 2.0)


def buscar_com_scores(question: str, k: int = K_PADRAO) -> list[tuple[Document, float]]:
    """Recupera trechos ja com similaridade de cosseno calculada.

    Faz a mesma busca de ``get_retriever``, mas preserva a pontuacao de cada
    trecho. E o que permite ao gate de "fora do contexto" decidir, ANTES de
    gastar uma chamada ao LLM, se a base tem alguma resposta a oferecer.

    Args:
        question: Pergunta do usuario, em texto puro.
        k: Numero de chunks a recuperar.

    Returns:
        list[tuple[Document, float]]: Trechos e sua similaridade de cosseno,
        do mais para o menos semelhante. Lista vazia se nada for recuperado.
    """
    db = get_vector_store()
    espaco = _espaco_da_collection(db)

    return [
        (doc, similarity_from_distance(distancia, espaco))
        for doc, distancia in db.similarity_search_with_score(question, k=k)
    ]