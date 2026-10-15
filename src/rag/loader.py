import glob
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

DIRETORIO_PADRAO = "data/knowledge_base"


def load_pdfs(directory: str = DIRETORIO_PADRAO) -> list[Document]:
    """Carrega os PDFs da base de conhecimento como lista de ``Document``.

    Percorre ``directory`` recursivamente, localizando todo arquivo com extensao
    ``.pdf`` e extraindo o texto pagina a pagina via ``PyMuPDFLoader``. O
    resultado e uma lista unica (flat) com os documentos de todos os arquivos,
    ja com os metadados normalizados por ``_normalizar_metadados``.

    Por que normalizar em vez de usar os metadados do PyMuPDFLoader direto:

    * ``source`` sai do loader como o **caminho completo** do arquivo
      (``data/knowledge_base/manual_goodwe.pdf``). O prompt de RAG
      (``src/rag/prompt_rag.py``) instrui o modelo a citar
      ``Fonte: <arquivo> (p. <numero>)``, entao o caminho vaza para dentro da
      resposta, polui a citacao e consome tokens do prompt sem nenhum ganho
      para o usuario final. Aqui ele passa a ser apenas o nome do arquivo.
    * ``page`` sai do loader com **base 0** (``page=0`` para a primeira pagina).
      Como a citacao e escrita em linguagem natural ("p. 1"), um indice em base 0
      faria o modelo citar "p. 0" para o primeiro conteudo de um documento --
      uma fonte que o usuario nao consegue conferir. Aqui a pagina passa a ser
      1-indexada, alinhada ao que o leitor ve ao abrir o PDF.

    Args:
        directory: Diretorio onde os PDFs da base de conhecimento estao
            armazenados. Varredura recursiva, entao subpastas tambem sao lidas.

    Returns:
        list[Document]: Documentos extraidos, um por pagina, cada um com
        ``metadata = {"source": <nome do arquivo>, "page": <nº da pagina>}``.
        Retorna lista vazia, com aviso, quando nenhum PDF e encontrado.

    Raises:
        FileNotFoundError: Se ``directory`` nao existir.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Diretorio nao encontrado: {directory}")

    pdf_files = sorted(
        glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
    )

    if not pdf_files:
        print(f"[AVISO] Nenhum arquivo PDF encontrado em: {directory}")
        return []

    documents: list[Document] = []
    for pdf_path in pdf_files:
        try:
            for doc in PyMuPDFLoader(pdf_path).load():
                documents.append(_normalizar_metadados(doc, pdf_path))
        except Exception as exc:
            # Um PDF corrompido ou protegido por senha nao deve derrubar a
            # ingestion inteira: o resto da base continua indexavel.
            print(f"[ERRO] Falha ao carregar '{pdf_path}': {exc}")

    if not documents:
        print(f"[AVISO] Nenhum documento extraido de: {directory}")

    return documents


def _normalizar_metadados(doc: Document, pdf_path: str) -> Document:
    """Reescreve os metadados de um ``Document`` no formato de citacao.

    Substitui o dicionario de metadados por um com exatamente duas chaves:
    ``source``, com o nome do arquivo, e ``page``, com o numero da pagina em base
    1. Todo o resto e descartado: o ``PyMuPDFLoader`` tambem anexa ``producer``,
    ``creator``, ``creationdate``, ``title``, ``author``, ``file_path``,
    ``total_pages`` e afins, que nao servem para a citacao e, se persistidos,
    engordam inutilmente o registro de cada chunk no ChromaDB.

    Esse formato e o contrato consumido pelo prompt de RAG ao montar a citacao
    da resposta e pelo pipeline de avaliacao de recuperacao
    (``evals/rag_cases.json``), que compara o campo ``source`` recuperado com a
    lista ``expected_sources``. Alterar o formato aqui quebra os dois.

    Args:
        doc: Documento recem-extraido pelo ``PyMuPDFLoader``.
        pdf_path: Caminho do PDF de origem, usado para extrair o nome do
            arquivo.

    Returns:
        Document: Novo documento com o mesmo ``page_content`` e metadados
        normalizados.
    """
    pagina = doc.metadata.get("page", 0)

    return Document(
        page_content=doc.page_content,
        metadata={
            "source": os.path.basename(pdf_path),
            "page": pagina + 1,
        },
    )