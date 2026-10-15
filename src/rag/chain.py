"""Compatibilidade: a montagem da chain RAG passou a viver em ``prompt_rag``.

A implementacao real de `format_docs` e `build_rag_chain` esta em
`src.rag.retriever` e `src.rag.prompt_rag` -- junto do prompt, que e a parte da
chain que realmente evolui entre as versoes do RAG.

Este modulo continua existindo porque `app/app.py` importa
`from src.rag.chain import build_rag_chain`, e `tests/test_app_rag_flow.py`
faz `patch("src.rag.chain.build_rag_chain")`. As duas importacoes seguem
valendo; aqui as funcoes sao apenas reexportadas. Nao ha duas implementacoes
divergentes para se separarem, apenas um segundo nome para as mesmas funcoes.

O motivo da separacao: `prompt_rag` depende de `retriever` (monta a chain) e
`retriever` nao depende de `prompt_rag`. Se `chain.py` continuasse compondo a
LCEL e importasse o prompt, e o prompt por sua vez importasse o retriever, o grafo
ficaria `chain -> prompt_rag -> retriever`, sem ciclo. A inversao oposta
(importar `format_docs` de `chain.py` dentro de `prompt_rag.py`) criaria o ciclo
`chain -> prompt_rag -> chain`, que o Python nao resolve com esses nomes de
modulo. Por isso `format_docs` vive no retriever: ele pertence ao lado da
busca, que e quem produz os documentos que ele formata.
"""

from src.rag.prompt_rag import build_rag_chain
from src.rag.retriever import format_docs

__all__ = ["build_rag_chain", "format_docs"]