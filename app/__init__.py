"""Camada de apresentação do chatbot RAG da Sprint 4.

Este pacote expõe a interface web. Toda a inteligência — recuperação no
ChromaDB, montagem do prompt com grounding e chamada ao Ollama — vive na chain
LCEL de `src.rag.chain`; aqui ficam apenas configuração da página, identificação
do projeto, persistência do histórico e renderização das mensagens.

O arquivo de execução é `app/app.py`, e ele é executado diretamente pelo
framework (`python -m app.app` ou `gradio app/app.py`), sem passar por este
`__init__.py`.
"""