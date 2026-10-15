# `data/knowledge_base/` — Base de conhecimento

Diretório onde ficam os PDFs que alimentam a base vetorial do chatbot RAG.

Coloque aqui os manuais e documentos normativos em **PDF**. O loader
(`src/rag/loader.py`) percorre este diretório recursivamente, carrega todo
arquivo com extensão `.pdf` via `PyMuPDFLoader` e extrai os documentos.

## Uso

Depois de adicionar ou remover PDFs, a base vetorial precisa ser regerada — o
índice do ChromaDB não se atualiza sozinho:

```bash
python -m src.rag.ingest
```

Para acrescentar documentos sem descartar o que já foi ingerido, use
`add_documents()` de `src/rag/vector_store.py`. Reexecutar a ingestão inteira
sobre os mesmos arquivos criaria chunks duplicados, porque o Chroma não
deduplica chunks idênticos por padrão.

## Conteúdo atual

| Arquivo | Origem | Assunto |
|---|---|---|
| `manual_goodwe.pdf` | Fabricante (GoodWe) | Especificações técnicas, segurança elétrica, FAQ |
| `regimento_condominio.pdf` | Condomínio | Rateio de custos, regras de uso, restrições |

Ambos têm 1 página cada. São documentos curtos e densos em valores normativos
(corrente de proteção, tarifa por kWh, multa e tolerância), que é exatamente o
material que o chunking de `src/rag/chunking.py` precisa preservar na
fronteira entre chunks.

## O que deve entrar aqui

- Manuais de equipamentos e protocolos de comunicação.
- Regulamentos, regimentos e convenções do condomínio.
- Normas técnicas aplicáveis à instalação.

Evite introduzir: qualquer documento que não seja do domínio de recarga de
veículos elétricos em condomínio introduz ruído na busca semântica, porque o
retriever não distingue assuntos — ele apenas mede distância angular entre vetores.