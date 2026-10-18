# GoodWe EV Chatbot — Sprint 4 (RAG com grounding)

Chatbot técnico para dúvidas de **recarga de veículos elétricos em condomínios**,
usando IA generativa ancorada em uma base de conhecimento própria (RAG). Esta
sprint evolui o chatbot das Sprints 2 e 3 para um sistema que **responde apenas
com o que está nos manuais**, **cita a fonte** de cada afirmação e recusa quando
a base não tem a resposta.

Desenvolvido para o Challenge FIAP + GoodWe 2026.

---

## Integrantes

Turma: **1CCPI**

- Rafael Lins — RM: 570588 — Turma: 1CCPI
- Rafael Gandolfi — RM: 569036 — Turma: 1CCPI
- Carlos Tavares — RM: 572949 — Turma: 1CCPI
- Cauã Paes — RM: 569906 — Turma: 1CCPI
- Guilherme Miranda — RM: 573107 — Turma: 1CCPI
- João Pedro Soler — RM: 569725 — Turma: 1CCPI

---

## O que a Sprint 4 entrega

O diferencial da Sprint 4 é o **pipeline RAG (Retrieval-Augmented Generation)**
com grounding auditável:

- **Ingestão de PDFs** da base de conhecimento em uma coleção vetorial persistente
  no ChromaDB.
- **Busca semântica** dos trechos mais próximos da pergunta.
- **Prompt de grounding** que obriga o modelo a responder só com o `<context>`,
  citar `[arquivo.pdf, p. N]` e recusar quando não houver resposta.
- **Guardrails** (herdados da Sprint 3) aplicados **antes** de qualquer chamada ao
  modelo, mais um gate novo de "fora do contexto".
- **Interface web** em Gradio com conversa e coluna de fontes citadas clicáveis.
- **Memória por sessão** herdada da Sprint 3, anexada à pergunta antes da busca.
- **Avaliação automatizada** com RAGAS (*faithfulness*, *answer relevancy*) e uma
  rubrica manual de fallback.
- **Benchmark de modelos** (latência, citação, recusa) que fundamenta a escolha do
  modelo em produção.

---

## Arquitetura

Todo o fluxo de um turno converge para `src/rag/pipeline.py::responder()`, que é o
único caminho de execução — a interface (`app/`) e a avaliação (`evals/`) consomem
essa função em vez de remontar o fluxo.

```
pergunta
   |
   +-- 1. moderate()            [Sprint 3]  detecção de prompt injection
   +-- 2. validate_scope()      [Sprint 3]  escopo temático
   +-- 3. buscar_com_scores()   [Sprint 4]  busca vetorial + gate de contexto
   +-- 4. build_rag_chain()     [Sprint 4]  prompt de grounding | ChatOllama
   |
   +--> {"resposta": str, "fontes": [...], "bloqueado": bool}
```

Os três primeiros passos são locais e custam milissegundos. Uma pergunta bloqueada
**nunca chega ao LLM**: não consome token, não depende de rede e não produz
resposta. Uma falha de infraestrutura (Ollama fora do ar, base ausente) é tratada
separadamente, para não se confundir com uma recusa de guardrail.

A ordem dos passos é deliberada:

- **Moderação antes de escopo** — uma injeção como "ignore todas as regras" não tem
  palavra-chave de domínio e seria classificada como fora de escopo, devolvendo uma
  mensagem que mente sobre a causa.
- **Escopo antes da busca** — validação por regex, sem I/O; evita uma busca
  vetorial (que custa embedding) para uma pergunta já destinada à recusa.
- **Busca antes do LLM** — o limiar de contexto só existe depois que a busca
  devolve scores.

---

## Estrutura do projeto

```text
.
├── app/                          # Interface web (Gradio)
│   ├── app.py
│   └── README.md
├── data/
│   ├── knowledge_base/           # PDFs de origem
│   │   ├── manual_goodwe.pdf
│   │   ├── regimento_condominio.pdf
│   │   └── README.md
│   ├── chroma/                   # base antiga (gitignored)
│   └── chroma_db/                # base vetorial atual (gitignored)
├── docs/
│   ├── relatorio_modelos.md      # comparação de modelos no RAG
│   └── benchmark_modelos.json    # dados brutos de latência/citação
├── evals/                        # Avaliação do pipeline RAG
│   ├── README.md
│   ├── rag_cases.json            # 9 casos (pergunta + fontes esperadas)
│   ├── run_rag_eval.py           # executa os casos e calcula as métricas
│   ├── run_repetitions.py        # N execuções por prompt, com média/desvio
│   ├── ragas_eval.py             # registro do shim do RAGAS
│   ├── rubrica_manual.md         # rubrica 0-1 (fallback humano)
│   └── iteration_*.json          # resultados por iteração
├── prompts/
│   ├── system_prompt_rag_v3.0.txt
│   └── system_prompt_rag_v3.1.txt
├── scripts/
│   ├── benchmark_modelos.py      # latência, citação e recusa
│   └── demo_ingestion.py         # carga + chunking, sem Ollama/Chroma
├── src/rag/                      # Pipeline RAG (Sprint 4)
│   ├── loader.py                 # PDF -> Document (metadados normalizados)
│   ├── chunking.py               # split em chunks 500/100
│   ├── embeddings.py             # nomic-embed-text
│   ├── vector_store.py           # ChromaDB (collection goodwe_kb)
│   ├── ingest.py                 # pipeline de ingestão
│   ├── retriever.py              # busca + similaridade de cosseno
│   ├── prompt_rag.py             # prompt versionado + build_rag_chain
│   ├── guardrails_rag.py         # guardrails da Sprint 3 + gate de contexto
│   ├── chain.py                  # reexport de compatibilidade
│   └── pipeline.py               # responder() — orquestração ponta a ponta
├── tests/
│   └── test_app_rag_flow.py      # 37 testes do fluxo do app
├── goodwe-chatbot-sprint3/       # dependência: memória e guardrails (Sprint 3)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Tecnologias

- Python 3.12
- LangChain / `langchain-core` / `langchain-ollama` / `langchain-chroma`
- Ollama (local para embeddings; Ollama Cloud para o modelo de chat)
- ChromaDB (banco vetorial persistente)
- PyMuPDF + `langchain-text-splitters` (ingestão e chunking)
- RAGAS + `datasets` (avaliação)
- Gradio 6 (interface web)
- Pydantic v2, python-dotenv, pytest

Instalação:

```bash
pip install -r requirements.txt
```

---

## Configuração

Copie `.env.example` para `.env` e preencha:

```text
OLLAMA_API_KEY=            # chave da Ollama Cloud (vazia em Ollama local puro)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=              # opcional; sem ela vale MODEL_PADRAO do código
OLLAMA_EMBEDDING_MODEL=    # opcional; padrão nomic-embed-text
PROMPT_RAG_VERSION=3.1     # versão do prompt RAG (padrão: 3.1)
```

- O modelo de chat padrão é `gpt-oss:120b`, que roda na **Ollama Cloud** e exige
  `OLLAMA_API_KEY`. Modelos locais (ex.: `llama3`) funcionam sem credencial: a
  distinção está em `_build_model()` de `src/rag/prompt_rag.py`.
- Os **embeddings** (`nomic-embed-text`) rodam no Ollama local. O modelo de
  embeddings precisa ser o mesmo na ingestão e na consulta — ele define a dimensão
  do espaço vetorial.

---

## Ingestão da base

Rode uma vez (ou sempre que adicionar/remover PDFs):

```bash
python -m src.rag.ingest
```

O pipeline lê os PDFs de `data/knowledge_base/`, divide em chunks e grava os
vetores na collection `goodwe_kb` em `data/chroma_db/`. Sem isso, a busca não
retorna nada e toda pergunta é recusada pelo gate de contexto.

Para ver o chunking sem tocar em Ollama/Chroma:

```bash
python scripts/demo_ingestion.py
```

---

## Execução

### Interface web (Gradio)

A partir da raiz do projeto:

```bash
python -m app.app
# ou
gradio app/app.py
# porta personalizada:
python -m app.app --port 7890
```

Abre em `http://127.0.0.1:7860`. Layout em duas colunas: à esquerda a conversa; à
direita as **fontes do último turno**, no formato `[arquivo.pdf, p. X]`, com link
clicável para o PDF na página citada. Uma resposta bloqueada é exibida em destaque
e **sem fontes** — um guardrail recusa antes de consultar a base, então não há
documento honesto para citar.

### Avaliação (evals)

```bash
python evals/run_rag_eval.py                 # RAGAS (faithfulness + answer relevancy)
python evals/run_rag_eval.py --manual        # rubrica manual (fallback)
python evals/run_rag_eval.py --limit 2       # iteração rápida
PROMPT_RAG_VERSION=3.1 python evals/run_rag_eval.py --output evals/iteration_2.json
python evals/run_repetitions.py --versions 3.0 3.1 --repeticoes 3
```

Detalhes em [`evals/README.md`](evals/README.md).

### Benchmark de modelos

```bash
python scripts/benchmark_modelos.py
python scripts/benchmark_modelos.py --modelos gemma4:cloud gpt-oss:120b-cloud --repeticoes 3
```

Mede latência por turno (com a busca vetorial inclusa), fidelidade de citação e
taxa de recusa. Detalhes em [`docs/relatorio_modelos.md`](docs/relatorio_modelos.md).

### Testes

```bash
python -m unittest discover -s tests -v
pytest tests/test_app_rag_flow.py -v
```

Os 37 testes cobrem o fluxo de turno, a seção de fontes, o caminho de bloqueio, a
sessão e a linha de comando. **Nenhum teste toca Ollama, ChromaDB ou embeddings** —
`responder()` é mockado e a lógica de cada turno está em funções puras.

---

## Pipeline RAG

### Ingestão

- `loader.py` carrega cada PDF via `PyMuPDFLoader` e normaliza os metadados para
  exatamente `{source: <nome do arquivo>, page: <1-indexada>}`. O caminho completo
  do arquivo vaza para a citação e a página em base 0 faria o modelo citar "p. 0".
- `chunking.py` usa `RecursiveCharacterTextSplitter` com `chunk_size=500`,
  `chunk_overlap=100`. Com a base atual (2 PDFs, ~2065 caracteres) isso produz
  **6 chunks**, cada um com um tema dominante. Os separadores são ordenados do
  corte mais estrutural (parágrafo) para o menos, para não fragmentar seções.
- `vector_store.py` grava a collection `goodwe_kb` em `data/chroma_db/` usando
  `nomic-embed-text`.

### Recuperação

- `retriever.py` busca por similaridade com `k=4`.
- A similaridade é **cosseno** calculada a partir da distância do Chroma
  (`similarity_from_distance`). O scoring nativo do LangChain para espaço `l2`
  empurraria tudo para `[0.9987, 1.0]`, sem contraste para um limiar.
- `format_docs()` rotula cada trecho com `[arquivo.pdf, p. N]` no **mesmo formato**
  exigido para a citação, para que o modelo só copie o rótulo.

### Prompt e versões

- O prompt vive em `prompts/system_prompt_rag_v<X.Y>.txt`, nunca como string no
  código, para permitir comparar iterações.
- `PROMPT_RAG_VERSION` decide a versão; sem ela, vale `VERSAO_PROMPT_PADRAO = "3.1"`.
- `build_rag_chain(model=...)` monta a LCEL
  `retriever | format_docs | prompt | ChatOllama(temperature=0) | StrOutputParser`.
- O `<context>` é enviado como **mensagem de usuário**, não dentro do system
  prompt: conteúdo extraído de PDF é dado não confiável e precisa ficar em um nível
  de autoridade abaixo das regras.

### Memória (integração com a Sprint 3)

- `session_id` é opcional. Quando informado, os últimos turnos são transcritos em
  `<conversation_history>` e anexados à pergunta **antes** da busca vetorial, para
  que o embedding enxergue "e a corrente dele?" junto do turno que usa o "ele".
- `registrar_turno()` é separado de `responder()` e só roda quando a resposta
  **não** foi bloqueada: uma recusa não é uma conversa e não deve voltar como
  contexto.

### Guardrails

- `guardrails_rag.py` **importa** `moderate()` e `validate_scope()` da Sprint 3 em
  vez de copiá-los, para não manter duas listas de regex que divergem com o tempo.
- O único guardrail novo é o **gate de contexto** (`avaliar_contexto()`): se a
  busca não devolver nenhum trecho relevante, a recusa sai antes do LLM. O limiar
  é `LIMIAR_SIMILARIDADE = 0.30`, deliberadamente **não calibrado** (ver limitações).

---

## Avaliação

O pipeline é avaliado por dois eixos independentes, medidos pelo RAGAS:

- **Faithfulness** — a resposta se sustenta no contexto recuperado?
- **Answer relevancy** — a resposta responde ao que foi perguntado?

Com a base atual (2 PDFs) e o judge `gpt-oss:120b-cloud`, os resultados das
iterações estão em `evals/iteration_*.json`. A comparação honesta entre os prompts
v3.0 e v3.1 — inclusive o caso em que o judge penalizou uma recusa correta — está
documentada em [`evals/README.md`](evals/README.md).

O RAGAS 0.4.3 exige um **shim** (`evals/ragas_eval.py`) para importar neste
ambiente: ele registra em `sys.modules` um módulo sintético para satisfazer um
import legado do `langchain_community`, sem instalar ou rebaixar nada.

---

## Escolha do modelo

O relatório [`docs/relatorio_modelos.md`](docs/relatorio_modelos.md) compara
`gemma4:cloud`, `gpt-oss:120b-cloud` e `gpt-oss:20b-cloud` com o mesmo prompt,
casos e parâmetros. Resumo provisório:

- **`gemma4:cloud`** é o mais rápido e tem a melhor *answer relevancy*;
- a diferença de *faithfulness* entre os três é pequena demais para decidir;
- a recomendação é **provisória** — depende de um judge independente, repetições e
  uma base maior (itens marcados como TODO no relatório).

---

## Base de conhecimento

| Arquivo | Origem | Assunto |
|---|---|---|
| `manual_goodwe.pdf` | Fabricante (GoodWe) | Especificações técnicas, segurança elétrica, FAQ |
| `regimento_condominio.pdf` | Condomínio | Rateio de custos, regras de uso, restrições |

Ambos têm 1 página. Para acrescentar documentos sem descartar o índice atual use
`add_documents()` de `src/rag/vector_store.py`; reexecutar a ingestão inteira sobre
os mesmos arquivos cria chunks duplicados. Detalhes em
[`data/knowledge_base/README.md`](data/knowledge_base/README.md).

---

## Integração com a Sprint 3

A Sprint 3 (memória e guardrails) vive no projeto irmão `goodwe-chatbot-sprint3/`.
Os dois `src/` são **namespace packages** (PEP 420): com as duas raízes no
`sys.path`, o interpretador as funde em um único pacote, e `src.rag` (Sprint 4)
convive com `src.chain` e `src.guardrails` (Sprint 3).

> **Atenção:** essa fusão só funciona enquanto **nenhum** dos dois `src` tiver um
> `src/__init__.py`. Se algum ganhar um, o outro deixa de ser encontrado e os
> imports quebram com `ModuleNotFoundError`.

---

## Limitações conhecidas

1. **Caminho do banco vetorial divergente.** `vector_store.py` aponta para
   `data/chroma_db/`, mas há uma base antiga em `data/chroma/` (metadados pré-KB).
   Ambos estão no `.gitignore`; o app aponta para o diretório correto até a
   ingestão ser re-executada.
2. **`validate_scope()` tem falsos positivos.** Ele reprova perguntas legítimas do
   domínio e até saudações ("Bom dia", "Obrigado!"). A recalibração é trabalho da
   Sprint 3.
3. **Limiar de contexto não calibrado.** Com a base atual, `0.30` fica abaixo de
   toda a faixa de similaridades observada, então o gate só dispara quando a busca
   não devolve nada. Calibrar exige uma base maior e casos anotados.
4. **Base pequena.** Dois PDFs de uma página cada (6 chunks) — pouca granularidade
   para separar "a base respondeu" de "a busca não achou nada".
