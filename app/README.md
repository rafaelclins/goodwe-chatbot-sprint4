# `app/` — Interface web

Interface de chat do chatbot RAG (Sprint 4), integrada à Sprint 3 (Memória) e ao
pipeline com guardrails (`src/rag/pipeline.py`).

## Stack: Gradio

A interface **realmente implementada e testada** aqui usa **Gradio**
(`app/app.py`), com `gradio` declarado em `requirements.txt`.

> **Nota histórica.** Até a Sprint 4 esta interface foi escrita em Streamlit e
> chamava `build_rag_chain()` direto, sem gate de guardrail. Ela foi reescrita
> em Gradio e passou a consumir `src.rag.pipeline.responder()`, que aplica os
> guardrails da Sprint 3 **antes** de qualquer chamada ao modelo.
> `tests/test_app_rag_flow.py` foi reescrito junto e hoje cobre o caminho novo.

## Como executar

### 1. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 2. Ingerir a base vetorial (uma vez)

```bash
python -m src.rag.ingest
```

Sem isso, a busca no ChromaDB não retorna nada e toda pergunta é recusada pelo
gate de contexto.

### 3. Subir a interface

A partir da **raiz do projeto**:

```bash
python -m app.app
```

ou, equivalentemente:

```bash
gradio app/app.py
```

A aplicação abre em `http://127.0.0.1:7860`. Para escolher a porta:

```bash
python -m app.app --port 7890
```

O comando precisa ser executado da raiz: o app resolve a raiz do projeto por
`Path(__file__).resolve().parent.parent` e a adiciona ao `sys.path`, de modo que
o pacote `src` (que vive um nível acima) fique importável.

### Requisitos de execução

| Requisito | Para quê |
| --- | --- |
| Ollama rodando, com o modelo e `nomic-embed-text` baixados | responder e recuperar |
| Base ingerida em `data/chroma_db` | busca vetorial |
| `.env` com `OLLAMA_API_KEY` e o modelo configurado | `src/rag/prompt_rag.py` |

## Arquitetura

`app/app.py` é um frontend fino. Ele não faz retrieval, não monta prompt e não
chama o modelo: delega tudo para `responder()`.

```
app/app.py  (Gradio)
   └─> src/rag/pipeline.py :: responder(pergunta, session_id)
        ├─> src/rag/guardrails_rag.py  → moderate() + validate_scope()  [Sprint 3]
        ├─> src/rag/retriever.py       → buscar_com_scores()           [Sprint 4]
        ├─> src/rag/prompt_rag.py      → grounding + recusa + citação  [Sprint 4]
        ├─> src/chain/memoria.py       → histórico por session_id      [Sprint 3]
        └─> Ollama
   └─> registrar_turno(session_id, ...)  [Sprint 3] — só quando não bloqueado
```

`responder()` devolve sempre o mesmo dicionário:

```python
{"resposta": str, "fontes": [{"source": str, "page": int, "similaridade": float}], "bloqueado": bool}
```

## Layout

Duas colunas dentro de um `gr.Row`:

- **Coluna 1 — Conversa.** `gr.Chatbot` com o histórico da sessão, a caixa de
  pergunta e os botões *Enviar* / *Limpar*.
- **Coluna 2 — Fontes consultadas.** As citações do **último turno**, uma por
  linha, no formato `[arquivo.pdf, p. X]` com a similaridade do trecho.

A coluna direita é atualizada a cada resposta e reflete somente o turno atual,
não um acúmulo da conversa: as duas leituras ("o que sustenta a última
resposta?" e "o que já foi citado?") divergem a partir do segundo turno, e a
pergunta que o usuário faz ao ver a coluna é sempre a primeira.

### Citações clicáveis

Cada citação aponta para o PDF na página citada:

```
- [[manual_goodwe.pdf, p. 12]](/gradio_api/file=.../manual_goodwe.pdf#page=12) — similaridade 0.73
```

O link usa a rota `/gradio_api/file=` do Gradio 6 (a `/file=` da série 5.x
responde 404 na 6) e depende de `allowed_paths`, configurado no `launch()`.

Quando o PDF não existe em `data/knowledge_base` — base ingerida em outra
máquina, por exemplo — `formatar_citacoes()` devolve texto puro em vez de gerar
um link quebrado. Um link para arquivo inexistente é pior que texto: parece
clicável e não leva a lugar nenhum.

## Bloqueio: recusa em destaque, sem fontes

Quando `bloqueado=True`, o balão do assistente é renderizado como citação de
bloco (`>`) com o marcador 🚫, e a coluna direita mostra *"Nenhuma fonte
consultada"* explicando que o guardrail respondeu antes da busca.

Não há fonte para exibir nesse caminho: um guardrail bloqueia **antes** de
consultar a base, então não existe documento honesto para citar. Exibir uma
citação ao lado de uma recusa seria afirmar que o modelo leu algo que ele não
leu.

Dois motivos para a recusa não virar memória da Sprint 3: uma recusa não é uma
conversa. Registrada, ela voltaria como contexto na pergunta seguinte e faria o
modelo tratar o assunto barrado como se já tivesse sido discutido. Por isso
`registrar_turno()` só roda quando `bloqueado` é falso — o comportamento está
travado em `test_bloqueio_nao_registra_turno_na_memoria`.

## Sessão e histórico

O `session_id` é gerado uma vez por sessão do Gradio e guardado em `gr.State`.
Ele cumpre dois papéis:

- a memória da Sprint 3 é indexada por ele, então sessões diferentes não
  compartilham histórico;
- `responder()` o usa para anexar o `<conversation_history>` à pergunta **antes**
  da busca vetorial, porque o embedding precisa enxergar "e a corrente dele?"
  junto do turno que utiliza o "ele".

O botão **Limpar** zera a tela **e** a memória da sessão. Limpar só a tela
deixaria a conversa descartada voltando como contexto em um caminho que o
usuário não consegue ver.

## Testes

```bash
python -m unittest discover -s tests -v
pytest tests/test_app_rag_flow.py -v
```

37 testes cobrem o fluxo de turno, a seção de fontes, o caminho de bloqueio, a
sessão, a montagem da UI e a linha de comando.

**Nenhum teste toca Ollama, ChromaDB ou embeddings.** O Gradio 6 não distribui
harness de testes (não existe `gradio.testing_utils`, que no Streamlit fora o
`AppTest`), então os testes não sobem servidor nem reexecutam o script: eles
importam `app.app` e chamam as funções de turno com `responder()` mockado. Isso
é possível porque `criar_app()` monta os `gr.Blocks` sem lançar, e a lógica de
cada turno está em funções puras (`formatar_resposta`, `formatar_bloco_fontes`,
`formatar_citacoes`).

Para verificar a UI de verdade, com o `responder` mockado:

```bash
python -m app.app
```

## Por que `src/` não tem `__init__.py`

Os dois `src` (Sprint 4 e Sprint 3) são *namespace packages* (PEP 420). Com as
duas raízes no `sys.path`, o interpretador os funde em um único pacote `src`, e
`src.rag` convive com `src.guardrails` e `src.chain` da Sprint 3.

**Atenção:** essa fusão só funciona enquanto nenhum dos dois `src` tiver um
`src/__init__.py`. Se algum deles ganhar um, o outro deixa de ser encontrado e
os imports da Sprint 3 quebram com `ModuleNotFoundError`. Por isso este `app/`
pode ter `__init__.py`, mas `src/` não pode.

## Inconsistências conhecidas

1. **Caminho do banco vetorial divergente.** `src/rag/vector_store.py` define
   `CHROMA_PATH = "data/chroma_db"`, mas o banco existente em disco está em
   `data/chroma/`, de uma ingestão antiga com metadados pré-KB. Ambos estão no
   `.gitignore`, então nada é versionado por engano, mas o app aponta para um
   diretório vazio até a ingestão ser re-executada.
2. **`validate_scope()` tem falsos positivos conhecidos.** Ele classifica como
   fora de escopo perguntas legítimas do domínio e até saudações — `"Bom dia"` e
   `"Obrigado!"` são reprovados. A interface herda esse comportamento porque o
   app consome o pipeline com guardrails. Recalibrar o classificador é trabalho
   da Sprint 3, não desta camada.