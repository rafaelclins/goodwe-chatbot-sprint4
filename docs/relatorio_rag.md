# Relatório do pipeline RAG

Documentação técnica do pipeline RAG do chatbot GoodWe EV (Sprint 4). Todo o
conteúdo abaixo já existe no repositório (`README.md`, `evals/README.md`,
`src/rag/*.py`); nada foi estimado. Onde o dado não existe no repo, está marcado
como **(pendente de medição)**.

---

## 1. Visão geral

Todo o fluxo de um turno converge para `src/rag/pipeline.py::responder()`, que é o
único caminho de execução — a interface (`app/`) e a avaliação (`evals/`) consomem
essa função em vez de remontar o fluxo. Os três primeiros passos são locais e
custam milissegundos; uma pergunta bloqueada **nunca chega ao LLM** (não consome
token, não depende de rede e não produz resposta), e uma falha de infraestrutura
(Ollama fora do ar, base ausente) é tratada separadamente, para não se confundir
com uma recusa de guardrail.

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

---

## 2. Decisões de chunking

Parâmetros em `src/rag/chunking.py`: `CHUNK_SIZE = 500` e `CHUNK_OVERLAP = 100`.
Os parâmetros são medidos em **caracteres**, não em tokens.

Medição na base atual (2 PDFs, 2065 caracteres):

| configuração | chunks | média por chunk |
|---|---:|---:|
| 1000 / 200 | 3 | 740 chars |
| 750 / 150 | 4 | 560 chars |
| **500 / 100** | **6** | **398 chars** (adotado) |
| 300 / 60 | 9 | 227 chars |

Por quê:

- **1000 caracteres** (configuração anterior) misturava seções distintas no mesmo
  vetor: o manual GoodWe tem, em uma página, Especificações Técnicas, Segurança
  Elétrica e FAQ. Com 740 caracteres por chunk, uma busca por "disjuntor DR"
  devolvia um chunk majoritariamente sobre potência e Modbus, e o sinal relevante
  ficava diluído.
- **500 caracteres** cortam, na prática, na fronteira de seção: a média cai para
  398 caracteres e cada chunk carrega um tema dominante. É a faixa em que um chunk
  continua sendo uma unidade citável — perto de 300 caracteres a especificação
  passa a ser separada de sua unidade ou condição.
- **Overlap de 100 (20% do chunk)** cobre o custo dessa granularidade maior: o
  final de um chunk reaparece no início do seguinte, então o par valor/unidade
  sobrevive inteiro em pelo menos um dos dois.

**Ordenação dos separadores.** Os separadores são tentados do corte mais
estrutural para o menos; o `RecursiveCharacterTextSplitter` só recua para o
próximo quando o trecho excede `chunk_size`, então os primeiros definem a
granularidade real. Ordem em `SEPARADORES`:

1. `"\n\n"` — fim de parágrafo; fronteira semântica mais confiável (cada seção de
   um manual técnico é um parágrafo).
2. `"\n"` — quebra de linha solta, para blocos que separaram itens em linhas.
3. `r"\n\d+\.\s+"` — cabeçalho de seção numerada; fica abaixo de `"\n\n"` de
   propósito, para não fragmentar dois parágrafos que já cabem em 500 caracteres.
4. `". "`, `"? "`, `"! "` — fim de sentença, com a pontuação de fim do português.
5. `"; "` — separador de oração coordenada, corte mais fraco que o ponto.
6. `", "` e `" "` — últimos recursos, garantidos pelo separador vazio `""` no fim.

O separador vazio final é obrigatório: sem ele, texto que não contém nenhum dos
separadores acima seria devolvido inteiro, acima de `chunk_size`.

**Limite conhecido:** 500/100 é um ponto de partida medido para os 2065 caracteres
desta base, não um valor definitivo para um manual de 200 páginas. Trocar o
`chunk_size` sem um antes/depois medido é otimizar no escuro.

---

## 3. Base de conhecimento

| Arquivo | Origem | Assunto |
|---|---|---|
| `manual_goodwe.pdf` | Fabricante (GoodWe) | Especificações técnicas, segurança elétrica, FAQ |
| `regimento_condominio.pdf` | Condomínio | Rateio de custos, regras de uso, restrições |

Ambos têm 1 página; a base produz 6 chunks no total. Para acrescentar documentos
sem descartar o índice atual, use `add_documents()` de `src/rag/vector_store.py`.

---

## 4. Grounding e citação

- O prompt vive em `prompts/system_prompt_rag_v<X.Y>.txt` e o `<context>` é
  enviado como **mensagem de usuário**, não dentro do system prompt: conteúdo
  extraído de PDF é dado não confiável e precisa ficar em um nível de autoridade
  abaixo das regras.
- O formato de citação é `[arquivo.pdf, p. N]`. `format_docs()` rotula cada trecho
  no **mesmo formato** exigido para a citação, para que o modelo só copie o
  rótulo. O gate de contexto (`avaliar_contexto()`) usa
  `LIMIAR_SIMILARIDADE = 0.30` (`src/rag/guardrails_rag.py`): se a busca não
  devolver nenhum trecho, ou se o melhor score ficar abaixo do limiar, a recusa
  sai antes do LLM.
- A similaridade é de **cosseno**, em `[-1, 1]`, calculada a partir da distância do
  Chroma (`similarity_from_distance`) — não é a pontuação normalizada do LangChain
  (naquela escala os valores ficariam entre 0.9987 e 1.0).
- **Limiar não calibrado.** Com os embeddings atuais do `nomic-embed-text`, as
  similaridades entre documentos desta base ficam na faixa 0.69–0.74, acima de
  0.30: na prática, o gate só dispara quando a busca não devolve NENHUM trecho.
  Recalibrar exige uma base maior e casos com relevância anotada.

---

## 5. Segurança

A ordem dos passos é deliberada (e documentada em `README.md` e em
`guardrails_rag.py`):

- **Moderação antes de escopo** — uma injeção como "ignore todas as regras" não
  tem palavra-chave de domínio e seria classificada como fora de escopo,
  devolvendo uma mensagem que mente sobre a causa. `moderate()` roda primeiro por
  ser o filtro mais barato e o mais urgente.
- **Escopo antes da busca** — validação por regex, sem I/O; evita uma busca
  vetorial (que custa embedding) para uma pergunta já destinada à recusa.
- **Busca antes do LLM** — o limiar de contexto só existe depois que a busca
  devolve scores.

`guardrails_rag.py` **importa** `moderate()` e `validate_scope()` da Sprint 3 em
vez de copiá-los, para não manter duas listas de regex que divergem com o tempo.
O único guardrail novo é o gate de contexto.

---

## 6. Resultados

Tabela de versões do prompt RAG (de `evals/README.md`, números inalterados). Todas
as linhas usam o mesmo judge (`gpt-oss:120b-cloud`), os mesmos 9 casos e
`temperature=0`.

| Versão | Arquivo | O que mudou | Faithfulness | Answer relevancy | Ganho |
|---|---|---|---:|---:|---|
| v3.0 | `prompts/system_prompt_rag_v3.0.txt` | Linha de base: grounding, citação `[arquivo.pdf, p. N]`, recusa, defesa contra injeção via documento | 0,6085 | 0,6107 | — |
| v3.1 | `prompts/system_prompt_rag_v3.1.txt` | Procedimento de decisão (A)/(B)/(C) contra recusa indevida; citação por afirmação com lista de formatos proibidos; auto-check antes de responder | **0,5852** | **0,5657** | **−0,023 / −0,045** |

Ambiente da execução: Python 3.12, `langchain-core` 1.6.2, `ragas` 0.4.3,
`gemma4:cloud` e `gpt-oss:120b-cloud` via Ollama Cloud, `nomic-embed-text` local.
9/9 casos avaliados nas duas iterações, 0 recusas de guardrail, 0 erros de
infraestrutura.

- Latência por modelo (benchmark): ver `docs/benchmark_modelos.json` /
  `docs/relatorio_modelos.md`.
- Repetições por prompt (variabilidade): **(pendente de medição)** — todas as
  médias vêm de uma única rodada por combinação.
