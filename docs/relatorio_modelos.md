# Relatório de modelos — RAG com grounding

Comparação de modelos de chat no pipeline RAG da Sprint 4, medidos pelo mesmo
prompt (v3.1), pelos mesmos casos (`evals/rag_cases.json`) e com os mesmos
parâmetros de geração.

Os dados vêm de duas fontes, ambas nesta pasta:

- **`../evals/iteration_*.json`** — notas de *faithfulness* e *answer relevancy*
  do RAGAS, com `gpt-oss:120b-cloud` como judge.
- **`benchmark_modelos.json`** — latência e taxa de citação, medidos por
  `scripts/benchmark_modelos.py` sobre o pipeline real.

Nenhum número deste documento foi estimado. Onde não houve medição, o campo está
marcado como **TODO** com a instrução de preenchimento.

---

## Modelos comparados

| Modelo | Onde roda | Acesso |
|---|---|---|
| `gemma4:cloud` | Ollama Cloud | sufixo `-cloud` no servidor local autenticado |
| `gpt-oss:120b` | Ollama Cloud | sufixo `-cloud` no servidor local autenticado |
| `gpt-oss:20b` | Ollama Cloud | sufixo `-cloud` no servidor local autenticado |

> **Sobre o nome.** `gpt-oss:120b` **não** é o identificador aceito por um
> servidor local autenticado: é preciso `gpt-oss:120b-cloud`. O nome sem
> sufixo é o identificador da API direta (`https://ollama.com/api/tags`, ex.
> `gemma4:31b`). A verificação de `evals/run_rag_eval.py` foi corrigida para
> sondar `/api/show` em vez de consultar `/api/tags`, porque a lista de tags do
> servidor local não contém os modelos de nuvem até o primeiro uso bem-sucedido —
> nela, `gpt-oss:120b-cloud` responderia "modelo indisponível" mesmo
> funcionando.

---

## Parâmetros de geração

Valores lidos do payload que `ChatOllama` realmente envia, extraído de
`_chat_params()` em `langchain-ollama` (versão instalada junto ao projeto):

```python
{"keep_alive": null, "logprobs": null, "model": "<modelo>",
 "options": {"temperature": 0.0}, "think": null, "top_logprobs": null}
```

| Parâmetro | Valor enviado | Quem define |
|---|---|---|
| `temperature` | **0.0** | `src/rag/prompt_rag.py` → `ChatOllama(temperature=0)` |
| `top_k` | **ausente** | não é enviado; o Ollama aplica o default do modelo |
| `top_p` | **ausente** | idem |
| `num_predict` (`max_tokens`) | **ausente** | idem — sem teto de tamanho de resposta |
| `num_ctx` | ausente | default do modelo |
| `seed` | ausente | sem fixação de semente |
| `think` | ausente | o servidor decide |

Os três modelos recebem **exatamente o mesmo payload**. Não há tabela por modelo
nesta seção porque a implementação não diferencia os modelos: `build_rag_chain()`
constrói o `ChatOllama` da mesma forma para qualquer nome.

### Por que `temperature=0`

`temperature=0` torna a geração determinística, e sem isso qualquer diferença de
score entre duas iterações de prompt vira ruído de amostragem em vez de evidência
de que o prompt mudou alguma coisa.

O custo: a família Gemma recomenda `temperature=1.0`, `top_p=0.95`,
`top_k=64` para seus melhores resultados. O projeto usa `0` em todos os modelos por uniformizar essa variável — comparar
modelos com temperaturas diferentes mediria a temperatura, não o modelo. **É um
desvio consciente do recomendado, e vale reavaliar se a qualidade importa mais
que a comparabilidade.**

### `num_predict` ausente tem consequência

Sem `num_predict`, o modelo pode gerar até o limite do contexto. Nenhum caso
deste conjunto atingiu esse limite, mas numa base maior um turno sem teto de
saída pode gerar centenas de tokens de recusa repetitiva.

---

## Resultados

### Qualidade (RAGAS, judge `gpt-oss:120b-cloud`, prompt v3.1, n=9)

| Modelo | Iteração | Faithfulness | Answer relevancy | Citações | Citação válida |
|---|---|---:|---:|---:|---:|
| `gemma4:cloud` | `iteration_2.json` | 0,5852 | 0,5657 | 12 | 100% |
| `gpt-oss:120b-cloud` | `iteration_3.json` | 0,5648 | 0,5142 | 12 | 100% |
| `gpt-oss:20b-cloud` | `iteration_4.json` | 0,5926 | 0,4878 | 7 | 100% |

### Latência e comportamento (`benchmark_modelos.json`, 6 perguntas, 1x cada)

| Modelo | Latência média | Mediana | Mín | Máx | Com citação | Recusou |
|---|---:|---:|---:|---:|---:|---:|
| `gemma4:cloud` | **9,30s** | 9,29s | 8,72s | 9,90s | 100% | 0% |
| `gpt-oss:120b-cloud` | 11,39s | 11,14s | 9,97s | 14,25s | 100% | 0% |
| `gpt-oss:20b-cloud` | 14,12s | 13,32s | 12,19s | 17,65s | 83% | 0% |

Latência é do `responder()` completo, **incluindo a busca vetorial** — que é o
que o usuário espera. O spread de um mesmo modelo (9,97s a 14,25s no
`gpt-oss:120b`) mostra que, nesta amostra, a rede da nuvem vale mais que a
diferença entre os modelos.

---

## Leitura

**`gemma4:cloud` é a escolha hoje**, por duas razões que os dados apoiam:

1. **É o mais rápido** (9,30s contra 11,39s e 14,12s) e tem o menor spread
   (8,72s–9,90s; o `gpt-oss:120b` variou de 9,97s a 14,25s **no mesmo modelo**).
2. **Tem a melhor *answer relevancy*** (0,5657 contra 0,5142 e 0,4878), a métrica
   que mais distingue os três — e é a que penaliza a recusa indevida.

**O que os dados não sustentam:**

- **"`gemma4:cloud` tem o melhor *faithfulness*."** Não tem. O
  `gpt-oss:20b-cloud` ficou **acima**, em 0,5926 contra 0,5852. A diferença
  (0,007) é pequena demais para sustentar ordenação, mas a afirmação de que o
  `gemma4` leader não é verdadeira.

- **"Cita mais que os outros."** Não. O `gemma4:cloud` e o `gpt-oss:120b-cloud`
  emitiram **12 citações cada**; o `gpt-oss:20b-cloud` emitiu 7. A diferença real
  é do `20b`, que deixou **1 das 6** respostas do benchmark sem citação alguma.

- **"O argumento decisivo é o número de parâmetros."** Removido: este relatório
  **não mediu** o tamanho dos modelos, e afirmar "um terço dos parâmetros" sem
  medição seria inventar o dado que a tarefa pediu para não inventar. Ver o TODO
  de custo por token abaixo.

**As três ressalvas que impedem uma recomendação firme:**

1. **O `gpt-oss:120b` foi julgado por si mesmo** em `iteration_3.json`. Modelo
   que se autoavalia tende a pontuar mais alto, e ainda assim ficou abaixo do
   `gemma4` em ambas as métricas. Isso é um indício a favor do `gemma4`, não uma
   prova.

2. **Uma execução por modelo.** As três médias vêm de uma rodada. A diferença
   entre 0,5852 e 0,5926 (0,007) é menor que a variação entre execuções do mesmo
   modelo que o spread de latência sugere.

3. **O benchmark tem 6 perguntas e 1 repetição** — não é amostra suficiente para
   taxa de citação. O `83%` do `gpt-oss:20b` é **1 resposta de 6**.

---

## Conclusão

Com os dados medidos: **`gemma4:cloud` é o mais rápido** (9,30s, ~18% abaixo do
`gpt-oss:120b`) **e tem o melhor *answer relevancy***, com *faithfulness*
 estatisticamente empatado com o `gpt-oss:20b` (0,5852 × 0,5926).

A diferença de qualidade é pequena e não está dentro do ruído de uma execução
única. A de latência (2,1s por turno) é perceptível mas não decisiva sozinha.

**Recomendação provisória:** `gemma4:cloud`, por ser o mais rápido sem ceder
qualidade mensurável. **Não é uma conclusão firme** — depende dos quatro itens
de TODO abaixo, em particular de um judge que não seja um dos candidatos.

### TODO — pendências de medição

Estes campos ficaram sem medição e **não** foram preenchidos por estimativa:

- [ ] **Custo por turno em tokens (cloud).** As iterações aqui rodaram com o
      Ollama Cloud autenticado no servidor local; a cobrança aparece em
      <https://ollama.com/settings/usage>. Para comparar custo:
      ```bash
      python -c "import sys; sys.path.insert(0,'.'); \
        from src.rag.retriever import buscar_com_scores; \
        print(sum(len(d.page_content) for d,_ in buscar_com_scores('O que e kWh?')))"
      ```
      Some os tokens de entrada (system prompt + contexto + pergunta) com o
      tamanho da resposta, e multiplique pelo preço do modelo. Anote a data: o
      preço por 1M de tokens varia.

- [ ] **Qualidade com judge independente de todos os modelos.** As três linhas da
      tabela de qualidade usam `gpt-oss:120b-cloud` como judge, e uma delas o
      avalia como candidato. Rodar com um quarto modelo como judge eliminaria a
      dependência:
      ```bash
      python evals/run_rag_eval.py --rag-model gemma4:cloud \
          --judge-model <modelo-avaliado-em-nenhuma-outra-rodada> \
          --output evals/iteration_5.json
      ```

- [ ] **Repetições.** Repetir cada combinação 3× e reportar média e desvio:
      ```bash
      PROMPT_RAG_VERSION=3.1 python evals/run_rag_eval.py --rag-model gemma4:cloud \
          --judge-model gpt-oss:120b-cloud --output evals/iter_1_gemma_run1.json
      # ... run2, run3
      python scripts/benchmark_modelos.py --modelos gemma4:cloud --repeticoes 3
      ```

- [ ] **Comportamento local (`llama3`, 8B, CPU).** Medido nesta máquina apenas
      para comparação de velocidade: **2,4 tok/s**, 30s de carregamento, contra
      9,9 tok/s do `qwen3.5:0.8b` e rede do `gemma4:cloud`. Com 7,9 GB de RAM e
      sem GPU, é inviável para o RAG. Registrado aqui porque é o modelo padrão
      quando `PROMPT_RAG_VERSION` não está em modelo de nuvem, e o número
      justifica não usá-lo nos evals.

---

## Ambiente das medições

| Componente | Versão |
|---|---|
| Python | 3.12 |
| `langchain-core` | 1.6.2 |
| `langchain` | 1.4.0 |
| `ragas` | 0.4.3 |
| Modelo de embeddings | `nomic-embed-text` (local, `data/chroma_db`) |
| Judge | `gpt-oss:120b-cloud` |
| Prompt | `prompts/system_prompt_rag_v3.1.txt` |
| Casos | `evals/rag_cases.json` (9 casos, 9/9 avaliados) |
| Máquina | Windows, 8 threads lógicos, 7,9 GB RAM, **sem GPU** |

Cloud exige Ollama local autenticado (`ollama signin`) ou
`OLLAMA_API_KEY` no `.env`. Modelos locais continuam funcionando sem credencial —
a distinção está em `_build_model()` (`src/rag/prompt_rag.py`).

## Como reproduzir

```bash
python -m src.rag.ingest          # base vetorial

python scripts/benchmark_modelos.py \
    --modelos gemma4:cloud gpt-oss:120b-cloud gpt-oss:20b-cloud \
    --saida docs/benchmark_modelos.json

PROMPT_RAG_VERSION=3.1 python evals/run_rag_eval.py \
    --rag-model gemma4:cloud --judge-model gpt-oss:120b-cloud \
    --output evals/iteration_2.json
```