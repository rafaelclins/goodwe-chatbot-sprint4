# `evals/` — Avaliação do pipeline RAG

Mede se o chatbot responde **apenas** com o que está na base de conhecimento e
**responde** o que foi perguntado. As duas dimensões são independentes e falham
de jeitos opostos: uma resposta pode ser fiel e irrelevante (cita o manual, mas
não responde) ou relevante e infiel (resolve a pergunta, mas inventa um número).

## Arquivos

| Arquivo | Papel |
|---|---|
| `rag_cases.json` | Casos de teste: pergunta, fontes esperadas e o comportamento correto |
| `run_rag_eval.py` | Executa os casos pelo pipeline real e calcula as métricas |
| `rubrica_manual.md` | Rubrica 0-1 para avaliação humana (fallback do RAGAS) |
| `ragas_eval.py` | Registro de por que o RAGAS foi abandonado e o que o shim faz |
| `iteration_1.json` | v3.0 + `gemma4:cloud` |
| `iteration_2.json` | v3.1 + `gemma4:cloud` |
| `iteration_3.json` | v3.1 + `gpt-oss:120b-cloud` |
| `iteration_4.json` | v3.1 + `gpt-oss:20b-cloud` |

---

## Como executar

```bash
# Iteração automatizada (RAGAS)
python evals/run_rag_eval.py

# Avaliação humana, com a rubrica de evals/rubrica_manual.md
python evals/run_rag_eval.py --manual

# Versão específica do prompt, sem editar código
PROMPT_RAG_VERSION=3.0 python evals/run_rag_eval.py --output evals/iteration_1.json
PROMPT_RAG_VERSION=3.1 python evals/run_rag_eval.py --output evals/iteration_2.json

# Modelo local em vez da nuvem
python evals/run_rag_eval.py --rag-model llama3 --judge-model llama3

# Iteração rápida
python evals/run_rag_eval.py --limit 2
```

Pré-requisitos: Ollama rodando, base ingerida (`python -m src.rag.ingest`) e o
modelo de judge disponível.

---

## Tabela de versões do prompt RAG

Todas as linhas usam o mesmo judge (`gpt-oss:120b-cloud`) e `temperature=0`.
As iterações 1 e 2 (v3.0/v3.1) foram medidas em 9 casos; a iteração 5 (v3.2)
foi medida em 15 casos (base `evals/rag_cases.json` expandida), então o ganho
direto entre iterações com bases diferentes não é estritamente comparável.

| Versão | Arquivo | O que mudou | Faithfulness | Answer relevancy | Ganho |
|---|---|---|---:|---:|---|
| v3.0 | `prompts/system_prompt_rag_v3.0.txt` | Linha de base: grounding, citação `[arquivo.pdf, p. N]`, recusa, defesa contra injeção via documento | 0,6085 | 0,6107 | — |
| v3.1 | `prompts/system_prompt_rag_v3.1.txt` | Procedimento de decisão (A)/(B)/(C) contra recusa indevida; citação por afirmação com lista de formatos proibidos; auto-check antes de responder | **0,5852** | **0,5657** | **−0,023 / −0,045** |
| v3.2 | `prompts/system_prompt_rag_v3.2.txt` | Primeira frase responde diretamente (sem introdução genérica); citação única ao final de cada parágrafo `[arquivo.pdf, p. X]`; mantém (A)/(B)/(C), auto-check e defesa contra injeção | **0,7315** | **0,5090** | **−0,08 / −0,08** |

> **O v3.2 perdeu nas duas métricas (comparação justa).** Como o v3.2 foi
> medido em 15 casos e o v3.1 na tabela acima em 9 casos, o ganho da coluna
> "Ganho" (−0,08 / −0,08) refere-se à comparação nos mesmos 15 casos:
> v3.1 = 0,8130 / 0,5933 versus v3.2 = 0,7315 / 0,5090. As duas mudanças
> cirúrgicas do v3.2 (primeira frase direta e citação no fim do parágrafo)
> reduziram tanto a fidelidade quanto a relevância, e a taxa de recusa correta
> manteve-se em 1,00 nas duas versões. **Conclusão: o v3.2 não é uma
> melhoria; manter o v3.1 como versão ativa.**

> **O v3.1 perdeu nas duas métricas.** Os números acima são o resultado da
> execução, não o esperado. A seção seguinte explica o que aconteceu e por que
> não é um resultado para descartar sem mais dados.

### O que o v3.1 tentava corrigir

Rodando o v3.0 com `gemma4:cloud`, a falha dominante era **recusa indevida**,
não alucinação. No caso `conceptual_dr` ("O que é um DR na instalação de
recarga?"), o contexto trazia *"é obrigatório o uso de disjuntor DR Tipo A
(30mA) para proteção contra fuga de corrente residual"* e o v3.0 respondeu
*"não encontrei essa informação na base de conhecimento"*. O v3.0 proibia
inventar em nove linhas e nunca disse, em momento nenhum, que recusar também é
erro quando o contexto tem o dado.

### O que o v3.1 conseguiu

| Medida | v3.0 | v3.1 | Efeito |
|---|---:|---:|---|
| Citações na resposta | 8 | 12 | +50% |
| Citações com fonte válida no contexto | 8 (100%) | 12 (100%) | — |
| Respostas com recusa | 5 (56%) | 2 (22%) | −60% |
| Formatos de citação proibidos | 0 | 0 | — |
| `conceptual_dr` — respondia? | não | **sim** | corrigido |
| `happy_kwh_concept` — respondia? | não | **sim** | corrigido |

A meta do v3.1 foi atingida: a recusa indevida caiu de 5 casos para 2, e as
citações subiram com 100% de acerto de fonte. **Nenhuma citação incorreta foi
introduzida** — o `p. N` continuei sempre correspondente a um rótulo real do
contexto.

### Por que, então, as métricas caíram

O ganho de comportamento não apareceu nas médias, por dois motivos que os dados
separam:

**1. O caso `conceptual_demand` inverteu sozinho.** É o teste negativo: a
expressão "demanda contratada" não existe em nenhum dos dois PDFs, e o
comportamento correto é recusar. Nos dois prompts o modelo recusa — a resposta
está correta nos dois casos. Mas o judge deu:

| | v3.0 | v3.1 |
|---|---:|---:|
| `conceptual_demand` — recusou? | sim | sim |
| `conceptual_demand` — faithfulness | 0,00 | 0,00 |
| `conceptual_demand` — answer relevancy | **0,81** | **0,00** |

O mesmo comportamento produziu notas diferentes. A relevância dessa recusa
certa caiu de 0,81 para 0,00 **porque a redação do segundo parágrafo mudou** —
o v3.1 pede que a recusa diga "o que seria necessário para responder", e o judge
penalizou a resposta por não ser mais uma reformulação da anterior. Um único
caso com essa amplitude (0,81) vale 0,09 na média de 9.

**2. `ignore_false_positive` perdeu relevância.** O v3.1 respondeu 2 casos que o
v3.0 recusava (`happy_kwh_concept`, `conceptual_dr`) — o objetivo — mas também
passou a responder com menos disciplina em casos onde a resposta no contexto era
parcial. A média de faithfulness caiu de 0,6085 para 0,5852.

### Leitura honesta

Com 9 casos, uma execução e um judge que é uma LLM, **estas diferenças não
soportam decisão de troca de prompt**. O que os dados sustentam:

- o v3.1 **corrige** um defeito real e reproduzível (recusa indevida), e o faz
  sem introduzir citação inventada;
- o v3.1 **não demonstrou** ganho nas métricas do RAGAS, e o caso que mais pesa
  contra ele é um artefato de judge, não de comportamento;
- o custo é de tamanho: o prompt dobrou (6,9 KB → 13,5 KB).

O que falta para decidir:

1. **Repetir a execução.** Com `temperature=0` a variabilidade vem do judge e da
   rede. Duas ou três repetições por prompt diriam se 0,6085 × 0,5657 é diferença
   ou ruído.
2. **Corrigir o `conceptual_demand` na métrica.** Como `evals/rubrica_manual.md`
   já estabelece (§3.4), recusar corretamente não deveria penalizar relevância —
   lá isso vale `n/a`, mas o RAGAS está pontuando. Enquanto o judge e a rubrica
   divergirem nesse ponto, a métrica mede o judge, não o grounding.
3. **Ampliar a base.** 9 casos dão 0,11 de granularidade por caso. As duas PDFs
   da base não sustentam mais que isso.

**Recomendação: manter `VERSAO_PROMPT_PADRAO = "3.1"` no código** (o defeito
corrigido é real e documentado), **com a decisão de adoção formal pendente das
três verificações acima.** Não registrar o v3.1 como melhoria de score.

### Como reproduzir esta comparação

```bash
PROMPT_RAG_VERSION=3.0 python evals/run_rag_eval.py \
    --rag-model gemma4:cloud --judge-model gpt-oss:120b-cloud \
    --output evals/iteration_1.json

PROMPT_RAG_VERSION=3.1 python evals/run_rag_eval.py \
    --rag-model gemma4:cloud --judge-model gpt-oss:120b-cloud \
    --output evals/iteration_2.json
```

Ambiente desta execução: Python 3.12, `langchain-core` 1.6.2, `ragas` 0.4.3,
`gemma4:cloud` e `gpt-oss:120b-cloud` via Ollama Cloud, `nomic-embed-text`
local. 9/9 casos avaliados nas duas iterações, 0 recusas de guardrail, 0 erros de
infraestrutura.

---

## Escolha do prompt em uso

`PROMPT_RAG_VERSION` (ver `.env.example`) decide qual arquivo
`prompts/system_prompt_rag_v<X.Y>.txt` a chain carrega. Sem ela, vale
`VERSAO_PROMPT_PADRAO` em `src/rag/prompt_rag.py`. Apontar para uma versão
inexistente faz o pipeline levantar um erro **listando as versões disponíveis** —
a falha mais provável é um erro de digitação, não um arquivo ausente de fato.

## Comparar iterações

Cada execução grava um arquivo próprio, e o JSON registra `prompt_rag_versao`,
`modelo_rag`, `modelo_judge` e `embedding`. Sem esses metadados, dois arquivos
com a mesma estrutura seriam indistinguíveis e a comparação não teria base.

Ao comparar, confira **antes** das médias:

- `resumo.avaliados` — quantos casos realmente entraram na média
- `resumo.recusados_guardrail` — recusados por filtro local, sem resposta do modelo
- `resumo.erros_infraestrutura` — Ollama fora do ar, base ausente; contam como
  falha, não como recusa

Uma média de 1,00 em 3 casos e uma de 0,70 em 9 não são a mesma coisa.

## O que o RAGAS não mede

`answer_relevancy` é a métrica que penaliza a recusa indevida — mas o RAGAS a
calcula por similaridade de embeddings entre pergunta e resposta, e uma recusa
correctamente formulada tem baixa similaridade com a pergunta. Por isso o
comportamento certo e o comportamento bem avaliado divergem justamente em
`conceptual_demand`. `faithfulness` não sofre disso: compara afirmações contra o
contexto, e uma recusa não afirma nada para sustentar.

Para medir recusa corretamente, a contagem direta (quantos casos responderam
versus recusaram) é mais confiável que a nota. Ver a tabela "O que o v3.1
conseguiu" acima: ela usa contagem, não similaridade.

## Limitações conhecidas

1. **Base pequena.** Dois PDFs de uma página cada, 6 chunks. Com uma base de
   uma página, "a base respondeu" e "a busca não achou nada" são difíceis de
   separar, e o gate de contexto (`LIMIAR_SIMILARIDADE` em
   `src/rag/guardrails_rag.py`) não tem distribuição para ser calibrado.

2. **Judge igual ao avaliado na iteração 3.** Em `iteration_3.json` o modelo do
   RAG e o judge são ambos `gpt-oss:120b-cloud`. Modelo que se autoavalia tende a
   pontuar a si mesmo mais alto. Use `iteration_3.json` para latência e formato
   de citação, não para comparar qualidade com as demais.

3. **Uma execução por combinação.** Todas as médias vêm de uma rodada. Ver
   "Leitura honesta" acima.