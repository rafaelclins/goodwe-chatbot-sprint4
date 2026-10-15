# Rubrica de avaliação manual (fallback) — Sprint 4

Critérios de pontuação usados por `python evals/run_rag_eval.py --manual`.

Esta rubrica é o **fallback** do caminho automatizado. O primário é o RAGAS, com
`gpt-oss:20b` (Ollama) como judge, medindo as mesmas duas dimensões. Os dois
caminhos produzem o mesmo formato de JSON, o que permite comparar uma iteração
avaliada por LLM com outra avaliada por humano — desde que a comparação leve em
conta que são medidas de origens diferentes (ver
[Seção 6](#6-comparar-iterações)).

---

## 1. O que está sendo medido

Cada caso de `evals/rag_cases.json` é executado pelo pipeline real
(`src.rag.pipeline.responder`) e produz três artefatos: a **pergunta**, o
**contexto recuperado** (os chunks, já rotulados como
`[arquivo.pdf, p. N]`) e a **resposta do modelo**.

Duas dimensões, as mesmas do RAGAS:

| Métrica | Pergunta que responde |
| --- | --- |
| **Faithfulness** | A resposta se apoia **apenas** no contexto recuperado? |
| **Answer Relevancy** | A resposta endereça **diretamente** a pergunta feita? |

A separação importa porque elas falham de jeitos diferentes e opostos: uma
resposta pode ser perfeitamente fiel e irrelevante (cita o manual, mas não
responde), ou altamente relevante e infiel (resolve a pergunta, mas inventa um
número que não está em lugar nenhum).

---

## 2. Faithfulness (fidelidade / groundedness)

**Escala: 1 / 0.5 / 0**

### 2.1 `1` — Totalmente sustentada pelo contexto

Toda afirmação da resposta é rastreável a um trecho do contexto exibido, **e** a
citação de fonte confere com o rótulo `[arquivo.pdf, p. N]` que aparece no
mesmo contexto.

Exemplo — contexto:
```
[manual_goodwe.pdf, p. 1]
Corrente nominal: 32 A. Tensão de entrada: 380 V trifásico.
```
Resposta `1`: *"A corrente nominal é 32 A, com tensão de 380 V trifásico
[manual_goodwe.pdf, p. 1]."* → cada número existe no contexto e o rótulo citado
é literalmente um dos rótulos recuperados.

### 2.2 `0.5` — Parcialmente sustentada

Metade do conteúdo tem lastro no contexto; a outra metade não. Includes:

- um valor numérico que não aparece no contexto (ex.: o contexto diz 32 A e a
  resposta diz "32 A a 40 A conforme o modelo");
- um procedimento descrito de forma compatível, mas com um passo a mais que o
  contexto não traz;
- resposta fiel no conteúdo, mas cuja **citação aponta para a página errada** —
  o fato existe no documento, mas a âncora não bate. A citação é parte do
  contrato de grounding (regra do prompt V3.0), então um erro nela degrada a
  fidelidade sem ser alucinação pura.

### 2.3 `0` — Inventada (alucinação)

A resposta afirma qualquer coisa que **não está no contexto**:

- especificação, corrente, potência, prazo, norma ou procedimento que não
  aparecem em nenhum dos trechos recuperados;
- número inventado com precisão (`R$ 1,47/kWh` quando o contexto diz `R$ 0,95`);
- informação plausível e verdadeira no mundo, mas **não presente nesta base** —
  o RAG responde a partir do `<context>`, não do conhecimento geral do modelo.

> **Atenção.** O último item é o que mais importa nesta dimensão. Um modelo que
> responde corretamente algo que está no manual, mas usando conhecimento que
> **não** está nos trechos recuperados, ainda é infiel: o usuário foi servido por
> uma fonte que ele não consegue conferir. A regra é sempre "está *aqui*?",
> nunca "está certo?".

### 2.4 Como conferir

1. Leia o contexto exibido, trecho por trecho.
2. Para cada afirmação factual da resposta, aponte o trecho que a sustenta.
3. Uma afirmação sem lastro → `0.5` se as demais tiverem, `0` se a resposta
   inteira se apoiar nela.
4. Confira a citação contra os rótulos `[arquivo.pdf, p. N]` do contexto.

---

## 3. Answer Relevancy (relevância)

**Escala: 1 / 0.5 / 0**

### 3.1 `1` — Endereça diretamente

A resposta resolve a pergunta, sem desvios. Se a pergunta pede um dado
específico, esse dado está na resposta. Perguntar "quanto custa" e receber a
tarifa e a fórmula de cálculo é `1`.

### 3.2 `0.5` — Endereça parcialmente

A resposta trata do assunto mas não resolve:

- pede o cálculo mas **não diz como fazer** (ou o contrário);
- responde em parte e ompre o resto do que foi perguntado;
- responde à pergunta **errada** — a mais próxima, mas não a feita.

### 3.3 `0` — Não endereça

- Responde a outra coisa.
- É vazia, ou é só um erro.
- Recusa **sem necessidade**, isto é, quando a resposta estava no contexto.

### 3.4 Recusa de guardrail: `não avaliável`

> **Recusar não é automaticamente errado.** É o comportamento correto quando a
> informação não está na base — é a regra `<refusal>` do prompt V3.0, e o caso
> `conceptual_demand` de `rag_cases.json` existe para medir exatamente isso.

Uma recusa de guardrail **não recebe nota**: não houve resposta do modelo para
comparar com o contexto. O script marca o caso com `recusa_guardrail: true` e as
notas ficam `null`, fora da média. O que **penaliza** Relevância é recusar quando
o contexto **tinha** a resposta.

### 3.5 Erro de infraestrutura: `não avaliável`

Falha de infraestrutura (Ollama fora do ar, `OLLAMA_API_KEY` ausente, base não
ingerida) **também** não recebe nota, e é registrada separadamente como
`erro_infraestrutura: true`.

Não confundir com recusa. `responder()` devolve `bloqueado=True` nos dois casos,
mas um é o grounding funcionando e o outro é a avaliação inválida — contá-los
juntos daria a impressão de que o sistema recusou quando, na verdade, ele não
funcionou. Se a maioria dos casos der erro de infraestrutura, o número a
conservar é "não avaliado", não "não respondeu".

---

## 4. Justificativa

**Obrigatória** para cada nota, em uma linha.

Uma nota sem justificativa não pode ser revisada por outra pessoa, nem
reconciliada contra a do avaliador seguinte. Em caso de divergência entre
avaliadores, é a justificativa que permite decidir quem estava certo — e
reconhecer que o caso era ambíguo.

Formato sugerido:

```
Faithfulness: 0.5 — "32 A" está no contexto, mas a resposta afirma
                "R$ 0,95/kWh" e o regimento não foi recuperado.
Relevancy:    1   — a resposta entrega exatamente a tarifa pedida.
```

---

## 5. Como usar

```bash
python evals/run_rag_eval.py --manual
python evals/run_rag_eval.py --manual --limit 3
python evals/run_rag_eval.py --manual --output evals/iteration_2.json
```

O script executa o pipeline, imprime pergunta/contexto/resposta de cada caso e
pede as duas notas com justificativa. `q` pula um caso sem avaliá-lo;
`Ctrl+C` interrompe e salva o que já foi lido.

A rubrica completa também está disponível dentro do próprio script, para quem
está avaliando em fullscreen:

```
python evals/run_rag_eval.py --manual
```

---

## 6. Comparar iterações

Cada execução grava um arquivo próprio — `iteration_1.json`, `iteration_2.json`,
... — com, por caso, pergunta, resposta, contextos, fontes, as duas notas e a
justificativa, além do resumo com as médias.

O `--output` é o que torna a comparação possível: duas rodadas nunca disputam o
mesmo arquivo.

Ao comparar, confira sempre `resumo.avaliados`, `resumo.recusados_guardrail` e
`resumo.erros_infraestrutura` **antes** das médias. Uma média de 1.0 em três
casos e uma de 0.7 em nove não são a mesma coisa, e uma média alta com erros de
infraestrutura significa que pouco foi avaliado de verdade.

Ao comparar RAGAS com avaliação manual, lembre que o judge é uma LLM local
`gpt-oss:20b` e o avaliador humano é outra coisa. A diferença entre os dois
números não é automaticamente um defeito do prompt — pode ser critério de
avaliação diferente. Por isso as justificativas são gravadas no JSON.

---

## 7. Rubrica em uma tela

| Situação | Faithfulness | Relevancy |
| --- | --- | --- |
| Toda afirmação no contexto, citação confere | `1` | `1` |
| Responde certo, mas com um valor fora do contexto | `0.5` | `1` |
| Fiel e no ponto, porém cita a página errada | `0.5` | `1` |
| Dá a tarifa, mas não explica como calcular | `1` | `0.5` |
| Responde com conhecimento geral, fora dos trechos | `0` | `1` |
| Recusa sem necessidade, contexto tinha a resposta | `1` | `0` |
| Recusa de guardrail, contexto não tinha | `n/a` | `n/a` |
| Erro de infraestrutura (sem resposta gerada) | `n/a` | `n/a` |