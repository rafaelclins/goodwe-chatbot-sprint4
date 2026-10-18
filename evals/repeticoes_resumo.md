# Repetições do eval RAG — v3.0 × v3.1 (3 execuções por versão)

**Configuração:** `PROMPT_RAG_VERSION` 3.0 / 3.1 · RAG `gemma4:cloud` · judge `gpt-oss:120b-cloud` · embedding `nomic-embed-text` · 15 casos (13 answerable + 2 refusal_expected)

**Arquivos auditados:**

| Versão | Rodadas |
|---|---|
| v3.0 | `evals/iter_1_v30_run1.json`, `iter_1_v30_run2.json`, `iter_1_v30_run3.json` (15/10) |
| v3.1 | `evals/iter_2_v31_run1.json`, `iter_2_v31_run2.json`, `iter_2_v31_run3.json` (16/10) |

Todas as 6 rodadas rodaram o conjunto completo de 15 casos, sem falha de infraestrutura (nenhum `erro_infraestrutura`). A média abaixo é a média das médias por rodada; o desvio é o desvio padrão **amostral entre as 3 rodadas** — é ele que diz se a nota se repete.

## Métricas (bloco answerable, RAGAS)

| Métrica | v3.0 (n=3 rodadas) | v3.1 (n=3 rodadas) | Δ (v3.1 − v3.0) |
|---|---|---|---|
| faithfulness | **0,7871 ± 0,1302** (min 0,6500 / max 0,9091) | **0,8086 ± 0,0533** (min 0,7583 / max 0,8646) | +0,0215 |
| answer_relevancy | **0,4693 ± 0,0197** (min 0,4467 / max 0,4829) | **0,5881 ± 0,0218** (min 0,5629 / max 0,6020) | **+0,1188** |
| recusa correta (2 casos/rodada) | 6/6 | 6/6 | — |

Notas por rodada:

| Rodada | v3.0 f | v3.0 r | v3.1 f | v3.1 r |
|---|---|---|---|---|
| 1 | 0,6500 | 0,4467 | 0,8030 | 0,5629 |
| 2 | 0,8021 | 0,4784 | 0,8646 | 0,6020 |
| 3 | 0,9091 | 0,4829 | 0,7583 | 0,5993 |

## 0,6085 × 0,5657: diferença ou ruído?

Os números de uma execução única (`iteration_1.json` × `iteration_2.json`, base antiga de 9 casos: f 0,6085 × 0,5852 e r 0,6107 × 0,5657) sugeriam que o v3.1 piorou. Com 3 repetições na base atual de 15 casos, **essa leitura não se sustenta em nenhuma das duas métricas** — e em answer_relevancy a diferença aparece com o sinal trocado:

**faithfulness — é ruído.** Os intervalos por rodada se sobrepõem ([0,6500; 0,9091] × [0,7583; 0,8646]), o delta de +0,0215 é menor que o desvio de qualquer das versões, e um teste t de Welch sobre as médias por rodada dá t ≈ 0,27 (p ≈ 0,81). Além disso, o desvio de v3.0 (±0,13) é 2,4× o de v3.1 (±0,05): a maior parte da "diferença" observada antes era a dispersão do próprio v3.0, que numa rodada sorteada deu 0,65 e noutra 0,91. Dizer que um prompt é mais fiel que o outro com esses dados é ler sorte.

**answer_relevancy — é diferença, e a direção é a inversa da que os números únicos sugeriam.** Os intervalos estão separados ([0,4467; 0,4829] × [0,5629; 0,6020] — o pior v3.1 ainda bate o melhor v3.0), a dispersão entre rodadas é minúscula nas duas versões (±0,020 e ±0,022) e o teste t dá t ≈ 7,0 (p ≈ 0,002, df ≈ 4). O v3.0 é consistentemente ~0,47 em toda rodada; o v3.1, consistentemente ~0,59. Não é amostra: o v3.1 responde com mais pertinência à pergunta.

O ganho não vem de um caso isolado — quatro casos puxam, com o maior deles subindo +0,71:

| Caso (answer_relevancy, média das 3 rodadas) | v3.0 | v3.1 | Δ |
|---|---|---|---|
| calculation_charge_cost | 0,225 | 0,932 | +0,706 |
| conceptual_dr | 0,000 | 0,501 | +0,501 |
| happy_kwh_concept | 0,233 | 0,680 | +0,447 |
| memory_session_context | 0,531 | 0,799 | +0,268 |
| chargegrid_specs | 0,815 | 0,637 | −0,177 |
| faq_charging_time | 1,000 | 0,853 | −0,147 |

(v3.0 teve 0,000 em conceptual_dr nas três rodadas; calculation_charge_cost e memory_session_context também subiram de 0,00/resposta irrelevante para resposta pertinente.)

**Conclusão:** 0,6085 × 0,5657 (e o 0,6107 × 0,5657 de relevância) era **ruído de execução única lido como sinal** — inclusive no sinal. Com repetição, faithfulness não diferencia os prompts (p ≈ 0,81) e answer_relevancy diferencia sim, com o v3.1 **acima** do v3.0 por ~0,12 (p ≈ 0,002). A escolha do v3.1 como iteração a manter se confirma; o que não se sustenta é a leitura de que ele "piorou" relevância.

## Ressalvas

- **n=3 rodadas por versão.** O teste t de Welch com 3+3 observações tem pouquíssimo grau de liberdade (df ≈ 4 em relevância). O sinal de relevância é forte o suficiente (intervalos separados, t≈7) para sobreviver a isso; um delta menor que o de faithfulness não seria.
- **Falhas pontuais do judge.** O `gpt-oss:120b-cloud` devolveu `IncompleteOutputException` em alguns casos, e a `n` de faithfulness por rodada ficou entre 8 e 11 (de 13). Os casos que falham mais cedo são sempre `calculation_charge_cost` e `memory_session_context`; em uma rodada foram 5 falhas (v3.0 rodada 2 e v3.1 rodada 2). answer_relevancy teve n=13 em todas as rodadas. As falhas do judge não são aleatórias (causadas pelo tamanho da resposta), então a faithfulness agregada se apoia em subconjuntos levemente diferentes por rodada — mais um motivo para tratar o delta de faithfulness como indecidível.
- **A base mudou.** As rodadas antigas (`iteration_*.json`) usavam 9 casos; estas usam os 15 casos de `evals/rag_cases.json` atual. Os números absolutos não são comparáveis entre gerações — a comparação válida é dentro destes 6 arquivos.
- Recusa: 2/2 correta em todas as 6 rodadas, ambas as versões (rubrica binária, não RAGAS).

## Como reproduzir

```bash
python evals/run_repetitions.py --versions 3.0 3.1 --repeticoes 3 \
  --rag-model gemma4:cloud --judge-model gpt-oss:120b-cloud \
  --nome-detalhe "evals/iter_{idx}_{ver}_run{rep}.json"
```

O `--nome-detalhe` gera exatamente os arquivos auditados aqui (`{ver}` = `v30`/`v31`, `{idx}` = ordem da versão, `{rep}` = repetição). O `run_repetitions.py` também emite o JSON agregado com média, desvio e a checagem de sobreposição de intervalos via `--saida`.
