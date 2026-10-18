# Comparação de `chunk_size` (Aula 07)

## Metodologia

- Base: `data/knowledge_base/` (PDFs do domínio GoodWe)
- Splitter: `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=chunk_size // 8)`
- Collections: `goodwe_kb_512` e `goodwe_kb_1024` no `data/chroma_db/`
- Casos: `evals/rag_cases.json` — apenas blocos `answerable` (13)
- Execuções por pipeline: 3
- Modelo RAG: `gemma4:cloud` (`temperature=0`)
- Judge RAGAS: `gpt-oss:120b-cloud` (`temperature=0`)


## Resultados (média ± desvio padrão entre execuções)

| chunk_size | overlap | faithfulness (média ± dp) | answer_relevancy (média ± dp) |
|------------|---------|---------------------------|-------------------------------|
| 512 | 64 | 0.8202 ±0.0396 (n=3) | 0.5597 ±0.0413 (n=3) |
| 1024 | 128 | 0.7593 ±0.0161 (n=3) | 0.5602 ±0.0172 (n=3) |

## Conclusão

- **faithfulness:** 1024 = 0.7593 vs 512 = 0.8202 → venceu 512 ou empate

- **answer_relevancy:** 1024 = 0.5602 vs 512 = 0.5597 → venceu 1024


**Não houve vencedor consistente** — o `chunk_size = 1024` não superou o 512 em ambas as métricas.

## Hipótese

Os documentos do domínio (manuais técnicos de carregadores e inversores) são curto-texto e densos: especificação, tabela tarifária e procedimento de segurança costumam ocupar poucos parágrafos. Um ``chunk_size`` maior (1024) tende a manter a especificação completa e sua unidade de medida em um único vetor, reduzindo o corte que separa o valor da condição de medição — o que aumenta a fundamentação (faithfulness) e a pertinência da resposta (answer_relevancy). O custo é um vetor mais "ruidoso" quando o chunk cruza seções distintas do manual.


---
Gerado em: 2026-10-17T15:36:30
