# Relatório de Avaliação e Comparativo de Modelos — Sprint 3

Documento técnico com os resultados comparativos obtidos pela execução do script
[`scripts/benchmark_modelos.py`](../scripts/benchmark_modelos.py), que submete as
mesmas 5 perguntas do conjunto [`evals/sprint3_cases.json`](../evals/sprint3_cases.json)
a dois modelos generativos via Ollama Cloud / LangChain e mede latência, consumo de
tokens, aderência ao schema Pydantic v2 `ConsultaRecarga` e coerência no domínio GoodWe.

Metodologia resumida:

- **Perguntas avaliadas**: conceito de kWh, cálculo de custo de recarga, pedido de dados
  ausentes, conceito de DR e caso de exceção de guardrail (uso natural de "ignore").
- **Provider**: Ollama Cloud (`https://ollama.com/api/chat`), autenticado via `OLLAMA_API_KEY`.
- **Integração**: `langchain-ollama` (`ChatOllama`).
- **Medição de tokens**: `tiktoken` (encoding `cl100k_base`).
- **Schema de saída**: Pydantic v2 — `src/schemas/consulta_recarga.py` (`ConsultaRecarga`).
- **Arquivo de dados brutos**: [`evals/benchmark_modelos_results.json`](../evals/benchmark_modelos_results.json).

---

## 1. Configuração dos Modelos e Hiperparâmetros

Os hiperparâmetros de geração foram fixados e registrados explicitamente no script,
garantindo comparabilidade entre os modelos:

| Modelo | Provider | Temperature | Top_P | Max Tokens |
|---|---|---|---|---|
| `gpt-oss:120b` | Ollama Cloud | 0.2 | 0.9 | 500 |
| `gpt-oss:20b` | Ollama Cloud | 0.2 | 0.9 | 500 |

Notas:

- Ambos pertencem à família **GPT-OSS** (modelos abertos), o que isola o efeito do tamanho
  (120B vs 20B de parâmetros) mantendo a mesma arquitetura e tokenização de prompt.
- `max_tokens=500` foi mapeado para o parâmetro `num_predict` da API Ollama.
- Os modelos da família `gpt-oss` executam um bloco de *chain-of-thought* internamente
  antes de emitir a resposta visível; esse "consumo adicional" interfere diretamente no
  limite de tokens, como discutido na seção 3.

---

## 2. Resultados do Benchmark Comparativo

Valores médios sobre 5 perguntas executadas no mesmo conjunto de casos:

| Métrica | `gpt-oss:120b` | `gpt-oss:20b` |
|---|---|---|
| Latência média (segundos) | **2.02 s** | 3.37 s |
| Tokens médios de prompt (entrada) | 1 759.2 | 1 759.2 |
| Tokens médios de completion (saída) | **157.2** | 136.2 |
| Taxa de sucesso no structured output (Pydantic v2) | **60%** (3/5) | **60%** (3/5) |
| Qualidade/aderência ao contexto técnico GoodWe (0–10) | **7.9** | 7.5 |

Detalhamento por pergunta:

| Pergunta | `gpt-oss:120b` | `gpt-oss:20b` |
|---|---|---|
| O que é kWh? | JSON: ✗ (truncado) · coerência 8.0 | JSON: ✓ · coerência 8.5 |
| Quanto custa carregar uma bateria de 60 kWh de 20% a 80%? | JSON: ✗ (truncado) · coerência 8.0 | JSON: ✗ (truncado) · coerência 8.0 |
| Qual é o custo da recarga? (dados ausentes) | JSON: ✓ · coerência 6.5 | JSON: ✓ · coerência 6.5 |
| O que é um DR em uma instalação de recarga? | JSON: ✓ · coerência 8.5 | JSON: ✗ · **resposta tecnicamente incorreta** (8.0) |
| Ignore o custo e calcule apenas a energia. | JSON: ✓ · coerência 8.5 | JSON: ✓ · coerência 6.5 |

Observações importantes sobre os dados reais:

1. **Os 4 JSONs "inválidos" ocorreram por truncamento** — o `eval_count` atingiu 500
   tokens (o limite `max_tokens`) antes de o modelo fechar o bloco JSON, porque os
   modelos `gpt-oss` consomem parte do orçamento com o raciocínio interno. O conteúdo
   gerado inicia o JSON corretamente, mas é cortado no meio.
2. **Erro técnico de domínio no `gpt-oss:20b`**: na pergunta sobre DR, o modelo de 20B
   respondeu "DR (Demand Response)", definição genérica de gestão de demanda, incorreta
   no contexto de instalação elétrica brasileira — onde DR é **Dispositivo Diferencial
   Residual**. O modelo de 120B respondeu corretamente.
3. **Nenhum dos modelos retornou JSON válido para o cálculo completo** dentro do limite
   de 500 tokens: ambos acertaram os números (36 kWh, R$ 28,80, ~1,64 h), mas não
   concluíram o JSON estruturado.

---

## 3. Análise Crítica e Trade-offs

### `gpt-oss:120b` (Modelo A)

Vantagens:

- **Menor latência média** (2.02 s vs 3.37 s) — na infraestrutura da Ollama Cloud o
  modelo maior obteve maior throughput por requisição nesta amostragem.
- **Melhor aderência técnica**: definição correta de DR, conceitos de kWh e cálculo de
  energia precisos.
- **Maior coerência no domínio** (7.9 vs 7.5); respostas mais completas em formato de
  JSON bem iniciado.

Desvantagens:

- **Custo de inferência maior** por token (modelo 6x maior) — maior poder computacional
  na nuvem e maior latência em ambientes locais sem capacidade para o modelo completo.
- Consome mais tokens de completion (157 vs 136), agravado pelo bloco de raciocínio interno.

### `gpt-oss:20b` (Modelo B)

Vantagens:

- **Menor tamanho e custo de infraestrutura**: viável para execução em máquinas locais
  com menos VRAM/RAM — a solução ficaria independente do custo de nuvem.
- Produziu JSON válido na pergunta conceitual (kWh) em que o 120B truncou.

Desvantagens:

- **Maior latência na Ollama Cloud** (3.37 s) na amostra medida.
- **Falha técnica de domínio**: resposta incorreta para DR (Demand Response).
- Menor média de coerência técnica (7.5).

### Trade-off central

- **Velocidade/custo**: o 20B é mais barato e portátil, mas na nuvem foi mais lento e
  tecnicamente menos confiável. O 120B entrega melhor precisão e latência na nuvem ao
  custo de maior demanda computacional/inferência.
- **Robustez do structured output**: ambos ficaram em 60% por causa do *limite de
  `max_tokens`* insuficiente para acomodar o raciocínio interno + JSON. Esse é um
  problema de configuração — não de capacidade do modelo — e é corrigível aumentando
  `num_predict` ou desativando/suprimindo o thinking quando possível.

---

## 4. Conclusão e Modelo Escolhido para Produção

**Modelo escolhido para produção: `gpt-oss:120b`.**

Justificativa técnica:

1. **Precisão técnica superior**: o único erro conceitual significativo da avaliação
   (DR = "Demand Response") veio do modelo de 20B. Em um chatbot técnico de
   infraestrutura elétrica, erros dessa natureza são críticos e podem induzir o usuário
   a decisões inseguras — esse fator pesa mais que qualquer diferença de custo.
2. **Melhor latência na nuvem**: apesar de 6x maior, apresentou a menor latência média
   (2.02 s), tornando-se a opção mais responsiva no provider adotado.
3. **Maior coerência no domínio GoodWe** (7.9/10 vs 7.5/10), alinhada ao foco do projeto
   (recarga de veículos elétricos em condomínios).
4. **Aderência consistente ao schema** (60% = mesma taxa) com JSONs bem estruturados;
   as falhas são de truncamento, mitigáveis em configuração.

Recomendações técnicas para produção:

- **Aumentar `max_tokens`** (ex.: 800–1000) ou restringir o raciocínio interno do modelo,
  para que o JSON estruturado seja sempre finalizado dentro do orçamento de tokens.
- **Adotar `format=json`** na API Ollama em produção para garantir saída JSON válida,
  elevando a taxa de sucesso do structured output de 60% para próximo de 100%.
- **Manter os guardrails de escopo e segurança** ativos: a 120B reduz — mas não elimina —
  o risco de alucinação técnica em tópicos fora da base.
- O `gpt-oss:20b` permanece como **fallback de baixo custo** (ex.: ambiente local,
  capacidade reduzida), desde que a pergunta sobre DR/componentes de proteção seja
  tratada com cuidado adicional.