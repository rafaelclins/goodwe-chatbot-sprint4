# GoodWe Chatbot — Sprint 3

Chatbot técnico desenvolvido para auxiliar em questões relacionadas à recarga de veículos elétricos em condomínios, utilizando IA generativa com contexto especializado da GoodWe.

---

## Integrantes

- Rafael Lins — RM: 570588
- Rafael Gandolfi — RM: 569036
- Carlos Tavares — RM: 572949
- Cauã Paes — RM: 569906
- Guilherme Miranda — RM: 573107
- João Pedro Soler — RM: 569725

---

## Objetivo do Projeto

O objetivo deste projeto é desenvolver um chatbot contextualizado para responder perguntas relacionadas ao ecossistema de recarga de veículos elétricos em condomínios.

O sistema foi desenvolvido como parte do Challenge FIAP + GoodWe 2026, evoluído da Sprint 2 (chatbot funcional) para a Sprint 3 (pipeline com memória, guardrails, structured output e avaliação automatizada).

O chatbot utiliza:
- histórico de mensagens por sessão
- system prompt especializado
- integração com modelo de IA via Ollama Cloud (`gemma4:cloud`)
- controle de contexto técnico
- limite de histórico por tokens
- guardrails de escopo e prompt injection
- saída estruturada validada com Pydantic v2

O foco principal do projeto é garantir respostas coerentes, contextualizadas e alinhadas ao domínio de infraestrutura de recarga elétrica.

---

## Funcionalidades Implementadas

Sprint 2:
- Chatbot funcional em linha de comando
- Histórico de conversa
- Contexto persistente via system prompt
- Integração com API de IA generativa
- Leitura segura de API Key via `.env`

Sprint 3:
- Arquitetura baseada em LCEL (`ChatPromptTemplate | ChatOllama | StrOutputParser`)
- Memória de conversa por `session_id` com `RunnableWithMessageHistory`
- Controle de histórico por tokens com `trim_messages` (limite padrão de 1000 tokens)
- Guardrails de escopo temático e detecção de prompt injection
- Saída estruturada validada com Pydantic v2 (`ConsultaRecarga`)
- Prompts versionados (`v1`, `v2`, `v2.1`, `v2.2`)
- Medição de tokens de prompt com `tiktoken`
- Avaliação automatizada com 16 casos de teste e fake model
- Bônus: demonstração multi-provider (múltiplos modelos x versões de prompt, modo paralelo/alternado, com consolidação da melhor resposta)

---

## Tecnologias Utilizadas

- Python 3.12
- LangChain / langchain-core / langchain-ollama
- Ollama Cloud
- Modelo `gemma4:cloud`
- Pydantic v2
- tiktoken
- python-dotenv

---

## Estrutura do Projeto

```text
goodwe-chatbot-sprint3/
│
├── chatbot.py                  # Chatbot CLI (Sprint 2)
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── prompts/                    # Prompts versionados
│   ├── system_prompt.txt       # Prompt original (Sprint 2)
│   ├── system_prompt_v1.txt
│   ├── system_prompt_v2.txt
│   ├── system_prompt_v2.1.txt
│   └── system_prompt_v2.2.txt  # Prompt ativo na arquitetura
│
├── src/
│   ├── chain/
│   │   ├── builder.py          # LCEL: prompt | modelo | parser
│   │   ├── memoria.py          # RunnableWithMessageHistory + trim_messages
│   │   └── multi_provider.py   # Demo bônus: compara modelos x versões de prompt
│   ├── guardrails/
│   │   ├── moderation.py       # Detecção de prompt injection
│   │   ├── scope_validator.py  # Validação de escopo temático
│   │   └── pipeline.py         # Orquestração da entrada do usuário
│   └── schemas/
│       └── consulta_recarga.py # Schema Pydantic v2 de saída
│
├── scripts/
│   └── measure_prompt.py       # Medição de tokens com tiktoken
│
├── evals/
│   ├── run_sprint3.py          # Executor de avaliação (fake model)
│   └── sprint3_cases.json      # 16 casos de avaliação
│
└── tests/                      # Suíte de testes automatizados
```

---

## Arquitetura (Sprint 3)

A cadeia principal é construída com LangChain Expression Language (LCEL):

```python
ChatPromptTemplate | ChatOllama | StrOutputParser
```

Em `src/chain/builder.py`, o prompt é montado com `ChatPromptTemplate` usando o system prompt versionado (`prompts/system_prompt_v2.2.txt`), um `MessagesPlaceholder` para o histórico e a mensagem humana com as instruções de formato. O modelo é o `gemma4:cloud` acessado via Ollama Cloud.

### Memória

Em `src/chain/memoria.py`, o histórico é mantido por sessão com `RunnableWithMessageHistory` e `InMemoryChatMessageHistory`, indexado por `session_id`. Sessões diferentes não compartilham histórico.

O controle de tamanho é feito com `trim_messages`:

- limite padrão de 1000 tokens (`max_history_tokens=1000`);
- contagem aproximada de tokens;
- mantém as mensagens mais recentes (`strategy="last"`);
- não mantém mensagens parcialmente (`allow_partial=False`);
- o histórico recortado começa em uma mensagem humana (`start_on="human"`).

O recorte é aplicado antes e depois de cada chamada à cadeia, garantindo que o histórico armazenado e o histórico fornecido ao modelo respeitem o limite.

### Guardrails

Em `src/guardrails/`, a entrada passa por duas validações antes de chegar ao modelo:

- `moderation.py`: detecta tentativas de prompt injection (jailbreak, revelação de instruções internas, alteração de escopo e outras);
- `scope_validator.py`: valida o escopo temático (recarga, energia, infraestrutura elétrica, rateio de custos) e bloqueia aconselhamento jurídico, financeiro, decisões de segurança elétrica e especificações exatas de produto sem base.

Entradas bloqueadas recebem resposta local definida no `pipeline.py` e não chegam ao modelo nem são gravadas no histórico.

### Structured Output

A resposta é validada como instância do schema Pydantic v2 `ConsultaRecarga` (`src/schemas/consulta_recarga.py`) com `PydanticOutputParser`. O parser falha se a resposta do modelo não for um JSON válido no formato esperado.

### Prompts Versionados

Os prompts evoluíram ao longo dos sprints:

- `system_prompt.txt`: prompt original (Sprint 2);
- `system_prompt_v1.txt`: primeira versão estruturada;
- `system_prompt_v2.txt`: escopo explícito, prioridade de instruções e fronteiras de guardrail;
- `system_prompt_v2.1.txt`: definição ampliada de escopo e fronteira de fontes de dados;
- `system_prompt_v2.2.txt`: versão ativa, com fonte de dados restrita ao contexto e instruções preservadas de fórmulas e constantes.

### Medição de Tokens

O script `scripts/measure_prompt.py` mede o tamanho dos prompts com `tiktoken` (encoding `cl100k_base`), retornando caracteres, palavras e tokens de qualquer prompt da pasta `prompts/`.

```bash
python scripts/measure_prompt.py
python scripts/measure_prompt.py --prompt prompts/system_prompt_v2.2.txt
```

---

## Instalação

Clone o repositório:

```bash
git clone LINK_DO_REPOSITORIO
```

Acesse a pasta do projeto:

```bash
cd goodwe-chatbot-sprint2
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Configuração

O modelo padrão `gemma4:cloud` é servido pela Ollama Cloud. Crie o arquivo `.env` a partir do exemplo e informe a chave de API:

```bash
cp .env.example .env
```

```text
OLLAMA_API_KEY = minha_chave_aqui
```

Confirme que o modelo está disponível na sua conta:

```bash
ollama pull gemma4:cloud
```

---

## Execução

Execute o chatbot CLI com:

```bash
python chatbot.py
```

Para executar a suíte de testes:

```bash
python -m unittest discover -s tests -v
```

Para executar a avaliação com fake model (sem chamadas reais ao modelo):

```bash
python evals/run_sprint3.py
```

Para executar o benchmark comparativo de modelos via Ollama Cloud (chamadas reais):

```bash
python scripts/benchmark_modelos.py
```

O benchmark executa 5 perguntas de `evals/sprint3_cases.json` em dois modelos
(`gemma4:cloud` e `gpt-oss:20b`) com hiperparâmetros fixos
(temperature=0.2, top_p=0.9, max_tokens=500) e reporta latência média, tokens
de entrada/saída (tiktoken), acurácia no JSON do schema `ConsultaRecarga` e
coerência no domínio. Resultados salvos em `evals/benchmark_modelos_results.json`.

### Demonstração Multi-Provider (Bônus)

A funcionalidade opcional `src/chain/multi_provider.py` consulta o mesmo domínio
com mais de um modelo **e** mais de uma versão de system prompt (ex.: `gemma4:cloud`
e `gpt-oss:20b` com `v2.1` e `v2.2`), em modo **paralelo** ou **alternado**. Para
cada combinação modelo x prompt, a resposta recebe um score 0-10 (validação no
schema Pydantic `ConsultaRecarga` + fluência + vocabulário do domínio) e a
demonstração devolve a melhor resposta consolidada junto da comparação completa.

> Se `OLLAMA_API_KEY` estiver no `.env`, as chamadas usam o Ollama Cloud; sem a
> chave, o script tenta o Ollama local. Qualquer modelo disponível pode ser
> informado em `--models` (ex.: `gemma4:cloud qwen3:8b`).

```bash
# Execução padrão (paralelo, 2 modelos x 2 prompts)
python -m src.chain.multi_provider

# Personalizado: modelos, versões de prompt, modo e salvar resultado em JSON
python -m src.chain.multi_provider \
  --question "Quanto custa carregar 40 kWh de 20% a 80%?" \
  --models gemma4:cloud gpt-oss:20b \
  --prompts v2.1 v2.2 \
  --mode paralelo \
  --output evals/multi_provider_results.json
```

---

## Resultados da Avaliação (Sprint 3)

A avaliação roda 16 casos contra o pipeline completo usando um fake model, sem fazer chamadas reais ao Ollama. Resultados:

- 16 casos executados;
- 7 bloqueados pelos guardrails (jailbreak, revelação de instruções, fora de escopo, especificação de produto sem base, segurança elétrica, aconselhamento jurídico e financeiro);
- 9 casos permitidos com chamada ao modelo;
- 9/9 structured outputs válidos como instância de `ConsultaRecarga`;
- 0 erros;
- memória validada em conversas de 3+ turnos: o histórico dos turnos anteriores é fornecido ao modelo no turno seguinte;
- limite de histórico de 1000 tokens validado: históricos acima do limite são reduzidos por `trim_messages` sem mensagens parciais e iniciando em mensagem humana;
- isolamento entre sessões validado: `session_a` e `session_b` não compartilham histórico.

### Limitação conhecida

A avaliação não garante eliminação de todas as alucinações técnicas. Quando o contexto fornecido não contém a base técnica necessária, o modelo ainda pode introduzir conhecimento técnico externo em algumas respostas.

---

## Exemplo de Uso

### Pergunta simples

Usuário:
> O que é kWh?

Chatbot:
> O kWh é a unidade de medida da energia elétrica utilizada para calcular o consumo de energia em sistemas elétricos e processos de recarga.

---

### Pergunta técnica

Usuário:
> Posso instalar 12 carregadores no meu condomínio?

Chatbot:
> Para determinar isso, é necessário validar a capacidade elétrica disponível da infraestrutura do condomínio, além da potência total demandada pelos carregadores. Recomenda-se análise com engenheiro eletricista.

---

## Testes Realizados

| Pergunta | Avaliação |
|---|---|
| O que é kWh? | Adequada |
| Quanto custa carregar uma bateria de 60 kWh de 20% a 80%? | Adequada |
| Quanto tempo leva para carregar uma bateria de 40 kWh? | Adequada |
| Qual é o melhor carro elétrico do mercado? | Adequada |
| Posso instalar 12 carregadores no meu condomínio? | Adequada |

---

## Limitações

O chatbot não substitui validação profissional de infraestrutura elétrica.

As respostas são baseadas:
- no contexto definido pelo system prompt
- no histórico de mensagens
- nas informações fornecidas pelo usuário

---

## Segurança

- O projeto utiliza `.gitignore` para proteger arquivos sensíveis
- A chave `OLLAMA_API_KEY` é lida via `.env` e nunca deve ser versionada
- As chamadas ao modelo `gemma4:cloud` exigem autenticação na Ollama Cloud

---

## Demonstração

Vídeo demonstrativo do projeto:
> LINK_DO_VIDEO(https://youtube.com/shorts/Jqk7_mAXsr0)

---

## Repositório

GitHub do projeto:
> LINK_DO_PROJETO(https://github.com/rafaelclins/goodwe-chatbot-sprint2)
