# GoodWe Chatbot — Sprint 2

Chatbot técnico desenvolvido para auxiliar em questões relacionadas à recarga de veículos elétricos em condomínios, utilizando IA generativa com contexto especializado da GoodWe.

---

## Integrantes

- Rafael Lins — RM: 570588
- Rafael Gandolfi — RM: 569036
- Carlos Tavares — RM: 592949
- Cauã Paes — RM: 569906
- Guilherme Miranda — RM: 573107
- João Pedro Soler — RM: 569725

---

## Objetivo do Projeto

O objetivo deste projeto é desenvolver um chatbot contextualizado para responder perguntas relacionadas ao ecossistema de recarga de veículos elétricos em condomínios.

O sistema foi desenvolvido como parte da Sprint 2 do Challenge FIAP + GoodWe 2026.

O chatbot utiliza:
- histórico de mensagens
- system prompt especializado
- integração com modelo de IA
- controle de contexto técnico

O foco principal do projeto é garantir respostas coerentes, contextualizadas e alinhadas ao domínio de infraestrutura de recarga elétrica.

---

## Funcionalidades Implementadas

- Histórico de conversa
- Contexto persistente via system prompt
- Controle de escopo temático
- Respostas técnicas contextualizadas
- Proteção básica contra perguntas fora do contexto
- Integração com API de IA generativa
- Leitura segura de API Key via `.env`

---

## Tecnologias Utilizadas

- Python 3.13
- OpenRouter API
- Meta Llama 3 8B Instruct
- python-dotenv
- PyCharm

---

## Estrutura do Projeto

```text
goodwe-chatbot-sprint2/
│
├── chatbot.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
└── prompts/
    └── system_prompt.txt
```

---

## Funcionamento do Sistema

O chatbot funciona através do envio de mensagens para um modelo de linguagem hospedado via OpenRouter API.

O sistema:
1. Carrega o system prompt especializado
2. Recebe a mensagem do usuário
3. Mantém o histórico da conversa
4. Envia o contexto completo para o modelo
5. Retorna a resposta contextualizada

O histórico de mensagens permite que o chatbot mantenha coerência durante conversas contínuas.

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

## Configuração da API

Crie um arquivo `.env` na raiz do projeto:

```env
OPENROUTER_API_KEY=sua_chave_aqui
```

---

## Execução

Execute o chatbot com:

```bash
python chatbot.py
```

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

- A API Key é armazenada via variável de ambiente
- O arquivo `.env` não é enviado ao GitHub
- O projeto utiliza `.gitignore` para proteger credenciais

---

## Demonstração

Vídeo demonstrativo do projeto:
> LINK_DO_VIDEO

---

## Repositório

GitHub do projeto:
> LINK_DO_REPOSITORIO
