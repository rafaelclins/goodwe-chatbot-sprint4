# Versionamento dos Prompts

Este documento registra a evolução dos system prompts do chatbot GoodWe EV Charging,
incluindo mudanças aplicadas, motivo e impacto medido.

## Medição

Todos os valores de tokens foram medidos com `tiktoken` (encoding `cl100k_base`).
Para reproduzir:

```bash
python scripts/measure_prompt.py --prompt prompts/system_prompt.txt
python scripts/measure_prompt.py --prompt prompts/system_prompt_v1.txt
python scripts/measure_prompt.py --prompt prompts/system_prompt_v2.txt
python scripts/measure_prompt.py --prompt prompts/system_prompt_v2.1.txt
python scripts/measure_prompt.py --prompt prompts/system_prompt_v2.2.txt
```

## Tabela de Versões

| Versão | Arquivo | Chars | Words | Tokens | Mudanças | Motivo | Ganho |
|--------|---------|------:|------:|-------:|----------|--------|-------|
| v0 (original) | `system_prompt.txt` | 1185 | 180 | 342 | Prompt inicial em texto plano. Regras básicas de escopo, fórmulas e estilo. | Estabelecer comportamento base do chatbot. | Baseline |
| v1 | `system_prompt_v1.txt` | 1185 | 180 | 342 | Idêntico ao v0. Primeira versão nominada para controle de versão. | Criar referência estável antes de reformulações. | Sem alteração (controle de versão) |
| v2 | `system_prompt_v2.txt` | 4251 | 454 | 1133 | Reescrita completa com XML tagging (`<identity>`, `<scope>`, `<calculations>`, `<guardrail_boundary>`, `<instruction_priority>`). Escopo explícito com lista de tópicos permitidos. Prioridade de instruções numerada. Boundary de guardrail integrado. Fronteira de produto e profissional definidas. | Prompt em texto plano era ambíguo para o modelo. XML tagging melhora a interpretação estruturada. Prioridade de instruções previne jailbreak. | Tokens: 342 → 1133 (+231%). Estruturação elimina ambiguidade e melhora aderência ao escopo em testes de guardrail. |
| v2.1 | `system_prompt_v2.1.txt` | 3081 | 347 | 835 | Simplificação do v2. Remoção de `<instruction_priority>` e `<guardrail_boundary>` (movidos para pipeline). Reforço em `<product_information>`: "Use somente dados disponíveis no contexto". `<cost_scope>` simplificado. `<missing_data>` mais direto. | Prompt v2 era excessivamente verboso para o que o pipeline já validava. Remoção de redundância entre prompt e guardrails. | Tokens: 1133 → 835 (-26%). Redução de 298 tokens sem perda de comportamento, pois o pipeline de guardrails já tratava as proteções removidas. |
| v2.2 | `system_prompt_v2.2.txt` | 5304 | 602 | 1396 | Context Engineering com XML tagging hierárquico: `<role>`, `<context>`, `<rules>`, `<guardrails>`, `<output_instructions>`. `<context>` contém `<data_source>`, `<constants>`, `<formulas>`. `<guardrails>` contém `<security>`, `<scope_enforcement>`, `<professional_boundary>`, `<financial_boundary>`, `<product_information>`. `<output_instructions>` define `<style>` e `<structure>` (passo a passo para cálculos e respostas conceituais). `<constants>` protege tarifa e potência como valores fixos. | Prompt v2.1 carecia de instruções de saída estruturada. Context Engineering com XML tagging organizado por responsabilidade facilita manutenção e permite que o modelo interprete seções de forma independente. | Tokens: 835 → 1396 (+67% vs v2.1). Ganho qualitativo: instruções de output estruturado, proteção explícita de constantes, separação clara entre contexto/dados e regras/guardrails. Manutenção facilitada pela organização por responsabilidade. |

## Resumo da Evolução

```
v0/v1 (texto plano, 342 tokens)
  └─→ v2 (XML tagging inicial, 1133 tokens)
        └─→ v2.1 (simplificação, 835 tokens)
              └─→ v2.2 (Context Engineering completo, 1396 tokens)
```

### Princípios aplicados no v2.2

- **Separation of Concerns**: cada tag XML tem uma responsabilidade única (`<role>`, `<context>`, `<rules>`, `<guardrails>`, `<output_instructions>`)
- **Data vs Rules**: dados (constant, fórmulas, fonte) separados de regras comportamentais
- **Guardrails como camada**: proteções organizadas por tipo (segurança, escopo, profissional, financeiro, produto)
- **Output Instructions**: instruções explícitas de estrutura de resposta para cálculos e conceitos
- **Constants as first-class**: tarifa e potência definidas como valores imutáveis no contexto
