"""
Avaliação do RAG por rubrica manual (fallback documentado).

===============================================================================
POR QUE ESTE ARQUIVO EXISTE — CONFLITO DE DEPENDÊNCIA INTRANSPONÍVEL
===============================================================================

Este script substitui a avaliação automatizada pelo RAGAS, que era a via
preferencial da Sprint 4. O RAGAS não roda neste ambiente por um conflito de
dependências que não pode ser resolvido sem quebrar o que já foi entregue:

    from ragas import evaluate
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

Causa raiz verificada: o RAGAS 0.4.3 importa o caminho legado
`langchain_community.chat_models.vertexai`, mas o `langchain-community`
instalado é o 0.4.2, no qual o módulo foi movido para
`langchain_community/llms/vertexai.py`. Como a falha acontece já no `import`
(módulo ausente, não um bug de runtime), **nenhuma métrica do RAGAS carrega** —
`evaluate`, `faithfulness`, `answer_relevancy` e `LangchainLLMWrapper` falham
todos juntos.

O conflito é intransponível nas duas direções:

  * Subir o `langchain-community` não resolve: a 0.4.2 ele já está descontinuado
    ("being sunset and no longer actively maintained") e não existe versão
    posterior que restaure o caminho antigo.
  * Descer o `langchain-community` para a série 0.2.x (que ainda expõe
    `chat_models/vertexai`) rebaixa `langchain-core` para < 1.0 e quebra
    `langchain-chroma` 1.1.0 e a chain LCEL da Sprint 3
    (`src/rag/chain.py`), que exige a API de runnables da linha 1.x.

Trocar o RAGAS por uma versão mais nova exigiria reescrever os prompts de
avaliação acoplados ao pacote, o que é esforço desproporcional ao ganho. Como a
dependência quebrada é do próprio pacote de avaliação — e não da nossa cadeia
RAG, que está funcional — o defeito fica isolado no harness de testes. Por isso
adotamos a rubrica de avaliação manual, alternativa permitida pela rubrica da
FIAP, sem alterar nenhuma dependência do projeto.

Vantagem adicional dessa abordagem: a avaliação passa a ser humana, o que é
relevante aqui porque as regras de *grounding* do prompt V1
(`src/rag/prompt_rag.py`) exigem que cada resposta cite arquivo e página —
um verificador automático de fidelidade não detectaria com precisão uma fonte
citada de forma incorreta ou uma recusa indevidamente evitada pelo modelo.

===============================================================================
A RUBRICA
===============================================================================

Duas dimensões, cada uma pontuada de 0.0 a 1.0 com passo de 0.1:

1. FAITHFULNESS (Fidelidade / groundedness) — 0.0 a 1.0
   Pergunta: a resposta usa APENAS fatos presentes no contexto fornecido?
     1.0 = 100% fiel: toda afirmação é rastreável ao contexto, e a citação de
          fonte (arquivo e página) conferida.
     0.0 = alucinação: inventa especificação, valor, corrente, potência, norma
          ou procedimento que não está no contexto.
   Âncoras intermediárias: 0.5 indica mistura de conteúdo fiel com um trecho
   sem lastro no contexto; 0.8 indica fidelidade total, porém com citação de
   fonte ausente, incorreta ou não conferível.

2. ANSWER RELEVANCY (Relevância) — 0.0 a 1.0
   Pergunta: a resposta atende diretamente ao que foi perguntado?
     1.0 = direta e completa: resolve a pergunta sem desvios.
     0.0 = foge do tema: responde outra coisa, ou se recusa a responder embora a
          resposta esteja disponível no contexto.
   Âncoras intermediárias: 0.5 indica resposta parcial ou com desvio relevante;
   0.8 indica resposta completa com um pequeno trecho supérfluo.

   NOTA SOBRE RECUSA: recusar de responder não é errado por si só — é o
   comportamento correto quando a informação não está no
   contexto (regra <refusal> do prompt V1). Nesse caso, a Relevância deve ser
   alta. O que penaliza Relevância é recusar sem necessidade, isto é, quando o
   contexto contém a resposta.

===============================================================================
COMO USAR
===============================================================================

Avaliando uma resposta real do chatbot:

    python evals/ragas_eval.py   # abre o mock de demonstração

    from evals.ragas_eval import run_manual_eval
    run_manual_eval(
        question="Qual a corrente nominal do carregador DC?",
        context=<texto recuperado por format_docs()>,
        generated_answer=<texto exibido pelo LLM>,
    )

O contexto deve ser o MESMO texto que foi entregue ao modelo, ou seja, a saída
de `format_docs(docs)` em `src/rag/retriever.py`, e não os chunks crus: as âncoras
de fidelidade são as fontes formatadas no bloco `<context>` do prompt.

Desde o prompt V3.0 cada trecho chega rotulado no formato `[arquivo.pdf, p. N]`,
que é o mesmo formato exigido para a citação na resposta. Ao avaliar, confira
que a citação presente na resposta aparece literalmente nesse rótulo — é a
forma mais direta de medir se a fonte citada existe de fato no contexto
recuperado.

Cada avaliação é registrada por acréscimo (append) em
`evals/evaluation_results.json`, preservando as iterações anteriores para
permitir a comparação de evolução entre prompts.
"""

import json
import os
from datetime import datetime

# O resultado é gravado ao lado deste script, garantindo que o arquivo fique
# sempre em `evals/evaluation_results.json` — independentemente do diretório de
# trabalho usado para executar a avaliação.
PASTA_EVALS = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_RESULTADOS = os.path.join(PASTA_EVALS, "evaluation_results.json")


def _ler_nota(rotulo: str) -> float:
    """
    Lê do terminal uma nota na escala da rubrica e valida o valor digitado.

    O avaliador pode encerrar a avaliação com "q", útil quando o caso avaliado
    não é conclusivo e não deve entrar na base de comparação entre iterações.

    Args:
        rotulo: Nome da dimensão exibido no prompt (ex.: "Faithfulness").

    Returns:
        float: Nota normalizada entre 0.0 e 1.0, com passo de 0.1.

    Raises:
        KeyboardInterrupt: Se o avaliador cancelar com Ctrl+C, propagado para
            encerrar o script sem gravar um registro incompleto.
    """
    while True:
        try:
            entrada = input(f"   {rotulo} [0.0 a 1.0, 'q' para sair]: ").strip().lower()

            if entrada in {"q", "quit", "sair"}:
                raise KeyboardInterrupt("avaliacao cancelada pelo avaliador")

            nota = float(entrada)

            if not 0.0 <= nota <= 1.0:
                print("   ✗ A nota deve estar entre 0.0 e 1.0. Tente novamente.")
                continue

            # Arredonda para uma casa decimal, mantendo a comparabilidade entre
            # avaliações feitas por avaliadores diferentes.
            return round(nota, 1)

        except ValueError:
            print("   ✗ Entrada inválida. Digite um número decimal, ex.: 0.8")


def _salvar_resultado(resultado: dict) -> None:
    """
    Acrescenta um resultado à lista de avaliações já existente no JSON.

    Lê o arquivo apenas se ele existir, inicializa a lista quando o arquivo
    ainda não foi criado (primeira iteração) e regrava o conjunto completo, de
    modo que a leitura e a escrita do JSON nunca sejam concorrentes.

    Args:
        resultado: Registro da iteração atual, pronto para ser serializado.
    """
    avaliacoes = []

    if os.path.exists(ARQUIVO_RESULTADOS):
        # Um arquivo vazio (0 bytes) produz None; um objeto único também é
        # tolerado, para não perder dados já registrados à mão.
        conteudo = None
        with open(ARQUIVO_RESULTADOS, "r", encoding="utf-8") as arquivo:
            try:
                conteudo = json.load(arquivo)
            except json.JSONDecodeError:
                # A avaliação corrente não pode ser perdida por causa de um
                # arquivo quebrado, então segue em uma lista nova. O avaliador
                # é avisado porque as iterações anteriores serão sobrescritas.
                print(
                    "\n⚠️  O arquivo de resultados anterior estava corrompido e "
                    "será substituído.\n"
                    "    Mova-o de lado se quiser preservar o histórico antigo."
                )

        if isinstance(conteudo, list):
            avaliacoes = conteudo
        elif isinstance(conteudo, dict):
            avaliacoes = [conteudo]

    avaliacoes.append(resultado)

    os.makedirs(PASTA_EVALS, exist_ok=True)

    with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as arquivo:
        json.dump(avaliacoes, arquivo, ensure_ascii=False, indent=2)

    print(f"\n✓ Iteração {len(avaliacoes)} registrada em {ARQUIVO_RESULTADOS}")


def run_manual_eval(question: str, context: str, generated_answer: str) -> None:
    """
    Avalia um turno do chatbot RAG com a rubrica manual de fallback.

    Exibe no terminal a pergunta, o contexto recuperado e a resposta do LLM,
    e então solicita ao avaliador humano as notas de Faithfulness e Answer
    Relevancy. O registro é acrescido ao final em
    `evals/evaluation_results.json`.

    A avaliação é encerrada sem gravar nada caso o avaliador cancele com "q",
    Ctrl+C ou EOF (Ctrl+D).

    Args:
        question: Pergunta enviada pelo usuário ao chatbot.
        context: Contexto recuperado da base vetorial, no mesmo formato
            entregue ao modelo (saída de `format_docs`).
        generated_answer: Resposta exibida pelo LLM ao usuário.
    """
    print("=" * 78)
    print("  AVALIAÇÃO MANUAL DO RAG — SPRINT 4 (FIAP + GoodWe)")
    print("  Fallback documentado: RAGAS inutilizável por conflito de dependência")
    print("=" * 78)

    print("\n[1] PERGUNTA DO USUÁRIO")
    print("-" * 78)
    print(question)

    print("\n[2] CONTEXTO RECUPERADO (base de conhecimento)")
    print("-" * 78)
    print(context)

    print("\n[3] RESPOSTA DO LLM")
    print("-" * 78)
    print(generated_answer)

    print("\n[4] AVALIAÇÃO — PREENCHA AS DUAS NOTAS")
    print("-" * 78)
    print("  Faithfulness: a resposta usa APENAS fatos presentes no contexto?")
    print("                 1.0 = 100% fiel (com fonte conferida)")
    print("                 0.0 = alucinação (fato fora do contexto)")
    print("  Answer Relevancy: a resposta atende diretamente ao que foi perguntado?")
    print("                 1.0 = direta e completa")
    print("                 0.0 = foge do tema")
    print()

    try:
        faithfulness = _ler_nota("Faithfulness    ")
        answer_relevancy = _ler_nota("Answer Relevancy")
    except (KeyboardInterrupt, EOFError):
        print("\n\n✗ Avaliação cancelada. Nada foi gravado no arquivo de resultados.")
        return

    _salvar_resultado(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metodo": "rubrica_manual_fallback",
            "question": question,
            "context": context,
            "generated_answer": generated_answer,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }
    )


if __name__ == "__main__":
    # Caso de demonstração para o avaliador exercitar o fluxo completo do
    # script e registrar a primeira iteração da linha de base do prompt V1.
    # O contexto abaixo é a ÚNICA página de data/knowledge_base/teste_manual.pdf,
    # transcrita como o retriever a entregaria ao modelo. O mock cita apenas
    # o que existe de fato nesse arquivo, porque um exemplo com fonte inventada
    # ensinaria o avaliador a aceitar alucinação — justamente o defeito que a
    # dimensão Faithfulness existe para penalizar.
    MOCK_QUESTION = (
        "Qual a corrente nominal, a potência e o protocolo de comunicação "
        "do carregador DC instalado no condomínio?"
    )

    MOCK_CONTEXT = (
        "Fonte: data/knowledge_base/teste_manual.pdf (p. 0)\n"
        "Carregador DC 22 kW - Especificacoes tecnicas\n"
        "Tensao de entrada: 380 V trifasico\n"
        "Corrente nominal: 32 A\n"
        "Eficiencia maxima: 94%\n"
        "Protocolo de comunicacao: Modbus TCP"
    )

    MOCK_GENERATED_ANSWER = (
        "O carregador DC instalado no condomínio tem potência de **22 kW**, "
        "**corrente nominal de 32 A** e opera com tensão de entrada de 380 V "
        "trifásico, com eficiência máxima de 94%. O protocolo de comunicação "
        "utilizado é o **Modbus TCP**.\n\n"
        "Fonte: teste_manual.pdf (p. 0)"
    )

    run_manual_eval(
        question=MOCK_QUESTION,
        context=MOCK_CONTEXT,
        generated_answer=MOCK_GENERATED_ANSWER,
    )
