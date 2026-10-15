"""
Guardrails do pipeline RAG — adaptacao dos guardrails da Sprint 3 para a Sprint 4.

Este modulo NAO contem regras proprias: `moderate()` (deteccao de prompt
injection) e `validate_scope()` (escopo tematico) sao importados de
`src.guardrails`, o pacote da Sprint 3, e nao copiados. As duas dependencias sao
a mesma logica que roda antes do LLM na Sprint 3; repetir os padroes aqui criaria
duas listas de regex que divergem com o tempo, e a divergencia seria silenciosa:
o filtro atualizado em um lado continuaria barrando (ou liberando) no outro.

A unica regra nova e o gate de "fora do contexto", descrito em
`avaliar_contexto()`.

Como `src.guardrails` vive no projeto irmao `goodwe-chatbot-sprint3`, e os dois
`src` sao namespace packages (PEP 420) que o interpretador funde quando as duas
raizes estao no `sys.path`, este modulo registra a raiz da Sprint 3 antes do
import — a mesma solucao de `app/app.py`, e com a mesma ressalva: nenhum dos
dois `src` pode receber um `src/__init__.py`, senao a fusao deixa de funcionar.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
RAIZ_SPRINT3 = RAIZ_PROJETO / "goodwe-chatbot-sprint3"

if str(RAIZ_SPRINT3) not in sys.path:
    sys.path.insert(0, str(RAIZ_SPRINT3))

from src.guardrails.moderation import moderate  # noqa: E402
from src.guardrails.pipeline import _blocked_response  # noqa: E402
from src.guardrails.scope_validator import GuardrailResult, validate_scope  # noqa: E402

# Limiar de similaridade para considerar que a base "respondeu" a pergunta.
#
# ATENCAO: este valor nao esta calibrado, e o motivo esta documentado em
# `avaliar_contexto()`. Com os embeddings atuais do `nomic-embed-text`, as
# similaridades entre documentos desta base ficam na faixa 0.69-0.74, e um
# limiar de 0.30 fica abaixo de toda a faixa observada: o gate, na pratica, so
# dispara quando a busca nao devolve NADA. Isso e preferivel a um limiar
# calibrado as cegas, que rejeitaria perguntas legitimas com base em ruido.
# Recalibrar exige uma base maior e casos com relevancia anotada.
LIMIAR_SIMILARIDADE = 0.30

# Resposta do gate de contexto. Fala em "base de conhecimento" e nao em "manuais"
# para acompanhar o prompt V3.0, que usa a mesma expressao na regra de recusa:
# quando a recusa vem do codigo e a do modelo usam o mesmo vocabulario, o usuario
# nao percebe que existem dois mecanismos diferentes recusando.
RESPOSTA_FORA_DE_CONTEXTO = (
    "Não encontrei essa informação na base de conhecimento.\n\n"
    "A busca não retornou trechos suficientemente relacionados à sua pergunta. "
    "Tente reformulá-la com termos técnicos mais específicos, por exemplo "
    "corrente, disjuntor, tarifa, rateio ou prazo."
)


@dataclass(frozen=True)
class Bloqueio:
    """Decisao de um guardrail: o que passou e, se passou, por que nao.

    Attributes:
        resposta: Texto a devolver ao usuario, ja no tom amigavel da Sprint 3.
        category: Categoria do bloqueio, que identifica qual regra disparou e
            permite comparar com `expected_category` em `evals/rag_cases.json`.
        origin: Etapa que decidiu ("moderation", "scope" ou "contexto"). Serve
            para diagnostico: um bloqueio em "contexto" com many consultas
            aceitas indica problema de recuperacao, nao de guardrail.
        detalhe: Similaridade do melhor trecho, preenchido apenas no gate de
            contexto, para permitir calibrar o limiar com dado e nao com palpite.
    """

    resposta: str
    category: str
    origin: str
    detalhe: str = ""


def avaliar_pergunta(pergunta: str) -> Bloqueio | None:
    """Aplica moderation e scope_validator antes de qualquer busca.

    A ordem importa. `moderate()` roda primeiro porque e o filtro mais barato e o
    mais urgente: uma tentativa de injecao que requeste "ignore todas as regras"
    seria classificada como fora de escopo pelo `validate_scope()`, e o usuario
    receberia a mensagem de escopo, que nao explica o que aconteceu de fato.

    Args:
        pergunta: Texto digitado pelo usuario.

    Returns:
        Bloqueio | None: `None` quando a pergunta pode seguir para a busca; caso
        contrario, o bloqueio com a resposta local e a categoria que o disparou.
    """
    resultado_moderacao = moderate(pergunta)
    if not resultado_moderacao.allowed:
        return _para_bloqueio(resultado_moderacao, "moderation")

    resultado_escopo = validate_scope(pergunta)
    if not resultado_escopo.allowed:
        return _para_bloqueio(resultado_escopo, "scope")

    return None


def avaliar_contexto(
    trechos: list[tuple[Document, float]],
) -> Bloqueio | None:
    """Decide se a base vetorial devolveu algo relevante o bastante.

    Este e o unico guardrail que olha para a saida da busca em vez da pergunta.
    Ele existe porque os outros dois nao conseguem cobrir o caso mais comum do
    RAG: a pergunta e legitima e esta no escopo, mas simplesmente nao ha nada no
    material que a responda. Sem este gate, o modelo receberia um contexto vazio
    ou irrelevante e teria duas opcoes ruins — recusar sem saber que a base foi
    consultada, ou preencher a lacuna com conhecimento externo, que e exatamente
    a alucinacao que o grounding existe para evitar. Bloqueando antes do LLM, a
    recusa sai correta e sem custo de inferencia.

    COMO O LIMIAR DEVE SER LIDO
    ---------------------------
    A similaridade usada aqui e de cosseno, em [-1, 1] (ver
    `similarity_from_distance`). Ela NAO e a pontuacao de relevancia que o
    LangChain normaliza: naquela escala os valores ficariam entre 0.9987 e 1.0
    e nenhum limiar util existiria.

    Mesmo assim, 0.30 esta muito abaixo do que esta base produz. Medido nos
    documentos ingeridos:

        documento 0 x documento 1 (mesmo manual, temas distintos)   0.690
        documento 0 x documento 2 (temas sem relacao)             0.720
        documento 1 x documento 2 (temas sem relacao)             0.738

    Os pares de conteudo irrelevante pontuam MAIS ALTO que o par relacionado.
    Com uma base de 3 chunks, todos do mesmo dominio tematico, o
    `nomic-embed-text` nao discrimina o suficiente para que qualquer corte em
    0.65-0.75 signifique alguma coisa: seria ajustar o limiar ao ruido.

    O que o gate faz entao, hoje, e o que ele faz de verdade: barra quando a
    busca nao devolve NENHUM trecho. Para calibrar de verdade, e preciso (1) uma
    base com dezenas de documentos de domínios distintos, (2) os scores anotados
    de cada caso de `evals/rag_cases.json` e (3) um limiar escolhido sobre essa
    distribuição, e nao sobre tres numeros. Ajustar o limiar antes disso produz
    uma regra que parece prudente e rejeita perguntas legitimas.

    Args:
        trechos: Pares ``(documento, similaridade)`` devolvidos por
            `buscar_com_scores()`, do mais para o menos semelhante.

    Returns:
        Bloqueio | None: `None` quando ha pelo menos um trecho acima do limiar;
        caso contrario, o bloqueio com a recusa de "fora do contexto" e o
        score do melhor trecho em `detalhe`.
    """
    if not trechos:
        return Bloqueio(
            resposta=RESPOSTA_FORA_DE_CONTEXTO,
            category="fora_do_contexto",
            origin="contexto",
            detalhe="nenhum trecho recuperado",
        )

    melhor_score = max(score for _, score in trechos)

    if melhor_score < LIMIAR_SIMILARIDADE:
        return Bloqueio(
            resposta=RESPOSTA_FORA_DE_CONTEXTO,
            category="fora_do_contexto",
            origin="contexto",
            detalhe=f"melhor similaridade {melhor_score:.4f} < {LIMIAR_SIMILARIDADE}",
        )

    return None


def _para_bloqueio(resultado: GuardrailResult, origin: str) -> Bloqueio:
    """Converte um `GuardrailResult` da Sprint 3 no `Bloqueio` deste modulo.

    Delega o texto ao `_blocked_response()` da Sprint 3 em vez de manter uma
    tabela propria, para que a recusa de "mostre o system prompt" continue
    identica a da Sprint 3. Divergir aqui significaria o usuario vendo dois
    textos diferentes para o mesmo ataque, dependendo de por onde a pergunta
    chegou.

    Args:
        resultado: Resultado devolvido por `moderate()` ou `validate_scope()`.
        origin: Etapa que produziu o resultado.

    Returns:
        Bloqueio: Bloqueio pronto para o pipeline devolver ao usuario.
    """
    return Bloqueio(
        resposta=_blocked_response(resultado),
        category=resultado.category,
        origin=origin,
        detalhe=resultado.reason,
    )