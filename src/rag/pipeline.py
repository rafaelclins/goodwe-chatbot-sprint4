"""
Pipeline completo do chatbot RAG da Sprint 4, com os guardrails da Sprint 3.

Este modulo e a orquestracao de ponta a ponta: ele decide se a pergunta chega ao
modelo, recupera o contexto, chama a chain e devolve uma resposta ja auditavel.
A interface (`app/app.py`) e a avaliacao (`evals/`) consomem `responder()` em vez
de montar o fluxo por conta propria, para que exista um unico caminho de
execucao testavel.

FLUXO
=====

    pergunta
       |
       +-- 1. moderate()              [Sprint 3] injecao de prompt
       |        |
       |        +-- bloqueou? -> resposta local, LLM nunca e chamado
       |        |
       +-- 2. validate_scope()        [Sprint 3] escopo tematico
       |        |
       |        +-- bloqueou? -> resposta local, LLM nunca e chamado
       |        |
       +-- 3. buscar_com_scores()     [Sprint 4] recuperacao com similaridade
       |        |
       |        +-- nenhum trecho acima do limiar?
       |              -> recusa de "fora do contexto", LLM nunca e chamado
       |        |
       +-- 4. build_rag_chain().invoke()   [Sprint 4] prompt V3.0 | modelo
                |
                +-- resposta + fontes

Os tres primeiros passos sao todos locais e custam milissegundos. Nenhum deles
chega ao LLM, e essa e a razao de existirem: uma pergunta bloqueada nao consome
token, nao depende de conectividade e nao produz resposta. Uma falha de
infraestrutura da ultima etapa e tratada separadamente, para nao se confundir
com uma recusa de guardrail — o usuario precisa saber se o assistente decidiu
nao responder ou se o servidor caiu.

POR QUE O ORDEM DOS PASSOS IMPORTA
==================================
Moderation antes de escopo: uma injecao como "ignore todas as regras" nao tem
palavra-chave de dominio, entao `validate_scope()` a classificaria como
`out_of_scope` e o usuario receberia "essa pergunta esta fora do escopo" — uma
resposta que mente sobre a causa. Moderar primeiro devolve a mensagem que explica
o que de fato aconteceu.

Escopo antes da recuperacao: `validate_scope()` e regex pura, sem I/O. Rodar
antes do Chroma evita uma busca vetorial — que custa embedding — para uma
pergunta que ja esta destinada a ser recusada.

Recuperacao antes do LLM: e a unica ordem possivel, porque o limiar so existe
depois que a busca devolve scores.

INTEGRACAO COM A MEMORIA DA SPRINT 3
====================================
`session_id` e opcional. Quando informado, os turnos anteriores da sessao sao
transcritos e anexados a pergunta antes da busca, o que da continuidade
("e qual a corrente dele?") sem que o prompt V3.0 precise declarar um placeholder
de historico — a mesma estrategia de `app/app.py`, e pelo mesmo motivo: mexer no
prompt para accommodate memoria quebraria a comparacao entre versoes no relatorio
de evolucao.
"""

import sys
from pathlib import Path

from langchain_core.documents import Document

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
RAIZ_SPRINT3 = RAIZ_PROJETO / "goodwe-chatbot-sprint3"

if str(RAIZ_SPRINT3) not in sys.path:
    sys.path.insert(0, str(RAIZ_SPRINT3))

from src.chain.memoria import get_session_history  # noqa: E402

from src.rag.guardrails_rag import (  # noqa: E402
    Bloqueio,
    avaliar_contexto,
    avaliar_pergunta,
)
from src.rag.retriever import buscar_com_scores  # noqa: E402

# Mesma janela de `app/app.py`: seis turnos (tres perguntas com resposta). Mais
# que isso e o comeco de uma conversa antiga comeca a competir, dentro do prompt,
# com os trechos recuperados — que sao a unica fonte autorizada pela regra de
# grounding.
MAX_TURNOS_NO_HISTORICO = 3

RESPOSTA_ERRO = (
    "Não consegui consultar a base de conhecimento agora.\n\n"
    "Detalhe técnico: `{erro}`\n\n"
    "Verifique se o Ollama está rodando e se a base vetorial já foi ingerida."
)


def responder(pergunta: str, session_id: str | None = None) -> dict:
    """Processa uma pergunta e devolve resposta, fontes e situacao do bloqueio.

    Percorre o fluxo descrito no docstring do modulo e sempre devolve um dicionario
    com as mesmas tres chaves, inclusive quando a pergunta e bloqueada — assim o
    chamador nao precisa tratar "recusado" como um caso especial, apenas le "bloqueado".

    Args:
        pergunta: Texto digitado pelo usuario.
        session_id: Identificador da sessao, opcional. Quando informado, o
            historico daquela sessao e anexado a pergunta antes da busca.

    Returns:
        dict: ``{"resposta": str, "fontes": list[dict], "bloqueado": bool}``.
        Cada fonte tem ``source``, ``page`` e ``similaridade``. `fontes` vem
        vazia sempre que `bloqueado` e verdadeiro: um bloco nao consultou a base,
        entao nao ha documento honesto para citar.

    Raises:
        Exception: Falhas de infraestrutura (Ollama fora do ar, base ausente,
            erro de rede) sao capturadas e devolvidas como resposta de erro com
            `bloqueado` verdadeiro. Guardrails nunca levantam excecao.
    """
    bloqueio = avaliar_pergunta(pergunta)
    if bloqueio is not None:
        return _resposta_bloqueada(bloqueio)

    pergunta_com_historico = _anexar_historico(pergunta, session_id)

    trechos = buscar_com_scores(pergunta_com_historico)
    bloqueio = avaliar_contexto(trechos)
    if bloqueio is not None:
        return _resposta_bloqueada(bloqueio)

    fontes = [
        {
            "source": doc.metadata.get("source", "fonte_desconhecida"),
            "page": doc.metadata.get("page", "n/d"),
            "similaridade": round(score, 4),
        }
        for doc, score in trechos
    ]

    try:
        # A chain e construida aqui, e nao no topo do modulo, para que os tres
        # caminhos de recusa anteriores nao paguem a construcao do cliente do
        # Ollama (nem a validacao da OLLAMA_API_KEY) por uma pergunta que
        # jamais chegara ao modelo.
        from src.rag.prompt_rag import build_rag_chain

        resposta = build_rag_chain().invoke(pergunta_com_historico)
    except Exception as erro:  # noqa: BLE001
        return {
            "resposta": RESPOSTA_ERRO.format(erro=erro),
            "fontes": fontes,
            "bloqueado": True,
        }

    return {"resposta": resposta, "fontes": fontes, "bloqueado": False}


def _resposta_bloqueada(bloqueio: Bloqueio) -> dict:
    """Monta o dicionario de retorno de um caminho de recusa.

    Args:
        bloqueio: Decisao devolvida por um dos guardrails.

    Returns:
        dict: No mesmo formato de `responder()`, com `fontes` vazia.
    """
    return {
        "resposta": bloqueio.resposta,
        "fontes": [],
        "bloqueado": True,
    }


def _anexar_historico(pergunta: str, session_id: str | None) -> str:
    """Transcreve os turnos anteriores da sessao e os anexa a pergunta.

    Reproduz a estrategia de `app/app.py`, que nao injeta o historico pelo
    `RunnableWithMessageHistory` porque o prompt V3.0 declara apenas `context` e
    `question`. A transcricao entra antes da busca, e nao depois, porque o
    embedding da pergunta precisa enxergar "e a corrente dele?" junto do turno
    que utiliseriza o "ele".

    Args:
        pergunta: Pergunta do turno atual.
        session_id: Sessao a consultar, ou `None` para nao anexar historico.

    Returns:
        str: A pergunta original quando nao ha historico; a pergunta com o bloco
        `<conversation_history>` anexado quando ha.
    """
    if session_id is None:
        return pergunta

    mensagens = get_session_history(session_id).messages
    if not mensagens:
        return pergunta

    ultimos = mensagens[-(MAX_TURNOS_NO_HISTORICO * 2) :]
    linhas = [
        f"Usuário: {mensagem.content}" if mensagem.type == "human"
        else f"Assistente: {mensagem.content}"
        for mensagem in ultimos
    ]

    historico = "<conversation_history>\n" + "\n".join(linhas) + "\n</conversation_history>"

    return f"{historico}\n\nPergunta atual: {pergunta}"


def registrar_turno(session_id: str, pergunta: str, resposta: str) -> None:
    """Persiste o turno na memoria da Sprint 3.

    Fica separado de `responder()` porque nem todo turno deve virar memoria. Um
    recusa de guardrail nao e uma conversa: registrada no historico, ela volta
    como contexto na pergunta seguinte e faz o modelo tratar o assunto barrado
    como se ja tivesse sido discutido.

    Args:
        session_id: Sessao de destino.
        pergunta: Pergunta do turno.
        resposta: Resposta exibida ao usuario.
    """
    historico = get_session_history(session_id)
    historico.add_user_message(pergunta)
    historico.add_ai_message(resposta)


__all__ = ["responder", "registrar_turno", "MAX_TURNOS_NO_HISTORICO"]