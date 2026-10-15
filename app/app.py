"""
Interface web oficial do chatbot RAG da Sprint 4 (FIAP + GoodWe) — integração
com a Sprint 3 (Memória).

A aplicação é um frontend Gradio fino: toda a inteligência (guardrails, recuperação
no ChromaDB, montagem do prompt com grounding e chamada ao Ollama) vive em
`src.rag.pipeline`. Este módulo cuida apenas da camada de apresentação: layout
em duas colunas, identificação do projeto, persistência do histórico por sessão
e renderização das fontes citadas.

FLUXO DE UM TURNO
=================

    texto digitado
       |
       +-- responder(pergunta, session_id)      [src.rag.pipeline]
       |     |
       |     +-- moderate() + validate_scope()  [Sprint 3] — pode bloquear
       |     +-- buscar_com_scores()            [Sprint 4] — pode bloquear
       |     +-- chain RAG | modelo             [Sprint 4]
       |     |
       |     -> {"resposta": str, "fontes": [...], "bloqueado": bool}
       |
       +-- formatar_resposta()  → balão do assistente (resposta + citações)
       +-- formatar_bloco_fontes() → coluna direita "📚 Fontes consultadas"
       +-- registrar_turno()    [Sprint 3] — só quando não bloqueado

INTEGRAÇÃO OFICIAL SPRINT 3 + SPRINT 4
======================================
A integração com a Sprint 3 é feita por dois módulos:

  * `src.rag.guardrails_rag`, que embrulha `moderate()` e `validate_scope()` da
    Sprint 3, é aplicado **antes** de qualquer chamada ao modelo;
  * `src.chain.memoria`, que guarda os turnos por `session_id`. O `session_id`
    é gerado uma vez por sessão do Gradio e fica em `gr.State`, então a
    continuidade ("e qual a corrente dele?") sobrevive aos reruns.

Por que a memória entra pelo pipeline e não pelo app: `responder()` já recebe
`session_id` e anexa o `<conversation_history>` à pergunta *antes* da busca
vetorial — o embedding precisa enxergar "e a corrente dele?" junto do turno que
utiliza o "ele". Duplicar essa montagem aqui renderia o histórico duas vezes.

POR QUE `registrar_turno()` SÓ RODA QUANDO NÃO BLOQUEIA
======================================================
`registrar_turno()` está fora de `responder()` de propósito: uma recusa de
guardrail não é uma conversa. Registrada na memória, ela voltaria como contexto
na pergunta seguinte e faria o modelo tratar o assunto barrado como se já
tivesse sido discutido. A interface respeita essa separação.

BLOCOIO: SEM FONTES, POR PRINCÍPIO
===================================
Quando `bloqueado` é verdadeiro, `fontes` vem vazio — um caminho que não
consultou a base não tem documento honesto para citar. A interface não inventa
citação para preencher a coluna direita: mostra a recusa em destaque e explica
que a base não foi consultada. Uma fonte exibida ao lado de uma recusa seria uma
alegação de que o modelo leu algo que ele não leu.

TESTABILIDADE
=============
`criar_app()` monta os `gr.Blocks` sem subir servidor, e a lógica de cada turno
está em funções puras (`formatar_resposta`, `formatar_bloco_fontes`) testáveis
sem Gradio, sem ChromaDB e sem Ollama. O import de `src.rag.pipeline` no topo
não abre conexão: ele só faz `from src.rag.retriever import ...`, e o cliente do
Ollama é construído dentro de `responder()`, no momento da chamada.

Execução (a partir da raiz do projeto):

    python -m app.app
    gradio app/app.py
"""

import argparse
import sys
import uuid
from pathlib import Path

import gradio as gr

# O `gradio app/app.py` insere somente a pasta `app/` no `sys.path`, então a raiz
# do projeto precisa ser adicionada para que o pacote `src` (que vive um nível
# acima) seja importável, independentemente do diretório de trabalho.
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

# INTEGRAÇÃO SPRINT 3 + SPRINT 4: a Sprint 3 é um projeto irmão, com a sua própria
# pasta raiz e o seu próprio pacote `src` (contendo `chain`, `guardrails` e
# `schemas`). Os dois `src` não possuem `__init__.py`, e portanto são *namespace
# packages* (PEP 420): ao colocar as duas raízes no `sys.path`, o interpretador as
# funde em um único pacote `src`, e `src.rag` (Sprint 4) passa a conviver com
# `src.guardrails` e `src.chain` (Sprint 3).
# ATENÇÃO: essa fusão só funciona enquanto nenhum dos dois `src` tiver um
# `src/__init__.py`; se algum deles ganhar um, o outro deixa de ser encontrado e
# os imports da Sprint 3 quebram com ModuleNotFoundError.
RAIZ_SPRINT3 = RAIZ_PROJETO / "goodwe-chatbot-sprint3"
if str(RAIZ_SPRINT3) not in sys.path:
    sys.path.insert(0, str(RAIZ_SPRINT3))

from src.rag.pipeline import registrar_turno, responder  # noqa: E402

INTEGRANTES = [
    "Rafael Lins — RM: 570588",
    "Rafael Gandolfi — RM: 569036",
    "Carlos Tavares — RM: 592949",
    "Cauã Paes — RM: 569906",
    "Guilherme Miranda — RM: 573107",
    "João Pedro Soler — RM: 569725",
]

TITULO = "GoodWe EV Chatbot — Assistente de Recarga em Condomínios"

# Onde os PDFs citados vivem. Usado apenas para montar o link clicável da
# citação; se o arquivo não estiver em disco (base ingerida em outra máquina),
# `formatar_citacoes` cai para texto puro em vez de gerar um link quebrado.
DIRETORIO_MANUAIS = RAIZ_PROJETO / "data" / "knowledge_base"

# Rótulo exibido no balão quando um guardrail recusa a pergunta.
MARCA_BLOQUEIO = "🚫 **Resposta não atendida — fontes indisponíveis.**"

CSS = """
/* Recusa de guardrail: destaca o balão sem depender de cor, para que a
   diferenciação também valha em alto contraste. */
.recusa {
    border-left: 4px solid #d9534f;
    padding: 0.6rem 0.9rem;
    border-radius: 4px;
    background: rgba(217, 83, 79, 0.08);
}
"""


def _link_da_fonte(source: str) -> str | None:
    """Devolve a URL interna do Gradio para o PDF citado, ou `None`.

    O `source` recuperado pelo RAG é apenas o nome do arquivo
    (`manual_goodwe.pdf`), porque `_normalizar_metados` de `src/rag/loader.py`
    guarda só `basename` e `page` para não inflar o ChromaDB. O caminho real
    precisa ser reconstituído aqui.

    Args:
        source: Nome do arquivo, tal como veio nos metadados do chunk.

    Returns:
        str | None: URL `/gradio_api/file=...` servida pelo Gradio, ou `None`
        quando o PDF não existe em `data/knowledge_base` — caso em que a citação
        é exibida como texto puro, sem link para um arquivo inexistente.
    """
    caminho = DIRETORIO_MANUAIS / Path(source).name

    if not caminho.is_file():
        return None

# O Gradio serve arquivos autorizados em `/gradio_api/file=<caminho>` (o
    # prefixo `/file=` era da série 5.x e responde 404 na 6); o caminho precisa
    # vir com "/" mesmo no Windows, e a âncora "#page=" abre o PDF na página.
    return "/gradio_api/file=" + caminho.as_posix()


def formatar_citacoes(fontes: list[dict]) -> str:
    """Monta a lista de citações `[arquivo.pdf, p. X]` da coluna direita.

    Cada fonte vira um bullet com link para o PDF na página citada e a
    similaridade do trecho, que é o número que justifica a citação: ela mostra
    o quão perto do texto da pergunta o trecho recuperado estava.

    Datasources repetidas no mesmo turno são preservadas (páginas diferentes do
    mesmo documento são citações distintas), mas a ordem segue a do pipeline,
    que já vem ranqueada por similaridade decrescente.

    Args:
        fontes: Lista de dicionários com `source`, `page` e `similaridade`.

    Returns:
        str: Markdown com um bullet por fonte, ou string vazia se não houver
        fontes. String vazia (e não um placeholder) porque o chamador decide
        entre "sem fontes" e "bloqueio" — a diferença é dele, não da lista.
    """
    linhas = []

    for fonte in fontes:
        source = fonte.get("source", "fonte_desconhecida")
        pagina = fonte.get("page", "n/d")
        similaridade = fonte.get("similaridade")
        rotulo = f"[{source}, p. {pagina}]"
        link = _link_da_fonte(source)

        if link:
            rotulo = f"[{rotulo}]({link}#page={pagina})"

        detalhe = f" — similaridade {similaridade:.2f}" if isinstance(similaridade, (int, float)) else ""
        linhas.append(f"- {rotulo}{detalhe}")

    return "\n".join(linhas)


def formatar_bloco_fontes(fontes: list[dict], bloqueado: bool) -> str:
    """Monta o conteúdo da seção "📚 Fontes consultadas" da coluna direita.

    Quando `bloqueado` é verdadeiro, devolve um aviso explicando que a base não
    foi consultada. A ausência de fontes é informação, não um espaço em branco:
    o usuário precisa distinguir "o assistente decidiu não responder" de "a base
    estava vazia".

    Args:
        fontes: Lista de fontes devolvida por `responder()`.
        bloqueado: Se a resposta veio de um caminho de recusa.

    Returns:
        str: Markdown da seção de fontes.
    """
    if bloqueado:
        return (
            "> **Nenhuma fonte consultada.**\n>\n"
            "> A pergunta foi respondida por um guardrail local, antes da "
            "busca na base. Nenhum documento foi recuperado e nenhuma citação "
            "seria honesta aqui."
        )

    if not fontes:
        return (
            "> **Nenhuma fonte consultada.**\n>\n"
            "> O modelo respondeu, mas o pipeline não devolveu trechos "
            "recuperados para esta pergunta."
        )

    return formatar_citacoes(fontes)


def formatar_resposta(resposta: str, bloqueado: bool) -> str:
    """Monta o texto do balão do assistente, com as citações logo abaixo.

    A resposta do modelo vem primeiro e as citações ficam imediatamente abaixo,
    separadas por uma régua: o usuário lê a resposta e, sem trocar de aba, vê
    em qual trecho ela se apoia.

    Args:
        resposta: Texto devolvido por `responder()`.
        bloqueado: Se a resposta veio de um caminho de recusa.

    Returns:
        str: Markdown do balão. Quando há bloqueio, a recusa é envolvidos em
        citação de bloco para ficar visualmente distinta de uma resposta comum.
    """
    if bloqueado:
        return f"> 🚫 {resposta}\n>\n> {MARCA_BLOQUEIO}"

    return resposta


def _novo_session_id() -> str:
    """Gera o identificador de sessão da Sprint 3.

    Guardado em `gr.State`, um id por sessão do Gradio. Como a memória é um
    dicionário indexado por esse id, sessões diferentes não compartilham
    histórico.

    Returns:
        str: Identificador hexadecimal único.
    """
    return uuid.uuid4().hex


def responder_turno(pergunta: str, historico: list[dict], session_id: str) -> tuple:
    """Processa um turno e devolve as saídas dos componentes do Gradio.

    Mantém as fontes *do turno atual* em `ultimas_fontes`, estado separado do
    histórico, porque a coluna direita responde "o que sustenta a última
    resposta?" — e não "quais documentos já foram citados na conversa?". As duas
    leituras divergem a partir do segundo turno.

    Um turno bloqueado não é registrado na memória da Sprint 3: ver o comentário
    sobre `registrar_turno` no docstring do módulo.

    Args:
        pergunta: Texto digitado pelo usuário.
        historico: Mensagens anteriores da sessão, no formato do `gr.Chatbot`.
        session_id: Identificador da sessão, usado pelo pipeline e pela memória.

    Returns:
        tuple: ``(historico, historico, fontes_formatadas, session_id)``.
        O histórico aparece duas vezes porque alimenta dois outputs — o
        `gr.Chatbot` e o próprio `gr.State` — e uma função só pode devolver
        uma saída por componente.
    """
    resultado = responder(pergunta, session_id=session_id)

    resposta = formatar_resposta(resultado["resposta"], resultado["bloqueado"])
    bloco_fontes = formatar_bloco_fontes(resultado["fontes"], resultado["bloqueado"])

    if not resultado["bloqueado"]:
        registrar_turno(session_id, pergunta, resultado["resposta"])

    mensagens = list(historico) + [
        {"role": "user", "content": pergunta},
        {"role": "assistant", "content": resposta},
    ]

    return mensagens, mensagens, bloco_fontes, session_id


def limpar_sessao(session_id: str) -> tuple:
    """Zera o histórico exibido e a memória da Sprint 3 da sessão.

    Limpar só a tela deixaria a memória por `session_id` viva: a conversa
    descartada voltaria como contexto na pergunta seguinte, por um caminho que
    o usuário não consegue ver.

    Args:
        session_id: Sessão a limpar.

    Returns:
        tuple: ``(historico_vazio, session_id)``.
    """
    from src.chain.memoria import get_session_history

    get_session_history(session_id).clear()

    return [], session_id


def criar_app() -> gr.Blocks:
    """Monta a interface sem subir servidor.

    Separado do `if __name__ == "__main__"` para que os testes possam construir
    os `gr.Blocks` e inspecionar a interface sem abrir porta nem chamar modelo.

    Returns:
        gr.Blocks: A aplicação montada, ainda não lançada.
    """
    # Gradio 6 moved `theme` and `css` from `Blocks(...)` to `launch()`; they
    # are accepted here only for backwards compatibility and emit a warning.
    with gr.Blocks(title=TITULO) as app:
        gr.HTML(
            f"<h1 style='text-align:center;margin-bottom:0'>{TITULO}</h1>"
            "<p style='text-align:center;opacity:0.75'>"
            "Respostas ancoradas nos manuais oficiais de recarga em "
            "condomínios. Todo conteúdo técnico consultado é citado com "
            "arquivo e página.</p>"
        )

        estado_historico = gr.State([])
        estado_sessao = gr.State(_novo_session_id)

        with gr.Row():
            # Coluna 1 — conversa
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversa",
                    height=520,
                )
                with gr.Row():
                    caixa = gr.Textbox(
                        label="Pergunte sobre recarga elétrica...",
                        placeholder="Ex.: qual a corrente nominal de um carregador de 7 kW?",
                        lines=2,
                        scale=5,
                    )
                    with gr.Column(scale=1, min_width=110):
                        botao_enviar = gr.Button(
                            "Enviar", variant="primary", scale=1
                        )
                        botao_limpar = gr.Button("Limpar", scale=1)

                gr.Markdown(
                    "**Integrantes** — " + " · ".join(INTEGRANTES)
                )

            # Coluna 2 — fontes do último turno
            with gr.Column(scale=2):
                painel_fontes = gr.Markdown(
                    "### 📚 Fontes consultadas\n\n"
                    "_As citações da última resposta aparecem aqui._"
                )
                gr.Markdown(
                    "---\n"
                    "<small>Fonte da base: manuais oficiais GoodWe e o "
                    "regimento do condomínio.</small>"
                )

        # `estado_historico` precisa aparecer em `saidas`, e não só em `entradas`:
        # um `gr.State` que é lido mas nunca reescrito continua valendo `[]` para
        # sempre, e a conversa mostraria apenas o último turno. O histórico volta
        # para o mesmo estado de onde foi lido, além de alimentar o `chatbot`.
        entradas = [caixa, estado_historico, estado_sessao]
        saidas = [chatbot, estado_historico, painel_fontes, estado_sessao]

        for gatilho in (caixa.submit, botao_enviar.click):
            gatilho(
                responder_turno,
                inputs=entradas,
                outputs=saidas,
            )

        botao_limpar.click(
            limpar_sessao,
            inputs=estado_sessao,
            outputs=[estado_historico, estado_sessao],
        )

    return app


def main(argv: list[str] | None = None) -> None:
    """Sobe a interface a partir da linha de comando.

    Gradio não lê `sys.argv` sozinho: sem o `argparse` abaixo, `python -m app.app
    --port 7890` abriria em 7860 e o usuário acreditaria que a porta foi honrada.
    Os argumentos saem daqui e são passados explicitamente para `launch()`.

    Args:
        argv: Argumentos a interpretar; `None` usa `sys.argv[1:]`.
    """
    parser = argparse.ArgumentParser(
        description="Interface web do chatbot RAG da Sprint 4 (FIAP + GoodWe)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="porta do servidor local (padrão do Gradio: 7860)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="endereço de escuta (ex.: 0.0.0.0 para acesso na rede)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="gera um link público temporário (requer internet)",
    )

    argumentos = parser.parse_args(argv)

    criar_app().launch(
        theme=gr.themes.Soft(),
        css=CSS,
        allowed_paths=[str(DIRETORIO_MANUAIS)],
        server_port=argumentos.port,
        server_name=argumentos.host,
        share=argumentos.share,
    )


if __name__ == "__main__":
    main()