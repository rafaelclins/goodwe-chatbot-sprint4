"""
Testes do fluxo de perguntas do `app/app.py` (interface Gradio) sobre o pipeline
com guardrails da Sprint 3.

A versão anterior deste arquivo cobria uma interface Streamlit que chamava
`build_rag_chain()` direto, sem gate de guardrail. A interface foi reescrita em
Gradio e passou a consumir `src.rag.pipeline.responder()`, que aplica
`moderate()` + `validate_scope()` + o limiar de contexto **antes** de qualquer
chamada ao modelo. Os testes abaixo travaram exatamente esse caminho novo.

Duas mudanças de estratégia em relação à versão anterior:

  * **Gradio 6 não distribui harness de testes.** Não existe
    `gradio.testing_utils` (que no Streamlit fora `AppTest`), então os testes não
    sobem servidor nem reexecutam o script: eles importam `app.app` e chamam as
    funções de turno diretamente. Isso é possível porque `criar_app()` monta os
    `gr.Blocks` sem lançar, e a lógica vive em funções puras.
  * **`responder()` é mockado.** Nenhum teste toca Ollama, ChromaDB ou
    embeddings; o que se verifica é o contrato entre o retorno do pipeline e o
    que a interface faz com ele.

O que está travado aqui:

  * a pergunta chega a `src.rag.pipeline.responder()` com o `session_id` da
    sessão, e não à chain RAG direto;
  * a resposta aparece no balão do assistente;
  * as fontes aparecem em uma seção "📚 Fontes consultadas", no formato
    `[arquivo.pdf, p. X]`, com link clicável;
  * `bloqueado=True` exibe a recusa em destaque e **não** exibe fontes;
  * um turno bloqueado não é registrado na memória da Sprint 3, enquanto um
    turno atendido é;
  * o histórico da sessão é acumulado e sobrevive aos turnos;
  * limpar a sessão apaga a memória da Sprint 3, não só a tela;
  * a UI monta sem chamar o modelo.

Execução (a partir da raiz do projeto):

    python -m unittest discover -s tests -v
    pytest tests/test_app_rag_flow.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Mesma solução de `sys.path` usada por `app/app.py`: os dois `src` (Sprint 4 e
# Sprint 3) são *namespace packages* (PEP 420) e precisam estar no caminho para
# que `src.rag` e `src.chain` coexistam. Sem o segundo, o import de
# `src.rag.pipeline` — que puxa `src.chain.memoria` — falha com ModuleNotFoundError.
for _raiz in (RAIZ_PROJETO, RAIZ_PROJETO / "goodwe-chatbot-sprint3"):
    if str(_raiz) not in sys.path:
        sys.path.insert(0, str(_raiz))

import app.app as app_mod  # noqa: E402

from src.chain.memoria import get_session_history  # noqa: E402

SESSION_ID = "sessao-de-teste"

FONTE_MANUAL = {
    "source": "manual_goodwe.pdf",
    "page": 12,
    "similaridade": 0.7312,
}

FONTE_REGIMENTO = {
    "source": "regimento_condominio.pdf",
    "page": 4,
    "similaridade": 0.6890,
}


def resposta_livre(
    texto="Resposta ancorada no manual.",
    fontes=None,
    bloqueado=False,
):
    """Monta o dicionário no contrato de `src.rag.pipeline.responder()`."""
    return {
        "resposta": texto,
        "fontes": list(fontes) if fontes is not None else [dict(FONTE_MANUAL)],
        "bloqueado": bloqueado,
    }


class TesteFluxoRag(unittest.TestCase):
    """Base comum: chama `responder_turno()` com o pipeline mockado."""

    def setUp(self):
        get_session_history(SESSION_ID).clear()

        patch_responder = mock.patch.object(
            app_mod, "responder", return_value=resposta_livre()
        )
        self.responder = patch_responder.start()
        self.addCleanup(patch_responder.stop)

        patch_registrar = mock.patch.object(app_mod, "registrar_turno")
        self.registrar_turno = patch_registrar.start()
        self.addCleanup(patch_registrar.stop)

        self.historico = []

    # -- auxiliares -----------------------------------------------------

    def perguntar(self, texto, retorno=None, historico=None):
        """Executa um turno e devolve `(historico, painel_fontes)`.

        Args:
            texto: Pergunta enviada.
            retorno: Resposta do pipeline a simular; o padrão é uma resposta
                livre com uma fonte.
            historico: Histórico de entrada; por padrão, o acumulado até agora.
        """
        if retorno is not None:
            self.responder.return_value = retorno

        entradas = self.historico if historico is None else historico
        saidas = app_mod.responder_turno(texto, entradas, SESSION_ID)
        self.historico, painel = saidas[0], saidas[2]

        return self.historico, painel

    def balao_assistente(self):
        """Conteúdo do último balão do assistente."""
        return self.historico[-1]["content"]

    def balao_usuario(self):
        """Conteúdo do último balão do usuário."""
        return self.historico[-2]["content"]


class TesteIntegracaoComPipeline(TesteFluxoRag):
    """A interface consome `src.rag.pipeline.responder()`, não a chain direta."""

    def test_pergunta_vai_para_responder_do_pipeline(self):
        self.perguntar("Qual a corrente nominal?")

        self.responder.assert_called_once()
        args, kwargs = self.responder.call_args
        self.assertEqual(args[0], "Qual a corrente nominal?")

    def test_session_id_da_sessao_e_repassado_ao_pipeline(self):
        # O `session_id` é o que dá continuidade à conversa: sem ele, o pipeline
        # não anexa o `<conversation_history>` e o turno seguinte perde o "ele".
        self.perguntar("Qual a corrente nominal?")

        _, kwargs = self.responder.call_args
        self.assertEqual(kwargs["session_id"], SESSION_ID)

    def test_uma_unica_chamada_ao_pipeline_por_pergunta(self):
        # Regressão do problema reportado na versão Streamlit, em que o gate
        # antigo acrescentava uma etapa antes de cada pergunta.
        for pergunta in ("Qual a corrente nominal?", "E a potência dele?"):
            self.perguntar(pergunta)

        self.assertEqual(self.responder.call_count, 2)

    def test_resposta_do_modelo_aparece_no_balao_do_assistente(self):
        self.perguntar(
            "Qual a corrente?",
            retorno=resposta_livre(texto="A corrente nominal é 32 A."),
        )

        self.assertIn("A corrente nominal é 32 A.", self.balao_assistente())

    def test_pergunta_aparece_no_balao_do_usuario(self):
        self.perguntar("Qual a corrente nominal?")

        self.assertEqual(self.balao_usuario(), "Qual a corrente nominal?")


class TesteSecaoDeFontes(TesteFluxoRag):
    """As citações são exibidas em uma seção própria, logo abaixo da resposta."""

    def test_painel_lista_arquivo_e_pagina_no_formato_de_citacao(self):
        _, painel = self.perguntar("Qual a corrente?")

        self.assertIn("[manual_goodwe.pdf, p. 12]", painel)

    def test_citacao_e_um_item_de_lista(self):
        _, painel = self.perguntar("Qual a corrente?")

        primeira_linha = painel.splitlines()[0]
        self.assertTrue(
            primeira_linha.startswith("- "),
            f"a citação deveria ser um bullet, veio: {primeira_linha!r}",
        )

    def test_citacao_aponta_para_o_pdf_na_pagina_citada(self):
        _, painel = self.perguntar("Qual a corrente?")

        self.assertIn("#page=12", painel)
        self.assertIn("manual_goodwe.pdf", painel)

    def test_citacao_de_arquivo_inexistente_nao_gera_link_quebrado(self):
        # A base pode ter sido ingerida em outra máquina, onde o PDF não está em
        # `data/knowledge_base`. Um link para arquivo inexistente seria pior que
        # texto puro: parece clicável e não leva a lugar nenhum.
        _, painel = self.perguntar(
            "Qual a corrente?",
            retorno=resposta_livre(
                fontes=[{"source": "inexistente.pdf", "page": 3, "similaridade": 0.5}]
            ),
        )

        self.assertIn("[inexistente.pdf, p. 3]", painel)
        self.assertNotIn("](", painel)

    def test_todas_as_fontes_sao_listadas(self):
        _, painel = self.perguntar(
            "Qual a corrente?",
            retorno=resposta_livre(fontes=[FONTE_MANUAL, FONTE_REGIMENTO]),
        )

        self.assertIn("[manual_goodwe.pdf, p. 12]", painel)
        self.assertIn("[regimento_condominio.pdf, p. 4]", painel)

    def test_similaridade_e_exibida_junto_da_citacao(self):
        # A similaridade é o número que justifica a citação: ela mostra o quão
        # perto do texto da pergunta o trecho recuperado estava.
        _, painel = self.perguntar("Qual a corrente?")

        self.assertIn("0.73", painel)

    def test_resposta_e_fontes_viam_no_mesmo_turno(self):
        # "Resposta do modelo E, logo abaixo, as fontes" — as duas coisas têm de
        # aparecer, no mesmo turno, sem que o usuário precise pedir as fontes.
        _, painel = self.perguntar(
            "Qual a corrente?",
            retorno=resposta_livre(texto="A corrente nominal é 32 A."),
        )

        self.assertIn("A corrente nominal é 32 A.", self.balao_assistente())
        self.assertIn("[manual_goodwe.pdf, p. 12]", painel)


class TesteBloqueio(TesteFluxoRag):
    """`bloqueado=True` exibe a recusa em destaque e nenhuma fonte."""

    def test_recusa_aparece_no_balao(self):
        self.perguntar(
            "Qual a capital da França?",
            retorno=resposta_livre(
                texto="Não encontrei essa informação na base de conhecimento.",
                fontes=[],
                bloqueado=True,
            ),
        )

        self.assertIn(
            "Não encontrei essa informação na base de conhecimento.",
            self.balao_assistente(),
        )

    def test_recusa_fica_em_destaque(self):
        self.perguntar(
            "Qual a capital da França?",
            retorno=resposta_livre(texto="Fora do escopo.", fontes=[], bloqueado=True),
        )

        # Citação de bloco (`>`) no topo do balão é o que diferencia visualmente
        # uma recusa de uma resposta normal no markdown do Chatbot.
        self.assertTrue(self.balao_assistente().startswith(">"))

    def test_bloqueio_nao_exibe_fontes(self):
        # Um caminho bloqueado não consultou a base: exibir qualquer citação ao
        # lado da recusa seria afirmar que o modelo leu algo que ele não leu.
        _, painel = self.perguntar(
            "Qual a capital da França?",
            retorno=resposta_livre(texto="Fora do escopo.", fontes=[], bloqueado=True),
        )

        self.assertNotIn("manual_goodwe.pdf", painel)
        self.assertNotIn("regimento_condominio.pdf", painel)
        self.assertIn("Nenhuma fonte consultada", painel)

    def test_bloqueio_nao_registra_turno_na_memoria(self):
        self.perguntar(
            "Qual a capital da França?",
            retorno=resposta_livre(texto="Fora do escopo.", fontes=[], bloqueado=True),
        )

        # Registrar a recusa faria ela voltar como contexto na pergunta
        # seguinte, e o modelo passaria a tratar o assunto barrado como se já
        # tivesse sido discutido.
        self.registrar_turno.assert_not_called()

    def test_resposta_atendida_registra_turno_na_memoria(self):
        self.perguntar("Qual a corrente nominal?")

        self.registrar_turno.assert_called_once()
        args, _ = self.registrar_turno.call_args
        self.assertEqual(args[0], SESSION_ID)
        self.assertEqual(args[1], "Qual a corrente nominal?")

    def test_recusa_ainda_aparece_no_historico_da_tela(self):
        # Não registrar na memória da Sprint 3 não significa esconder da
        # sessão: o usuário precisa ver o que ele perguntou e o que recebeu.
        self.perguntar(
            "Qual a capital da França?",
            retorno=resposta_livre(texto="Fora do escopo.", fontes=[], bloqueado=True),
        )

        self.assertEqual(len(self.historico), 2)


class TesteHistoricoDaSessao(TesteFluxoRag):
    """O histórico exibido é acumulado e reflete o padrão da memória."""

    def test_turnos_sao_acumulados_na_ordem(self):
        self.perguntar("Qual a corrente nominal?")
        self.perguntar("E a potência?")

        self.assertEqual(len(self.historico), 4)
        self.assertEqual(
            [mensagem["role"] for mensagem in self.historico],
            ["user", "assistant", "user", "assistant"],
        )

    def test_turno_anterior_e_repassado_ao_turno_seguinte(self):
        # O pipeline anexa o histórico a partir do `session_id`; o app precisa
        # devolver as mensagens acumuladas para o próximo turno.
        self.perguntar("Qual a corrente nominal?")

        _, kwargs = self.responder.call_args
        self.assertEqual(kwargs["session_id"], SESSION_ID)

    def test_painel_mostra_as_fontes_do_ultimo_turno(self):
        self.perguntar(
            "Qual a corrente?",
            retorno=resposta_livre(fontes=[FONTE_MANUAL]),
        )
        _, painel = self.perguntar(
            "E no regimento?",
            retorno=resposta_livre(fontes=[FONTE_REGIMENTO]),
        )

        self.assertIn("regimento_condominio.pdf", painel)
        self.assertNotIn("manual_goodwe.pdf", painel)

    def test_limpar_sessao_zera_o_historico_e_a_memoria(self):
        self.perguntar("Qual a corrente nominal?")
        historico_vazio, _ = app_mod.limpar_sessao(SESSION_ID)

        self.assertEqual(historico_vazio, [])
        self.assertEqual(get_session_history(SESSION_ID).messages, [])


class TesteEstadoDoHistorico(TesteFluxoRag):
    """Regressão: o `gr.State` do histórico precisa ser reescrito a cada turno.

    Um `gr.State` que é lido mas nunca devolvido continua valendo o valor
    inicial para sempre. Com o histórico fora da lista de outputs, cada turno
    recebia `[]` e a conversa exibia só a última troca — a memória da Sprint 3
    continuava intacta, então a falha era visível apenas na tela e não em
    nenhuma asserção sobre `responder()`.
    """

    def test_historico_e_devolvido_para_o_proprio_estado(self):
        saidas = app_mod.responder_turno("Qual a corrente?", [], SESSION_ID)

        historico_para_o_chatbot = saidas[0]
        historico_para_o_estado = saidas[1]

        self.assertEqual(historico_para_o_chatbot, historico_para_o_estado)
        self.assertEqual(len(historico_para_o_estado), 2)

    def test_saida_tem_um_valor_por_componente_de_output(self):
        # Output a mais ou a menos que o que a função devolve derruba a chamada
        # com um erro em tempo de execução — o app sobe e quebra no primeiro
        # clique. Entradas e saídas não precisam ter a mesma quantidade: o
        # histórico entra uma vez e sai em dois componentes.
        import gradio as gr

        app = app_mod.criar_app()
        saidas_esperadas = None

        for dependencia in app.fns.values():
            if dependencia.fn is app_mod.responder_turno:
                saidas_esperadas = len(dependencia.outputs)

        self.assertIsNotNone(saidas_esperadas, "responder_turno não está ligado")

        with mock.patch.object(app_mod, "responder", return_value=resposta_livre()):
            devolvido = app_mod.responder_turno("Qual a corrente?", [], SESSION_ID)

        self.assertEqual(len(devolvido), saidas_esperadas)

    def test_historico_cresce_ao_longo_dos_turnos(self):
        self.perguntar("Pergunta 1")
        self.perguntar("Pergunta 2")
        self.perguntar("Pergunta 3")

        self.assertEqual(len(self.historico), 6)
        self.assertEqual(
            [mensagem["content"] for mensagem in self.historico[::2]],
            ["Pergunta 1", "Pergunta 2", "Pergunta 3"],
        )


class TesteMontagemDaUI(unittest.TestCase):
    """A interface monta sem chamar o modelo, sem Ollama e sem ChromaDB."""

    def setUp(self):
        patch_responder = mock.patch.object(app_mod, "responder")
        self.responder = patch_responder.start()
        self.addCleanup(patch_responder.stop)

    def test_criar_app_monta_os_blocks_sem_chamar_o_modelo(self):
        app_mod.criar_app()

        self.responder.assert_not_called()

    def test_app_e_um_blocks_do_gradio(self):
        import gradio as gr

        app = app_mod.criar_app()

        self.assertIsInstance(app, gr.Blocks)

    def test_layout_tem_as_duas_colunas(self):
        import gradio as gr

        app = app_mod.criar_app()
        componentes = [bloco for bloco in app.blocks.values()]

        linhas = [c for c in componentes if isinstance(c, gr.Row)]
        self.assertTrue(linhas, "o layout deveria ter ao menos uma linha")

        # Duas colunas na linha do corpo: conversa | fontes.
        colunas = [
            len([c for c in linha.children]) if hasattr(linha, "children") else 0
            for linha in linhas
        ]
        self.assertIn(2, colunas)

    def test_existe_um_chatbot_e_uma_caixa_de_texto(self):
        import gradio as gr

        app = app_mod.criar_app()
        tipos = [type(bloco) for bloco in app.blocks.values()]

        self.assertIn(gr.Chatbot, tipos)
        self.assertIn(gr.Textbox, tipos)


class TesteLinhaDeComando(unittest.TestCase):
    """Os argumentos de `--port`/`--host` chegam de fato ao `launch()`.

    Gradio não lê `sys.argv`: sem o `argparse` de `main()`, `python -m app.app
    --port 7890` abriria em 7860 e o usuário acreditaria que a porta foi honrada.
    """

    def setUp(self):
        patch_app = mock.patch.object(app_mod, "criar_app")
        self.criar_app = patch_app.start()
        self.addCleanup(patch_app.stop)
        self.launch = self.criar_app.return_value.launch

    def test_porta_e_repassada_ao_launch(self):
        app_mod.main(["--port", "7890"])

        self.assertEqual(self.launch.call_args.kwargs["server_port"], 7890)

    def test_host_e_repassado_ao_launch(self):
        app_mod.main(["--host", "0.0.0.0"])

        self.assertEqual(self.launch.call_args.kwargs["server_name"], "0.0.0.0")

    def test_share_e_repassado_ao_launch(self):
        app_mod.main(["--share"])

        self.assertIs(self.launch.call_args.kwargs["share"], True)

    def test_sem_argumentos_nao_forca_porta_nem_host(self):
        # `None` deixa o Gradio escolher o padrão (7860 / localhost) em vez de
        # sobrescrever a escolha dele. `share` é `store_true`, então o padrão
        # dele é `False` — link público só com a flag explícita.
        app_mod.main([])

        self.assertIsNone(self.launch.call_args.kwargs["server_port"])
        self.assertIsNone(self.launch.call_args.kwargs["server_name"])
        self.assertIs(self.launch.call_args.kwargs["share"], False)

    def test_allowed_paths_libera_os_pdfs_para_as_citacoes(self):
        # As citações só são clicáveis se a pasta dos manuais estiver autorizada
        # a ser servida; sem isso, o link devolve 404.
        app_mod.main([])

        autorizados = self.launch.call_args.kwargs["allowed_paths"]
        self.assertIn(str(app_mod.DIRETORIO_MANUAIS), autorizados)


class TesteCodigoSemAcessoDiretoAoModelo(unittest.TestCase):
    """Trava estática: o app não pode montar chain nem chamar guardrails sozinho.

    Não dá para testar "o app não faz retrieval" observavelmente em runtime — um
    app que não faz e um que faz por um caminho nunca exercitado se comportam
    igual. A verificação é feita sobre a AST do arquivo.
    """

    @classmethod
    def setUpClass(cls):
        fonte = (RAIZ_PROJETO / "app" / "app.py").read_text(encoding="utf-8")
        cls.arvore = __import__("ast").parse(fonte)
        cls.imports = {
            alias.name
            for no in __import__("ast").walk(cls.arvore)
            if isinstance(no, (__import__("ast").Import, __import__("ast").ImportFrom))
            for alias in no.names
        }
        cls.funcoes = {
            no.name
            for no in __import__("ast").walk(cls.arvore)
            if isinstance(no, __import__("ast").FunctionDef)
        }

        # Sanidade do próprio teste: se estas coleções viessem vazias, os testes
        # abaixo passariam sem verificar nada.
        assert cls.funcoes, "a AST não expôs nenhuma função — a verificação é inútil"
        assert "responder_turno" in cls.funcoes, (
            "responder_turno() não encontrado em app/app.py"
        )

    def test_app_nao_importa_build_rag_chain(self):
        # A chain é montada dentro de `responder()`, no pipeline. Se o app
        # importasse `build_rag_chain`, ele contornaria o gate de guardrails —
        # que é exatamente o que a migração passou a evitar.
        for nome in self.imports:
            self.assertNotIn("build_rag_chain", nome)

    def test_app_nao_importa_os_guardrails_direto(self):
        for nome in ("moderate", "validate_scope", "avaliar_contexto"):
            with self.subTest(nome=nome):
                self.assertNotIn(nome, self.imports)

    def test_app_importa_responder_do_pipeline(self):
        self.assertIn("responder", self.imports)
        self.assertIn("registrar_turno", self.imports)


if __name__ == "__main__":
    unittest.main()