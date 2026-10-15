"""
Avaliação do pipeline RAG (Sprint 4) com as métricas *faithfulness* e
*answer_relevancy* do RAGAS, sobre os casos de `evals/rag_cases.json`.

Cada caso do arquivo passa pelo pipeline real (`src.rag.pipeline.responder`) e
os dois artefatos que a métrica precisa são capturados: a **resposta** gerada e
os **contextos** (os chunks recuperados). A fonte da verdade do grounding é o
mesmo texto que foi entregue ao modelo, nunca os chunks crus: as âncoras de
fidelidade são os rótulos `[arquivo.pdf, p. X]` que `format_docs()` monta.

EXECUÇÃO

    python evals/run_rag_eval.py                      # RAGAS, judge Ollama
    python evals/run_rag_eval.py --manual             # rubrica manual
    python evals/run_rag_eval.py --output evals/iteration_2.json

    python evals/run_rag_eval.py --judge-model llama3
    python evals/run_rag_eval.py --limit 2            # iteração rápida

O `--output` é o que torna o script reexecutável para iterações: nada é
sobrescrito por acidente, e cada rodada fica num arquivo próprio
(`iteration_1.json`, `iteration_2.json`, ...), que é a base da comparação de
evolução entre versões de prompt.

===============================================================================
O SHIM: POR QUE O RAGAS PRECISA DE UM ADUENDO PARA IMPORTAR
===============================================================================

`evals/ragas_eval.py` registra que o RAGAS foi abandonado por um conflito de
dependência considerado intransponível:

    from ragas import evaluate
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

Isso continua verdadeiro, e a causa raiz também: o `ragas` 0.4.3 importa, em
`ragas/llms/base.py`, o caminho legado `langchain_community.chat_models.vertexai`,
mas o `langchain-community` 0.4.2 instalado não tem esse submódulo — ele foi
movido para `langchain_community/llms/vertexai.py`. Como a falha ocorre no
`import`, nenhuma métrica carrega.

A conclusão de "intransponível", porém, não se sustenta mais, e a verificação
deste script a refaz:

  * **Subir** o `langchain-community` não resolve (ele está sendo descontinuado,
    sem versão que restaure o caminho antigo);
  * **Descer** para a série 0.2.x rebaixaria o `langchain-core` para < 1.0 e
    quebraria a chain LCEL da Sprint 3 e o `langchain-chroma` — logo, é
    realmente intranponível *por via de downgrade*;
  * mas existe uma terceira via, que o arquivo original não chegou a considerar:
    **satisfazer o import sem tocar em nenhuma dependência**.

O RAGAS importa `ChatVertexAI` de `langchain_community.chat_models.vertexai`
apenas para reconhecê-lo em um `isinstance` dentro de um mapa de providers
(*value-based*, não estrutural). Ele nunca é instanciado quando o judge é o
Ollama. `_instalar_shim_ragas()` registra, em `sys.modules`, um módulo
sintético com um `ChatVertexAI` que nunca será instanciado, satisfazendo o
import. Nenhum pacote é instalado, rebaixado ou alterado — o adendo vive no
`sys.modules` do processo e desaparece com ele.

    python evals/run_rag_eval.py            # RAGAS agora funciona
    python evals/ragas_eval.py              # a rubrica manual segue disponível

===============================================================================
O JUDGE: OLLAMA VIA ENDPOINT COMPATÍVEL COM OPENAI
===============================================================================

O judge é o `gpt-oss:20b` a `temperature=0`, servido pelo Ollama. Ele é
instanciado com `llm_factory(provider="openai")` apontando para
`http://localhost:11434/v1`, e não com `ChatOllama` + `LangchainLLMWrapper`, por
uma razão verificada neste ambiente: no RAGAS 0.4.3 as métricas de
`ragas.metrics.collections` **recusam** o wrapper do LangChain —

    ValueError: Collections metrics only support modern InstructorLLM.
    Found: LangchainLLMWrapper.

A separação é estrutural no 0.4.3: `Faithfulness` e `AnswerRelevancy` estendem
`SimpleBaseMetric`, que não é a classe `ragas.metrics.Metric` usada por
`evaluate()`. Por isso o script chama `score()` das instâncias diretamente, em
vez de passar os objetos para `evaluate()`, que recusaria o tipo. O mesmo vale
para os embeddings, que exigem `interface="modern"`.

Isso não é um contorno do Ollama: o judge continua sendo o Ollama local, sem
nenhuma chamada externa. É apenas o caminho que o RAGAS 0.4.3 aceita.

===============================================================================
BLOQUEIO: POR QUE CASO BLOQUEADO NÃO ENTRA NA MÉDIA
===============================================================================

Um caso cujo `bloqueado` é verdadeiro não tem resposta do modelo — quem
respondeu foi um guardrail local, antes da busca. Avaliar fidelidade e
relevância de uma frase de recusa não faria sentido: ela não foi gerada a
partir do contexto e o RAGAS a compararia com um contexto que nem chegou a ser
consultado.

Esses casos são gravados no JSON com `"bloqueado": true` e as notas como
`null`, e ficam **fora** da média, que passa a ser sobre as respostas que o
modelo de fato produziu. Descartá-los da base sem registro, porém, seria
perder informação: o comportamento do grounding é justamente o que o caso
`conceptual_demand` (teste negativo) existe para medir.
"""

import argparse
import json
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

# Os dois `src` (Sprint 4 e Sprint 3) são *namespace packages* (PEP 420) e
# precisam estar no `sys.path` para que `src.rag` conviva com `src.chain`. Sem o
# segundo, `src.rag.pipeline` — que importa `src.chain.memoria` — falha.
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
for _raiz in (RAIZ_PROJETO, RAIZ_PROJETO / "goodwe-chatbot-sprint3"):
    if str(_raiz) not in sys.path:
        sys.path.insert(0, str(_raiz))

ARQUIVO_CASOS = RAIZ_PROJETO / "evals" / "rag_cases.json"
ARQUIVO_RUBRICA = RAIZ_PROJETO / "evals" / "rubrica_manual.md"

MODELO_JUDGE_PADRAO = "gpt-oss:20b"
MODELO_EMBEDDING_PADRAO = "nomic-embed-text"
BASE_URL_OLLAMA = "http://localhost:11434"


def _construir_judge(modelo: str) -> tuple[Any, Any]:
    """Monta o LLM judge e os embeddings do RAGAS sobre o Ollama local.

    Usa o endpoint compatível com OpenAI do Ollama (`/v1`) com clientes
    assíncronos: as métricas chamam `agenerate()` e um cliente síncrono falha com
    "Cannot use agenerate() with a synchronous client".

    Args:
        modelo: Nome do modelo de judge, ex.: `gpt-oss:20b`.

    Returns:
        tuple: `(llm, embeddings)`, prontos para as instâncias de métrica.
    """
    _instalar_shim_ragas()

    from openai import AsyncOpenAI

    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    cliente = AsyncOpenAI(base_url=f"{BASE_URL_OLLAMA}/v1", api_key="ollama")

    llm = llm_factory(model=modelo, provider="openai", client=cliente)
    embeddings = embedding_factory(
        model=MODELO_EMBEDDING_PADRAO,
        provider="openai",
        client=cliente,
        interface="modern",
    )

    return llm, embeddings


# ===============================================================================
# OS DOIS BLOCOS DE AVALIAÇÃO
# ===============================================================================
# Cada caso de `evals/rag_cases.json` declara `tipo`:
#
#   "answerable"       a base responde. O comportamento correto é responder e
#                      citar. Avaliado pelo RAGAS (faithfulness + answer
#                      relevancy).
#
#   "refusal_expected" a base NÃO responde. O comportamento correto é recusar.
#                      Avaliado por rubrica binária: 1 = recusou, 0 = respondeu.
#
# POR QUE O RAGAS NÃO AVALIA OS CASOS `refusal_expected`
#
# Porque a `answer_relevancy` do RAGAS é calculada por similaridade de
# embeddings entre a pergunta e a resposta, e penaliza qualquer resposta que não
# "pareça" com a pergunta. Uma recusa correta é justamente o oposto disso: é uma
# resposta que se recusa a falar, e ela pontua baixo por definição.
#
# O efeito medido nesta base foi o inverso do desejado. Em `iteration_1.json`
# (prompt v3.0), o caso `conceptual_demand` — em que os dois prompts recusam
# corretamente — recebeu:
#
#     answer_relevancy = 0.81   (v3.0)
#     answer_relevancy = 0.00   (v3.1)
#
# Mesmo comportamento, notas diferentes: o judge acompanhou a redação do segundo
# parágrafo, não o fato de ter recusado. Uma métrica cuja nota muda com a
# formulação da recusa não está medindo o grounding, está medindo o judge.
#
# Por isso os dois blocos nunca são somados numa média única, e por isso a recusa
# é pontuada por contagem direta — "recusou sim ou não" — e não por similaridade.
# Ver `evals/rubrica_manual.md` §3.4, que já dizia que recusar sem necessidade
# penaliza relevância; aqui o problema é o inverso, e também precisa de regra.

TIPOS_VALIDOS = ("answerable", "refusal_expected")

# A frase que o prompt manda usar quando não recusa. A detecção é normalizada
# (sem acentos, minúscula) porque o texto volta do modelo com acentuação
# variável dependendo da página de codificação do console.
FRASE_RECUSA = "Não encontrei essa informação na base de conhecimento."


def _sem_acento(texto: str) -> str:
    """Minúscula e sem acento, para comparação de frases insensitive a diacríticos."""
    import unicodedata

    decomposto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def classificar_recusa(registro: dict[str, Any]) -> str:
    """Diz como um caso `refusal_expected` foi respondido.

    Três desfechos, e só um pontua 1:

    - `recusa_guardrail` — o filtro local (`moderate()`, `validate_scope()`, gate
      de contexto) barrou antes do LLM. O grounding funcionou: 1.
    - `recusa_modelo` — o LLM respondeu com a frase de recusa do prompt. 1.
    - `respondeu` — o LLM respondeu com conteúdo. Para um caso
      `refusal_expected` isso é falha: 0. Respondeu porque o contexto trazia
      trechos parecidos, mas não a resposta.

    Um caso `refusal_expected` que responde não é automaticamente alucinação —
    pode ser uma resposta correta a uma pergunta que o conjunto(classificou mal.
    Por isso o campo `resposta` fica gravado no JSON: a nota 0 precisa ser
    auditável, e é a revisão do avaliador que distingue os dois casos.

    Returns:
        str: Um de `recusa_guardrail`, `recusa_modelo` ou `respondeu`.
    """
    if registro.get("erro_infraestrutura"):
        return "erro_infraestrutura"

    if registro.get("bloqueado"):
        return "recusa_guardrail"

    resposta = registro.get("resposta") or ""

    if not resposta.strip():
        return "respondeu"

    if _sem_acento(FRASE_RECUSA) in _sem_acento(resposta):
        return "recusa_modelo"

    return "respondeu"


def avaliar_recusas(registros: list[dict[str, Any]]) -> None:
    """Aplica a rubrica binária aos casos `refusal_expected`.

    Recebe os registros por referência e escreve `recusa_resultado` e
    `recusa_nota` em cada um. Registros de outros tipos são ignorados: eles são
    pontuados pelo RAGAS, e o `tipo` ausente em um arquivo antigo é tratado como
    `answerable` para não perder a iteração inteira.

    A nota é binária de propósito. Uma escala intermediária exigiria julgar o
    quanto uma recusa é "parcialmente correta", que é justamente o julgamento
    que produziu o artefato de 0.81 x 0.00 documentado acima.
    """
    for registro in registros:
        if registro.get("tipo") != "refusal_expected":
            continue

        resultado = classificar_recusa(registro)
        registro["recusa_resultado"] = resultado
        registro["recusa_nota"] = (
            None
            if resultado == "erro_infraestrutura"
            else (1.0 if resultado.startswith("recusa") else 0.0)
        )


def _configurar_stdout() -> None:
    """Força UTF-8 no stdout antes do primeiro `print`.

    Sem isso, o script morre em qualquer execução redirecionada no Windows. O
    console padrão do Python ali é `cp1252`, que não tem `✓` (U+2713), `⚠️`
    (U+26A0) nem `⏭` (U+23ED) — todos usados nos prints de progresso. O
    `UnicodeEncodeError` acontece dentro de `avaliar_com_ragas()`, e o
    `except` que deveria cair na rubrica manual também imprime `⚠️`, então ele
    estoura de novo antes de gravar o JSON: a iteração inteira é perdida sem
    mensagem útil.

    Corrigir pelo ambiente (`PYTHONIOENCODING=utf-8`) resolve, mas exige que
    quem roda lembre disso. Fazer aqui deixa o script correto por conta própria,
    inclusive quando chamado por CI ou por outro script.
    """
    for fluxo in (sys.stdout, sys.stderr):
        # `reconfigure` só existe em TextIOWrapper. Sob redirecionamento para um
        # arquivo o stdout continua sendo um, mas sob `pythonw` ou capturadores
        # captura de saída pode não ser — daí o `getattr` em vez de acesso direto.
        reconfigurar = getattr(fluxo, "reconfigure", None)
        if reconfigurar is not None:
            reconfigurar(encoding="utf-8", errors="replace")


def _versao_prompt_em_uso() -> str:
    """Devolve a versão do prompt que a chain vai usar, para gravar no JSON.

    Sem este campo, `iteration_1.json` e `iteration_2.json` são indistinguíveis
    pelo conteúdo: os dois descrevem casos, respostas e notas, e nada diz qual
    prompt as produziu. A tabela de versões de `evals/README.md` depende
    exatamente desse dado.

    A versão é lida do mesmo lugar que a chain lê, então não há como o JSON
    discordar do que foi executado. Uma falha aqui não derruba a iteração: o
    eval ainda é válido, só perde a identificação do prompt.
    """
    try:
        from src.rag.prompt_rag import get_versao_prompt

        return get_versao_prompt()
    except Exception as erro:  # noqa: BLE001
        return f"indisponivel ({type(erro).__name__}: {erro})"


def _verificar_modelo_disponivel(modelo: str) -> None:
    """Falha cedo e com orientação se o modelo de judge não puder ser chamado.

    Sem isso, a falha só apareceria na primeira pergunta, depois de a base ter
    sido percorrida, e a mensagem seria a do Ollama sobre o modelo inexistente —
    que não diz o que fazer.

    A verificação sonda `/api/show` em vez de consultar `/api/tags`. A lista de
    tags é enganosa para o que este projeto usa: ela devolve o que o Ollama já
    conhece localmente, e os modelos da nuvem só entram nela depois de um
    primeiro uso bem-sucedido. Verificado nesta máquina, `/api/tags` devolvia
    `gemma4:cloud, nomic-embed-text, llama3, qwen3.5` — e `gpt-oss:120b-cloud`,
    que responde normalmente quando chamado, não estava na lista. Basear a
    checagem nela rejeitaria um judge perfectly funcional.

    `/api/show` pergunta ao servidor se ele consegue resolver o modelo, que é
    exatamente a pergunta que importa, e distingue as três situações que
    importam: servidor fora do ar, modelo ausente, modelo utilizável.

    Args:
        modelo: Nome do modelo de judge.

    Raises:
        SystemExit: Se o servidor do Ollama não responder ou o modelo não existir.
    """
    import urllib.error
    import urllib.request

    sonda = json.dumps({"model": modelo}).encode("utf-8")
    requisicao = urllib.request.Request(
        f"{BASE_URL_OLLAMA}/api/show",
        data=sonda,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=30) as r:
            json.load(r)
    except urllib.error.HTTPError as erro:
        # Resposta 4xx/5xx do próprio Ollama: o servidor está no ar, mas não
        # conhece este modelo. A distinção importa porque a orientação é outra.
        try:
            disponiveis = _listar_modelos()
            lista = ", ".join(sorted(disponiveis)) or "(nenhum)"
        except Exception:  # noqa: BLE001
            lista = "(não foi possível listar)"

        raise SystemExit(
            f"✗ O modelo de judge '{modelo}' não está disponível no Ollama.\n"
            f"  Modelos que o servidor conhece: {lista}\n"
            f"  Baixe com: ollama pull {modelo}\n"
            f"  Modelos de nuvem também podem responder pelo nome exato do "
            f"catálogo, ex.: --judge-model gpt-oss:120b-cloud"
        ) from erro
    except (urllib.error.URLError, OSError, TimeoutError) as erro:
        raise SystemExit(
            f"✗ Não consegui falar com o Ollama em {BASE_URL_OLLAMA} ({erro}).\n"
            f"  Ele precisa estar rodando: `ollama serve`"
        ) from erro


def _listar_modelos() -> set[str]:
    """Nomes de modelo que o servidor do Ollama conhece, para a mensagem de erro.

    Só alimenta a orientação: quando a sonda falha, este é o melhor palpite do que
    a pessoa pode usar. Uma falha aqui não pode derrubar a iteração, e por isso o
    chamador trata a exceção como "(não foi possível listar)".

    Returns:
        set[str]: Nomes sem a tag, por exemplo `{"llama3", "gemma4"}`.
    """
    import urllib.request

    with urllib.request.urlopen(f"{BASE_URL_OLLAMA}/api/tags", timeout=10) as r:
        return {m["name"].split(":")[0] for m in json.load(r)["models"]}


def carregar_casos(caminho: Path = ARQUIVO_CASOS) -> list[dict[str, Any]]:
    """Lê os casos de avaliação de `evals/rag_cases.json`.

    Args:
        caminho: Arquivo com a lista de casos.

    Returns:
        list[dict]: Os casos, na ordem do arquivo.

    Raises:
        SystemExit: Se o arquivo não existir ou não for uma lista.
    """
    if not caminho.exists():
        raise SystemExit(f"✗ Arquivo de casos não encontrado: {caminho}")

    casos = json.loads(caminho.read_text(encoding="utf-8"))

    if not isinstance(casos, list):
        raise SystemExit(f"✗ {caminho} deveria conter uma lista de casos.")

    return casos


def executar_caso(
    caso: dict[str, Any],
    modelo_rag: str | None = None,
) -> dict[str, Any]:
    """Roda um caso pelo pipeline real e captura resposta, contextos e fontes.

    Chama `src.rag.pipeline.responder()` — o mesmo caminho da interface — em vez
    de `build_rag_chain()` direto, para que a avaliação cubra também o
    comportamento dos guardrails.

    `responder()` chama `build_rag_chain()` sem argumento, e o padrão de
    `src/rag/prompt_rag.py` é `gpt-oss:120b`, que roda na Ollama Cloud e exige
    `OLLAMA_API_KEY`. Quando `modelo_rag` é informado, a chain é envolvida para
    fixar o modelo — sem isso, a avaliação só funcionaria com a credencial da
    nuvem, o que tornaria a iteração dependente de um serviço externo. Com um
    modelo local, a mesma iteração roda inteira no Ollama da máquina.

    Args:
        caso: Um item de `rag_cases.json`.
        modelo_rag: Modelo de chat para a chain RAG, ou `None` para o padrão do
            projeto.

    Returns:
        dict: `pergunta`, `resposta`, `contextos` (lista de strings), `fontes`,
        `bloqueado` e, quando aplicável, o erro que impediu a resposta.
    """
    from contextlib import contextmanager

    from src.rag.pipeline import responder
    from src.rag.retriever import buscar_com_scores

    @contextmanager
    def _modelo_fixo():
        """Fixa o modelo da chain RAG durante a chamada, se informado."""
        if modelo_rag is None:
            yield
            return

        import src.rag.prompt_rag as prompt_rag

        original = prompt_rag.build_rag_chain
        prompt_rag.build_rag_chain = lambda model=modelo_rag: original(modelo_rag)

        try:
            yield
        finally:
            prompt_rag.build_rag_chain = original

    pergunta = caso["input"]

    # A busca é refeita aqui para obter os chunks crus. `responder()` devolve as
    # fontes com metadados e similaridade, mas não o texto de cada trecho — e é o
    # texto que o RAGAS compara com a resposta.
    try:
        trechos = buscar_com_scores(pergunta)
        contextos = [
            f"[{doc.metadata.get('source', 'fonte_desconhecida')}, "
            f"p. {doc.metadata.get('page', 'n/d')}]\n{doc.page_content.strip()}"
            for doc, _ in trechos
        ]
    except Exception as erro:  # noqa: BLE001
        contextos = []
        print(f"   ⚠️  busca falhou para {caso.get('id')}: {erro}")

    with _modelo_fixo():
        resultado = responder(pergunta)

    # `responder()` devolve `bloqueado=True` tanto para uma recusa de guardrail
    # quanto para uma falha de infraestrutura (Ollama fora do ar, base ausente).
    # São fatos diferentes com consequências opostas — um é o grounding
    # funcionando, o outro é a avaliação inteira inválida —, então a distinção
    # precisa ser feita aqui. O texto de erro começa com uma frase fixa de
    # `RESPOSTA_ERRO` em `src/rag/pipeline.py`, que é a forma barata de
    # reconhecê-lo sem mudar a assinatura pública do pipeline.
    erro_infra = resultado["resposta"].startswith(
        "Não consegui consultar a base de conhecimento"
    )

    registro: dict[str, Any] = {
        "id": caso.get("id"),
        # `answerable` como padrão preserva a compatibilidade com arquivos de
        # casos anteriores ao split: sem `tipo`, o comportamento antigo (ir para
        # o RAGAS) é o menos enganoso, em vez de o caso ser silenciosamente
        # descartado da avaliação.
        "tipo": caso.get("tipo", "answerable"),
        "pergunta": pergunta,
        "expected_sources": caso.get("expected_sources", []),
        "resposta": resultado["resposta"],
        "contextos": contextos,
        "fontes": resultado["fontes"],
        "bloqueado": resultado["bloqueado"],
        # Recusa de guardrail: o grounding funcionou. Erro de infraestrutura: a
        # resposta não existe e a iteração está comprometida.
        "recusa_guardrail": resultado["bloqueado"] and not erro_infra,
        "erro_infraestrutura": erro_infra,
    }

    if erro_infra:
        registro["erro_pipeline"] = True

    return registro


def avaliar_com_ragas(
    registros: list[dict[str, Any]],
    modelo_judge: str,
) -> None:
    """Calcula as duas métricas do RAGAS e escreve as notas em cada registro.

    Recebe os registros por referência e os muta com `faithfulness` e
    `answer_relevancy`. Registros bloqueados ficam com as notas em `None` e não
    são enviados ao RAGAS: quem respondeu foi um guardrail local, sem passar pelo
    contexto.

    Uma falha em um caso não interrompe a avaliação — o judge é uma chamada de
    rede e uma delas não deve custar as outras.

    Args:
        registros: Saída de `executar_caso()`, mutada no lugar.
        modelo_judge: Nome do modelo de judge no Ollama.
    """
    _verificar_modelo_disponivel(modelo_judge)

    _instalar_shim_ragas()

    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    llm, embeddings = _construir_judge(modelo_judge)
    fidelidade = Faithfulness(llm=llm)
    relevancia = AnswerRelevancy(llm=llm, embeddings=embeddings)

    for registro in registros:
        registro["faithfulness"] = None
        registro["answer_relevancy"] = None

        if registro["erro_infraestrutura"]:
            print(
                f"   ⚠️ {registro['id']}: erro de infraestrutura, caso não avaliado"
            )
            continue

        # Casos `refusal_expected` não vão ao RAGAS. A `answer_relevancy` mede
        # similaridade entre pergunta e resposta, e penaliza a recusa correta —
        # ver o bloco "OS DOIS BLOCOS DE AVALIAÇÃO" no topo deste arquivo. Eles
        # são pontuados por `avaliar_recusas()`, que conta, e não por similaridade.
        if registro.get("tipo") == "refusal_expected":
            print(
                f"   ⏭ {registro['id']}: refusal_expected, "
                f"fora do RAGAS (rubrica de recusa)"
            )
            continue

        if registro["bloqueado"]:
            print(f"   ⏭ {registro['id']}: recusado por guardrail, não avaliado")
            continue

        try:
            nota_f = fidelidade.score(
                user_input=registro["pergunta"],
                response=registro["resposta"],
                retrieved_contexts=registro["contextos"],
            )
            registro["faithfulness"] = nota_f.value

            if nota_f.reason:
                registro["faithfulness_reason"] = str(nota_f.reason)
        except Exception as erro:  # noqa: BLE001
            registro["erro_faithfulness"] = f"{type(erro).__name__}: {erro}"
            print(f"   ⚠️  faithfulness falhou em {registro['id']}: {erro}")

        try:
            nota_r = relevancia.score(
                user_input=registro["pergunta"],
                response=registro["resposta"],
            )
            registro["answer_relevancy"] = nota_r.value

            if nota_r.reason:
                registro["answer_relevancy_reason"] = str(nota_r.reason)
        except Exception as erro:  # noqa: BLE001
            registro["erro_answer_relevancy"] = f"{type(erro).__name__}: {erro}"
            print(f"   ⚠️  answer_relevancy falhou em {registro['id']}: {erro}")

        print(
            f"   ✓ {registro['id']}: "
            f"faithfulness={_formatar_nota(registro['faithfulness'])} "
            f"relevancy={_formatar_nota(registro['answer_relevancy'])}"
        )


def _formatar_nota(nota: float | None) -> str:
    """Formata uma nota para exibição, sem quebrar quando ela é `None`.

    Args:
        nota: Valor entre 0 e 1, ou `None` quando a métrica não rodou.

    Returns:
        str: A nota com duas casas, ou `"n/d"`.
    """
    return "n/d" if nota is None else f"{nota:.2f}"


def _instalar_shim_ragas() -> None:
    """Permite `import ragas` sem alterar nenhuma dependência instalada.

    O RAGAS 0.4.3 importa `ChatVertexAI` de um caminho que não existe mais no
    `langchain-community` 0.4.2, e a falha no `import` derruba todas as métricas.
    O símbolo é usado apenas em `isinstance`, para classificar o provider, e nunca
    é instanciado quando o judge é o Ollama.

    Registra um módulo sintético em `sys.modules` sob o nome que o RAGAS espera.
    O adendo vale só para este processo e some com ele: nenhum arquivo do
    ambiente é tocado.

    Returns:
        None: Apenas prepara o `sys.modules` para o import do RAGAS.
    """
    nome = "langchain_community.chat_models.vertexai"

    if nome in sys.modules:
        return

    modulo = types.ModuleType(nome)

    class ChatVertexAI:  # noqa: D401 - substituto apenas para isinstance
        """Substituto de `langchain_community.chat_models.vertexai`.

        Existe só para satisfazer o `from ... import ChatVertexAI` do RAGAS.
        Se algum dia for instanciado, é sinal de que o shim saiu de validade e o
        RAGAS passou a usar esse caminho de verdade — por isso a falha é explícita
        em vez de silenciosa.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "ChatVertexAI substituto foi instanciado. O RAGAS passou a usar "
                "de verdade o caminho legado do langchain-community; este shim "
                "deve ser removido. Remova-o em _instalar_shim_ragas()."
            )

    modulo.ChatVertexAI = ChatVertexAI
    sys.modules[nome] = modulo


# Rótulo dos campos de saída, no mesmo nome das métricas do RAGAS, para que o
# JSON possa ser lido por qualquer consumidor sem dicionário de tradução.
METRICAS = ("faithfulness", "answer_relevancy")

# Instalar aqui, e não dentro de `_construir_judge()`, porque o shim precisa
# estar pronto antes de QUALQUER import do RAGAS — inclusive o de
# `ragas.metrics.collections`, que dispara `import ragas` e falha sem ele. Uma
# chamada que o instalasse tarde demais reprovaria o mesmo `ModuleNotFoundError`
# que o shim existe para evitar.
_instalar_shim_ragas()


def _ler_nota(rotulo: str) -> float | None:
    """Lê uma nota da rubrica manual no terminal.

    Aceita `1`, `0.5` e `0`, que são os três valores previstos em
    `evals/rubrica_manual.md`. `q` cancela a avaliação.

    Args:
        rotulo: Nome da dimensão exibido no prompt.

    Returns:
        float | None: A nota, ou `None` se o avaliador pediu para pular.
    """
    while True:
        try:
            entrada = input(f"   {rotulo} [1, 0.5, 0 — 'q' pula]: ").strip().lower()

            if entrada in {"q", "pular"}:
                return None

            nota = float(entrada)

            if nota not in {0.0, 0.5, 1.0}:
                print("   ✗ Use apenas 1, 0.5 ou 0 (ver evals/rubrica_manual.md).")
                continue

            return nota

        except ValueError:
            print("   ✗ Entrada inválida. Digite 1, 0.5 ou 0.")


def _ler_justificativa() -> str:
    """Lê a justificativa textual obrigatória para cada nota.

    A rubrica exige justificativa: uma nota sem justificativa não pode ser
    revisada depois, e é a justificativa que permite reconciliar dois
    avaliadores na mesma iteração.

    Returns:
        str: O texto digitado, ou string vazia se o avaliador pular.
    """
    texto = input("   Justificativa: ").strip()
    return "" if texto.lower() in {"q", "pular"} else texto


def avaliar_manualmente(registros: list[dict[str, Any]]) -> None:
    """Aplica a rubrica 0-1 documentada em `evals/rubrica_manual.md`.

    Fallback usado com `--manual`, e também quando o RAGAS não está disponível.
    Para cada caso, exibe pergunta, contexto recuperado e resposta, e pede as
    notas de *faithfulness* e *answer_relevancy* com justificativa.

    A rubrica é binária de meio ponto (1 / 0.5 / 0), e não a escala decimal de
    0.1 usada por `evals/ragas_eval.py`: aqui o avaliador humano é a fonte, e
    menos pontos de ancoragem deixam a decisão mais reprodutível entre
    avaliadores.

    Casos bloqueados não são avaliados, pelos mesmos motivos do caminho RAGAS.

    Args:
        registros: Saída de `executar_caso()`, mutada no lugar.
    """
    print("=" * 78)
    print("  AVALIAÇÃO MANUAL — rubrica 0-1 (fallback)")
    print(f"  Critérios completos em: {ARQUIVO_RUBRICA}")
    print("=" * 78)

    for registro in registros:
        print("\n" + "=" * 78)
        print(f"CASO: {registro['id']}")

        # Casos `refusal_expected` não passam pela rubrica 0-1: a pergunta "isto
        # responde?" não se aplica a eles, e as notas de fidelidade/relevância
        #ariam penalizar a recusa correta. A rubrica deles é binária
        # (recusou/não recusou) e é aplicada por `avaliar_recusas()`.
        if registro.get("tipo") == "refusal_expected":
            print("\n🚫 CASO refusal_expected — fora da rubrica 0-1.")
            print("   Aqui o comportamento correto é RECUSAR. A nota é binária")
            print("   (1 = recusou, 0 = respondeu) e sai de avaliar_recusas().")
            print(f"\n   Resposta do modelo:\n{registro['resposta'][:600]}")

            registro["faithfulness"] = None
            registro["answer_relevancy"] = None
            continue

        if registro["bloqueado"]:
            if registro["erro_infraestrutura"]:
                print("\n⚠️ ERRO DE INFRAESTRUTURA — caso não avaliado.")
                print("   O pipeline falhou antes de gerar resposta:")
                print(f"   {registro['resposta'][:200]}")
                print("\n   Verifique o Ollama, OLLAMA_API_KEY e a base ingerida.")
            else:
                print("\n🚫 RECUSA DE GUARDRAIL — caso não avaliado.")
                print("   A resposta veio de um filtro local, antes da busca. Não há")
                print("   resposta do modelo para comparar com o contexto.")

            registro["faithfulness"] = None
            registro["answer_relevancy"] = None
            continue

        print("\n[1] PERGUNTA")
        print("-" * 78)
        print(registro["pergunta"])

        print("\n[2] CONTEXTO RECUPERADO")
        print("-" * 78)
        print("\n\n---\n\n".join(registro["contextos"]) or "(vazio)")

        print("\n[3] RESPOSTA DO MODELO")
        print("-" * 78)
        print(registro["resposta"])

        print("\n[4] NOTAS — rubrica 0-1")
        print("-" * 78)
        print("  FAITHFULNESS (resposta sustentada pelo contexto?)")
        print("     1   = totalmente sustentada pelo contexto")
        print("     0.5 = parcialmente sustentada")
        print("     0   = inventada (fato que não está no contexto)")
        print("  ANSWER RELEVANCY (resposta endereça a pergunta?)")
        print("     1   = endereça diretamente")
        print("     0.5 = endereça parcialmente")
        print("     0   = não endereça")
        print()

        try:
            fidelidade = _ler_nota("Faithfulness    ")
            justificativa_f = _ler_justificativa() if fidelidade is not None else ""

            relevancia = _ler_nota("Answer Relevancy")
            justificativa_r = _ler_justificativa() if relevancia is not None else ""
        except (KeyboardInterrupt, EOFError):
            print("\n\n✗ Avaliação interrompida. O que foi lido até aqui será salvo.")
            break

        registro["faithfulness"] = fidelidade
        registro["answer_relevancy"] = relevancia

        if justificativa_f:
            registro["faithfulness_justificativa"] = justificativa_f
        if justificativa_r:
            registro["answer_relevancy_justificativa"] = justificativa_r


def resumir(registros: list[dict[str, Any]], metodo: str) -> dict[str, Any]:
    """Monta o resumo da iteração, separado nos dois blocos de avaliação.

    Devolve duas médias que **não** podem ser somadas:

    - `answerable`: faithfulness e answer_relevancy do RAGAS, sobre os casos em
      que a base responde.
    - `refusal_expected`: taxa de recusa correta, por contagem direta (recusou
      ou não), sobre os casos em que a base não responde.

    Misturar os dois numa média só era possível antes do split, e o resultado era
    um número sem interpretação: "0,60 de fidelidade" não diz se o modelo
    inventa, recusa demais ou recusa demais pouco. As contagens de cada bloco
    vêm junto, para que a média nunca seja lida sem saber de quantos casos veio.

    Casos bloqueados por guardrail e erros de infraestrutura ficam fora das médias
    de ambos os blocos, e são contados à parte: um é o grounding funcionando, o
    outro é a avaliação inválida.

    Args:
        registros: Registros já avaliados.
        metodo: `ragas` ou `rubrica_manual`, gravado no resumo.

    Returns:
        dict: Os dois blocos, as contagens e o total.
    """
    resumo: dict[str, Any] = {
        "metodo": metodo,
        "total_casos": len(registros),
        "recusados_guardrail": sum(1 for r in registros if r.get("recusa_guardrail")),
        "erros_infraestrutura": sum(
            1 for r in registros if r.get("erro_infraestrutura")
        ),
        "avaliados": 0,
    }

    resumo["answerable"] = _resumo_bloco(
        [r for r in registros if r.get("tipo") != "refusal_expected"],
        METRICAS,
    )

    resumo["refusal_expected"] = _resumo_bloco(
        [r for r in registros if r.get("tipo") == "refusal_expected"],
        ("recusa_nota",),
    )

    # A "recusa_nota" é binária: 1 quando recusou, 0 quando respondeu. A média é
    # a taxa de recusa correta, e a contagem diz se ela vale alguma coisa — com
    # um único caso `refusal_expected`, 1,00 ou 0,00 é um evento, não uma taxa.
    bloco_recusa = resumo["refusal_expected"]
    bloco_recusa["taxa_recusa_correta"] = bloco_recusa.get("recusa_nota_media")
    bloco_recusa["metodo"] = "rubrica_recusa_binaria"

    bloco = resumo["answerable"]
    bloco["metodo"] = metodo
    resumo["avaliados"] = min(
        (bloco.get(f"{m}_n", 0) for m in METRICAS),
        default=0,
    )

    return resumo


def _resumo_bloco(registros: list[dict[str, Any]], metricas: tuple[str, ...]) -> dict:
    """Média e contagem de cada métrica em um conjunto de registros.

    A média usa só os registros com nota: bloqueados e erros ficam fora, e a
    contagem (`*_n`) diz o tamanho real da amostra.

    Args:
        registros: Registros do bloco.
        metricas: Nomes dos campos de nota a resumir.

    Returns:
        dict: Um campo `<metrica>_media` e um `<metrica>_n` por métrica, mais o
        total de casos do bloco.
    """
    bloco: dict[str, Any] = {"total_casos": len(registros)}

    for metrica in metricas:
        notas = [r[metrica] for r in registros if r.get(metrica) is not None]
        bloco[f"{metrica}_media"] = round(mean(notas), 4) if notas else None
        bloco[f"{metrica}_n"] = len(notas)

    bloco["avaliados"] = min(
        (bloco.get(f"{m}_n", 0) for m in metricas),
        default=0,
    )

    return bloco


def salvar(
    registros: list[dict[str, Any]],
    saida: Path,
    metodo: str,
    modelo_judge: str | None,
    modelo_rag: str | None = None,
    versao_prompt: str | None = None,
) -> Path:
    """Grava a iteração em `saida`, com resumo, casos e metadados.

    A escrita vai para um arquivo temporário e é renomeada no fim: se o processo
    for interrompido no meio, o `iteration_N.json` anterior continua íntegro em
    vez de ficar truncado.

    Os dois modelos usados ficam registrados porque uma nota não é comparável com
    outra sem saber o que as produziu: o modelo da resposta e o judge da nota.
    A versão do prompt entra pelo mesmo motivo — é ela que diz de qual iteração
    de `prompts/` a nota é resultado.

    `metodologia` e `blocos` são gravados junto porque o arquivo precisa dizer
    sozinho como foi dividido: um `iteration_N.json` antigo, sem `tipo` nos
    casos, tem média em `resumo.faithfulness_media` na raiz, e o novo tem os
    dois blocos. Sem o marcador, comparar os dois daria leitura errada.

    Args:
        registros: Casos com as notas.
        saida: Caminho de destino.
        metodo: `ragas` ou `rubrica_manual`.
        modelo_judge: Modelo usado como judge, quando aplicável.
        modelo_rag: Modelo de chat da chain RAG, quando informado.
        versao_prompt: Versão do prompt em uso, quando conhecida.

    Returns:
        Path: O caminho gravado.
    """
    payload = {
        "iteracao": saida.stem.replace("iteration_", ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metodo": metodo,
        "metodologia": {
            "split": "tipo=answerable -> RAGAS; tipo=refusal_expected -> rubrica de recusa",
            "rubrica_recusa": "1 = recusou (guardrail ou modelo), 0 = respondeu",
            "nota_recusa_media": (
                "taxa de recusa correta; não somar com as métricas do RAGAS"
            ),
            "ragas_exclui_refusal_expected": (
                "a answer_relevancy do RAGAS penaliza a recusa correta, por "
                "ser similaridade entre pergunta e resposta"
            ),
        },
        "prompt_rag_versao": versao_prompt or _versao_prompt_em_uso(),
        "modelo_rag": modelo_rag or "padrao do projeto (gpt-oss:120b)",
        "modelo_judge": modelo_judge,
        "embedding": MODELO_EMBEDDING_PADRAO,
        "resumo": resumir(registros, metodo),
        "casos": registros,
    }

    saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = saida.with_suffix(saida.suffix + ".tmp")
    temporario.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporario.replace(saida)

    return saida


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada: lê os casos, avalia e grava a iteração.

    Args:
        argv: Argumentos da linha de comando; `None` usa `sys.argv[1:]`.

    Returns:
        int: `0` em sucesso, `1` se nenhum caso foi avaliado.
    """
    parser = argparse.ArgumentParser(
        description="Avalia o pipeline RAG com RAGAS ou rubrica manual.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python evals/run_rag_eval.py\n"
            "  python evals/run_rag_eval.py --manual\n"
            "  python evals/run_rag_eval.py --output evals/iteration_2.json\n"
            "  python evals/run_rag_eval.py --judge-model llama3 --limit 3\n"
            "  python evals/run_rag_eval.py --rag-model llama3   # iteração offline\n"
        ),
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="usa a rubrica manual 0-1 em vez do RAGAS (ver evals/rubrica_manual.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAIZ_PROJETO / "evals" / "iteration_1.json",
        help="arquivo de saída (padrão: evals/iteration_1.json)",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ARQUIVO_CASOS,
        help="arquivo de casos (padrão: evals/rag_cases.json)",
    )
    parser.add_argument(
        "--judge-model",
        default=MODELO_JUDGE_PADRAO,
        help=f"modelo de judge no Ollama (padrão: {MODELO_JUDGE_PADRAO})",
    )
    parser.add_argument(
        "--rag-model",
        default=None,
        help=(
            "modelo de chat da chain RAG. Sem esta flag, usa o padrão do projeto "
            "(gpt-oss:120b, na Ollama Cloud, exige OLLAMA_API_KEY). Informe um "
            "modelo local, ex.: --rag-model llama3, para uma iteração 100% offline"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="avalia apenas os N primeiros casos",
    )
    parser.add_argument(
        "--nao-pular-ragas",
        action="store_true",
        help="não cai para a rubrica manual se o RAGAS falhar",
    )

    args = parser.parse_args(argv)

    _configurar_stdout()

    # Os avisos do RAGAS sobre APIs legadas poluem o relatório de uma iteração que
    # só precisa mostrar as notas; eles não indicam problema aqui.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*will be removed in v1.0.*")

    casos = carregar_casos(args.cases)

    if args.limit is not None:
        casos = casos[: args.limit]

    if not casos:
        print("✗ Nenhum caso para avaliar.")
        return 1

    # Lida uma vez e exibida no cabeçalho: a versão do prompt é a informação que
    # distingue esta iteração da anterior, e descobrir qual era só depois de
    # aberto o JSON é tarde demais para corrigir uma rodada trocada.
    versao_prompt = _versao_prompt_em_uso()

    print("=" * 78)
    print(f"  AVALIAÇÃO DO PIPELINE RAG — {len(casos)} caso(s)")
    print(f"  Prompt RAG: v{versao_prompt}")
    print("=" * 78)

    registros: list[dict[str, Any]] = []

    for i, caso in enumerate(casos, 1):
        print(f"\n[{i}/{len(casos)}] {caso.get('id')}")
        print(f"   {caso['input']}")

        try:
            registros.append(executar_caso(caso, modelo_rag=args.rag_model))
        except Exception as erro:  # noqa: BLE001
            # Um caso que derruba o pipeline inteiro esconde os demais: melhor
            # registrar a falha e seguir, para a iteração cobrir tudo que deu.
            print(f"   ⚠️  caso falhou: {type(erro).__name__}: {erro}")
            registros.append(
                {
                    "id": caso.get("id"),
                    "pergunta": caso["input"],
                    "expected_sources": caso.get("expected_sources", []),
                    "resposta": "",
                    "contextos": [],
                    "fontes": [],
                    "bloqueado": True,
                    "recusa_guardrail": False,
                    "erro_infraestrutura": True,
                    "erro_pipeline": f"{type(erro).__name__}: {erro}",
                    "faithfulness": None,
                    "answer_relevancy": None,
                }
            )

    metodo = "rubrica_manual"

    if not args.manual:
        try:
            print("\n--- RAGAS (Ollama como judge) ---")
            avaliar_com_ragas(registros, args.judge_model)
            metodo = "ragas"
        except SystemExit as erro:
            print(f"\n{erro}")

            if args.nao_pular_ragas:
                return 1

            print(
                "\n→ Caindo para a rubrica manual. Use --judge-model <nome> com um "
                "modelo\n  já baixado, ou `ollama pull <modelo>`."
            )
            avaliar_manualmente(registros)
        except Exception as erro:  # noqa: BLE001
            print(f"\n⚠️  RAGAS indisponível: {type(erro).__name__}: {erro}")

            if args.nao_pular_ragas:
                return 1

            print("\n→ Caindo para a rubrica manual.")
            avaliar_manualmente(registros)
    else:
        avaliar_manualmente(registros)

    # A rubrica de recusa é automática e independente do método: ela conta se o
    # modelo recusou, e não depende de judge nem de avaliador humano. Roda nos
    # dois caminhos para que o bloco `refusal_expected` exista mesmo numa
    # iteração avaliada pelo RAGAS.
    avaliar_recusas(registros)

    caminho = salvar(
        registros,
        args.output,
        metodo,
        args.judge_model,
        args.rag_model,
        versao_prompt,
    )

    resumo = resumir(registros, metodo)
    bloco = resumo["answerable"]
    recusa = resumo["refusal_expected"]

    print("\n" + "=" * 78)
    print("  RESUMO")
    print("=" * 78)
    print(f"  Casos:            {resumo['total_casos']}")
    print(f"  Recusados:        {resumo['recusados_guardrail']} (guardrail, fora da média)")
    print(f"  Erros de infra:   {resumo['erros_infraestrutura']} (fora da média)")
    print("")
    print("  BLOCO 1 — answerable (RAGAS)")
    print(f"    Casos:          {bloco['total_casos']}")
    print(f"    Faithfulness:   {_formatar_nota(bloco['faithfulness_media'])}"
          f"  (n={bloco['faithfulness_n']})")
    print(f"    Answer relev.:  {_formatar_nota(bloco['answer_relevancy_media'])}"
          f"  (n={bloco['answer_relevancy_n']})")
    print("")
    print("  BLOCO 2 — refusal_expected (rubrica binária)")
    print(f"    Casos:          {recusa['total_casos']}")
    print(f"    Recusa correta: {_formatar_nota(recusa['taxa_recusa_correta'])}"
          f"  (n={recusa['recusa_nota_n']})")
    print("")
    print("  As médias dos dois blocos não são somáveis.")
    print(f"\n✓ Salvo em: {caminho}")

    if bloco["faithfulness_n"] == 0 and recusa["recusa_nota_n"] == 0:
        print("\n⚠️  Nenhum caso avaliado — o JSON registra os motivos.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())