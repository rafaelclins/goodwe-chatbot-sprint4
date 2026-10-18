"""Gera os 3 PDFs adicionais da base de conhecimento (Sprint 4).

Cria, em ``data/knowledge_base/``:

* ``faq_carregamento_condominio.pdf``
* ``tabela_tarifaria.pdf``
* ``manual_chargegrid_resumo.pdf``

Usa apenas ``reportlab`` (ja em ``requirements.txt``) para montar os PDFs e
``pymupdf`` (ja usado pelo loader) apenas para anexar um comentario (annotation)
no topo de cada arquivo, com as fontes reais de onde os valores foram tirados.

Os valores sao ilustrativos/didaticos e refletem a estrutura publica de tarifacao
da ANEEL (Tarifa Branca) e as especificacoes publicas de carregadores AC GoodWe.
"""

from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path("data/knowledge_base")

FONTE_FAQ = (
    "Fontes (valores e referencias): ANEEL - Resolucao Normativa n. 1.000/2021 "
    "(prestacao do servico de distribuicao); ANEEL - Tarifa Branca (RN 733/2016); "
    "ABNT NBR 5410 (instalacoes eletricas de baixa tensao); ABNT NBR 17019:2022 "
    "(instalacoes eletricas para veiculos eletricos); IEC 61851-1 e IEC 62196-2 "
    "(sistema de recarga condutiva e conector Tipo 2); datasheet publico do "
    "carregador AC GoodWe. Documento didatico."
)

FONTE_TARIFA = (
    "Fontes (valores e referencias): ANEEL - Tarifa Branca e estrutura de postos "
    "horarios (RN 733/2016, Nota Tecnica n. 137/2015); ANEEL - Resolucao "
    "Normativa n. 1.000/2021. Tarifas em R$/kWh ilustrativas, sem tributos "
    "estaduais; a tarifa efetiva varia por distribuidora e bandeira tarifaria. "
    "Documento didatico."
)

FONTE_CHARGEGRID = (
    "Fontes (valores e referencias): especificacoes publicas do carregador AC "
    "GoodWe (modelos HCA/GW, 7,4/11/22 kW, IP65, DR Tipo A integrado); IEC "
    "61851-1 (modos de recarga 3); IEC 62196-2 (conector Tipo 2); ABNT NBR "
    "17019:2022. Valores publicos de datasheet. Documento didatico."
)

ES = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=ES["Heading1"], fontSize=16, spaceAfter=10, textColor=colors.HexColor("#1f3864")
)
H2 = ParagraphStyle(
    "H2", parent=ES["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2e5496")
)
CORPO = ParagraphStyle("Corpo", parent=ES["BodyText"], fontSize=10, leading=14, spaceAfter=5)
FONTE = ParagraphStyle(
    "Fonte", parent=ES["BodyText"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#666666"), spaceAfter=10
)
PERGUNTA = ParagraphStyle(
    "Pergunta", parent=ES["BodyText"], fontSize=10, leading=14, spaceBefore=4, spaceAfter=2,
    textColor=colors.HexColor("#1f3864"), fontName="Helvetica-Bold",
)
NOTA = ParagraphStyle(
    "Nota", parent=ES["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#555555"), spaceAfter=6
)


def _fonte_flowable(texto: str) -> Paragraph:
    return Paragraph(f"<b>Comentario de origem (fontes):</b> {texto}", FONTE)


def _build(path: Path, titulo: str, fonte: str, flowables: list) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=titulo,
        author="Challenge FIAP + GoodWe 2026 - Grupo 1CCPI",
        subject=titulo,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    historia = [Paragraph(titulo, H1), _fonte_flowable(fonte), Spacer(1, 6)] + flowables
    doc.build(historia)

    with pymupdf.open(path) as pdf:
        pdf[0].add_text_annot((2 * cm, 1.0 * cm), f"Comentario de origem: {fonte}", icon="Comment")
        pdf.set_metadata({"title": titulo, "subject": fonte, "author": "Challenge FIAP + GoodWe 2026 - Grupo 1CCPI"})
        pdf.saveIncr()


def gerar_faq() -> None:
    itens = [
        ("1. Quais sao os horarios permitidos para carregar meu veiculo?",
         "A recarga pode ocorrer a qualquer momento, mas o condominio recomenda o periodo "
         "fora de ponta, entre 23h e 15h em dias uteis, quando a energia e mais barata. "
         "Em horario de ponta (18h as 21h, dias uteis) o uso simultaneo de varios "
         "carregadores deve ser evitado para nao elevar a demanda contratada do edificio."),
        ("2. Qual e a potencia maxima por carregador?",
         "Carregadores AC monofasicos operam tipicamente em 7,4 kW; trifasicos, em 11 kW "
         "ou 22 kW. A potencia liberada por ponto depende da capacidade instalada do "
         "condominio e do projeto eletrico aprovado."),
        ("3. Preciso de um projeto eletrico para instalar o carregador?",
         "Sim. A instalacao exige projeto eletrico assinado por engenheiro eletricista "
         "registrado no CREA, dimensionando disjuntor, cabo, aterramento e protecao "
         "diferencial residual (DR). A instalacao segue a ABNT NBR 5410 e a ABNT "
         "NBR 17019."),
        ("4. Que tipo de cabo e conector e usado?",
         "O padrao adotado e o conector Tipo 2 (IEC 62196-2), com cabo de recarga "
         "dedicado. A secao do cabo e definida no projeto em funcao da distancia e da "
         "corrente; para 7,4 kW costuma-se usar cabo de 6 mm2 a 10 mm2."),
        ("5. Quanto tempo leva para carregar?",
         "O tempo e o quociente entre a energia necessaria e a potencia do carregador: "
         "Tempo (h) = Energia (kWh) / Potencia (kW). Por exemplo, 60 kWh levam cerca de "
         "8,1 h a 7,4 kW, 5,5 h a 11 kW e 2,7 h a 22 kW."),
        ("6. Posso usar uma tomada comum com carregador portatil?",
         "Nao. O uso de carregador portatil em tomada comum e proibido pelas regras do "
         "condominio, por sobrecarregar o circuito e nao dispor da protecao exigida. "
         "Use apenas a estacao de recarga oficial homologada."),
        ("7. Qual protecao eletrica e exigida?",
         "E obrigatorio disjuntor DR Tipo A de 30 mA contra fuga de corrente residual, "
         "alem de disjuntor termomagnetico e aterramento conforme o projeto. Varios "
         "carregadores ja integram o DR Tipo A internamente."),
        ("8. Como o consumo e medido e cobrado?",
         "Cada carregador possui medidor MID que registra o consumo em kWh. O valor e "
         "cobrado individualmente ao morador, fora da taxa condominial, pelo consumo "
         "medido multiplicado pela tarifa (ver regimento e tabela tarifaria)."),
        ("9. O que fazer em caso de falha no carregador?",
         "Interrompa o uso, desligue o disjuntor do ponto e acione a administracao ou a "
         "assistencia tecnica. Nao abra o equipamento: a manutencao deve ser feita por "
         "profissional habilitado."),
        ("10. Quem pode instalar ou alterar o ponto de recarga?",
         "Somente eletricista ou instalador habilitado, sob o projeto aprovado e com "
         "autorizacao previa da administracao. Alteracoes sem projeto podem "
         "comprometer a protecao coletiva e a apolice de seguro do condominio."),
    ]

    story: list = []
    for pergunta, resposta in itens:
        story.append(Paragraph(pergunta, PERGUNTA))
        story.append(Paragraph(resposta, CORPO))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Observacao: este FAQ e um resumo didatico. Em caso de conflito, prevalecem o "
        "regimento do condominio, o projeto eletrico aprovado e as normas da "
        "distribuidora local.", NOTA))

    _build(BASE_DIR / "faq_carregamento_condominio.pdf",
           "FAQ - Carregamento de Veiculos Eletricos em Condominio", FONTE_FAQ, story)


def gerar_tarifaria() -> None:
    story: list = []

    story.append(Paragraph("1. Tarifas por posto horario (R$/kWh)", H2))
    dados = [
        ["Posto horario", "Horario (dias uteis)", "Tarifa (R$/kWh)"],
        ["Ponta", "18h as 21h", "1,10"],
        ["Cheio (intermediario)", "16h as 17h e 21h as 22h", "0,85"],
        ["Vazio (fora de ponta)", "23h as 15h", "0,55"],
    ]
    tabela = Table(dados, colWidths=[4.2 * cm, 7.2 * cm, 4.2 * cm])
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f9")]),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tabela)
    story.append(Paragraph(
        "Sabados, domingos e feriados nacionais sao tratados, integralmente, como "
        "posto vazio. Os valores sao ilustrativos e nao incluem tributos estaduais "
        "(ICMS) nem bandeiras tarifarias.", NOTA))

    story.append(Paragraph("2. Formula do custo de recarga", H2))
    story.append(Paragraph(
        "<b>Custo (R$) = Energia consumida (kWh) x Tarifa (R$/kWh)</b>", CORPO))
    story.append(Paragraph(
        "A energia consumida pode ser estimada por "
        "<b>Energia (kWh) = Potencia (kW) x Tempo de recarga (h)</b> ou lida "
        "diretamente no medidor MID do carregador.", CORPO))

    story.append(Paragraph("3. Exemplos de calculo por energia consumida", H2))
    exemplo = [
        ["Energia", "Vazio (R$ 0,55)", "Cheio (R$ 0,85)", "Ponta (R$ 1,10)"],
        ["30 kWh", "R$ 16,50", "R$ 25,50", "R$ 33,00"],
        ["40 kWh", "R$ 22,00", "R$ 34,00", "R$ 44,00"],
        ["60 kWh", "R$ 33,00", "R$ 51,00", "R$ 66,00"],
    ]
    tab2 = Table(exemplo, colWidths=[3.0 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm])
    tab2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e5496")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f9")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tab2)
    story.append(Paragraph(
        "Exemplo resolvido: recarregar 60 kWh no posto vazio custa "
        "60 x 0,55 = <b>R$ 33,00</b>. O condominio pode optar por repassar a tarifa "
        "por posto horario ou aplicar uma tarifa unica de rateio (ver regimento do "
        "condominio, secao de Rateio de Custos).", CORPO))

    _build(BASE_DIR / "tabela_tarifaria.pdf",
           "Tabela Tarifaria - Recarga de Veiculos Eletricos (R$/kWh)", FONTE_TARIFA, story)


def gerar_chargegrid() -> None:
    story: list = []

    story.append(Paragraph(
        "Documento didatico. Este e um <b>resumo generico de especificacoes</b> de um "
        "carregador AC de referencia (\"ChargeGrid\"), preparado para fins academicos, "
        "a partir de dados publicos de datasheet de carregadores AC GoodWe.", NOTA))

    story.append(Paragraph("1. Potencias disponiveis", H2))
    spec = [
        ["Modelo", "Potencia", "Fases", "Corrente nominal"],
        ["ChargeGrid 7,4", "7,4 kW", "Monofasico (230 V)", "32 A"],
        ["ChargeGrid 11", "11 kW", "Trifasico (380/400 V)", "16 A por fase"],
        ["ChargeGrid 22", "22 kW", "Trifasico (380/400 V)", "32 A por fase"],
    ]
    tab = Table(spec, colWidths=[3.6 * cm, 2.8 * cm, 4.6 * cm, 4.6 * cm])
    tab.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f9")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tab)

    story.append(Paragraph("2. Protecao e seguranca eletrica", H2))
    story.append(ListFlowable([
        ListItem(Paragraph("Protecao diferencial residual (DR) Tipo A integrada, com "
                           "limiar de 30 mA contra fuga de corrente residual.", CORPO)),
        ListItem(Paragraph("Deteccao de falta de aterramento e protecao contra surtos.", CORPO)),
        ListItem(Paragraph("Monitoramento de temperatura interna e desligamento automatico "
                           "em sobrecarga ou sobretensao.", CORPO)),
        ListItem(Paragraph("Grau de protecao IP65: resistencia a po e a jatos de agua.", CORPO)),
    ], bulletType="bullet"))

    story.append(Paragraph("3. Conectores e modos de recarga", H2))
    story.append(ListFlowable([
        ListItem(Paragraph("Conector Tipo 2 (IEC 62196-2), com trava eletrica e cabo "
                           "dedicado de 5 m.", CORPO)),
        ListItem(Paragraph("Recarga no Modo 3 (IEC 61851-1): comunicacao entre veiculo e "
                           "carregador pelo proprio cabo.", CORPO)),
        ListItem(Paragraph("Medidor MID integrado, que registra o consumo exato em kWh "
                           "para fins de rateio.", CORPO)),
    ], bulletType="bullet"))

    story.append(Paragraph("4. Condicoes de operacao", H2))
    story.append(ListFlowable([
        ListItem(Paragraph("Temperatura de operacao: -30 C a +50 C.", CORPO)),
        ListItem(Paragraph("Temperatura de armazenamento: -40 C a +70 C.", CORPO)),
        ListItem(Paragraph("Umidade relativa: 5% a 95%, sem condensacao.", CORPO)),
        ListItem(Paragraph("Altitude maxima de operacao: 2.000 m.", CORPO)),
    ], bulletType="bullet"))

    story.append(Paragraph("5. Tempo de carga estimado", H2))
    tempo = [
        ["Energia", "7,4 kW", "11 kW", "22 kW"],
        ["30 kWh", "4,1 h", "2,7 h", "1,4 h"],
        ["40 kWh", "5,4 h", "3,6 h", "1,8 h"],
        ["60 kWh", "8,1 h", "5,5 h", "2,7 h"],
    ]
    tab3 = Table(tempo, colWidths=[3.2 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm])
    tab3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e5496")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f9")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tab3)
    story.append(Paragraph(
        "Tempo calculado por Energia (kWh) / Potencia (kW); valores arredondados e sem "
        "considerar perdas de eficiencia, que podem elevar o tempo real em cerca de 10%.", NOTA))

    _build(BASE_DIR / "manual_chargegrid_resumo.pdf",
           "ChargeGrid - Resumo Didatico de Especificacoes do Carregador AC", FONTE_CHARGEGRID, story)


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    gerar_faq()
    gerar_tarifaria()
    gerar_chargegrid()
    for nome in (
        "faq_carregamento_condominio.pdf",
        "tabela_tarifaria.pdf",
        "manual_chargegrid_resumo.pdf",
    ):
        caminho = BASE_DIR / nome
        print(f"[OK] {caminho} ({caminho.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
