import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from src.schemas.consulta_recarga import ConsultaRecarga


def _build_model(model=None):
	load_dotenv()
	api_key = os.getenv("OLLAMA_API_KEY")
	if not api_key:
		raise RuntimeError("OLLAMA_API_KEY não configurada")

	if model is not None:
		return model

	return ChatOllama(
		model="gpt-oss:120b",
		base_url="https://ollama.com",
		client_kwargs={
			"headers": {
				"Authorization": f"Bearer {api_key}",
			}
		},
	)


def _build_prompt(format_instructions=None):
	prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "system_prompt_v2.2.txt"
	system_prompt = prompt_path.read_text(encoding="utf-8")
	human_template = "{input}"
	if format_instructions:
		human_template += "\n\n{format_instructions}"

	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			MessagesPlaceholder(variable_name="history", optional=True),
			("human", human_template),
		]
	)
	if format_instructions:
		return prompt.partial(format_instructions=format_instructions)
	return prompt


def build_chain(model=None, format_instructions=None):
	return _build_prompt(format_instructions) | _build_model(model) | StrOutputParser()


def build_structured_chain(model=None):
	parser = PydanticOutputParser(pydantic_object=ConsultaRecarga)
	chain = build_chain(
		model=model,
		format_instructions=parser.get_format_instructions(),
	)
	return chain | parser

