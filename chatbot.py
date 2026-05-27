import os
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# CONFIGURAÇÃO INICIAL
# =========================

# Carrega variáveis do .env
load_dotenv()

# Lê API Key
api_key = os.getenv("OPENROUTER_API_KEY")

# Verificação básica
if not api_key:
    print("Erro: API Key não encontrada no arquivo .env")
    exit()

# Cliente OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# =========================
# SYSTEM PROMPT
# =========================

try:
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as file:
        system_prompt = file.read()

except FileNotFoundError:
    print("Erro: arquivo system_prompt.txt não encontrado.")
    exit()

print("Conexão configurada com sucesso.")
print("Digite 'sair' para encerrar.\n")

# =========================
# HISTÓRICO DE CONVERSA
# =========================

messages = [
    {
        "role": "system",
        "content": system_prompt
    }
]

# =========================
# LOOP PRINCIPAL
# =========================

while True:

    # Entrada do usuário
    user_input = input("Você: ")

    # Encerrar chatbot
    if user_input.lower() == "sair":
        print("\nChatbot encerrado.")
        break

    # Adiciona mensagem do usuário ao histórico
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:

        # Requisição para IA
        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",

            messages=messages,

            temperature=0.3
        )

        # Resposta da IA
        assistant_response = response.choices[0].message.content

        # Exibe resposta
        print(f"\nChatbot: {assistant_response}\n")

        # Adiciona resposta ao histórico
        messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

    except Exception as error:

        print("\nErro ao gerar resposta.")
        print(error)
