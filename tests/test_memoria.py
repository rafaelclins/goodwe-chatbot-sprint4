import unittest

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda

from src.chain.memoria import build_memory_chain, get_session_history


def fake_response(payload):
    history = "|".join(message.content for message in payload["history"])
    return f"history={history};input={payload['input']}"


class ConversationMemoryTest(unittest.TestCase):
    def setUp(self):
        get_session_history("session-a").clear()
        get_session_history("session-b").clear()
        get_session_history("session-three-turns").clear()
        get_session_history("session-limit").clear()

    def test_sessions_are_independent_and_keep_context(self):
        chain = build_memory_chain(
            max_history_tokens=1000,
            chain=RunnableLambda(fake_response),
        )
        session_a = {"configurable": {"session_id": "session-a"}}
        session_b = {"configurable": {"session_id": "session-b"}}

        chain.invoke({"input": "A primeira mensagem"}, config=session_a)
        chain.invoke({"input": "B primeira mensagem"}, config=session_b)
        response_a = chain.invoke(
            {"input": "A terceira interação"},
            config=session_a,
        )

        self.assertIn("A primeira mensagem", response_a)
        self.assertNotIn("B primeira mensagem", response_a)
        self.assertEqual(len(get_session_history("session-b").messages), 2)

    def test_three_turn_conversation_passes_previous_messages_to_model(self):
        observed_histories = []

        def inspect_history(payload):
            observed_histories.append(
                [message.content for message in payload["history"]]
            )
            return fake_response(payload)

        chain = build_memory_chain(
            max_history_tokens=1000,
            chain=RunnableLambda(inspect_history),
        )
        config = {"configurable": {"session_id": "session-three-turns"}}

        chain.invoke({"input": "Minha bateria tem 60 kWh."}, config=config)
        chain.invoke({"input": "A carga começa em 20%."}, config=config)
        third_response = chain.invoke(
            {"input": "Quero chegar a 80%. Qual é a energia necessária?"},
            config=config,
        )

        self.assertIn("Minha bateria tem 60 kWh.", observed_histories[2])
        self.assertIn("A carga começa em 20%.", observed_histories[2])
        self.assertIn("Minha bateria tem 60 kWh.", third_response)
        self.assertIn("A carga começa em 20%.", third_response)
        self.assertEqual(len(observed_histories), 3)

    def test_history_token_limit_is_applied(self):
        max_tokens = 1000
        observed_history_tokens = []
        observed_history_messages = []

        def inspect_history(payload):
            observed_history_tokens.append(
                count_tokens_approximately(payload["history"])
            )
            observed_history_messages.append(payload["history"])
            return "ok"

        chain = build_memory_chain(
            max_history_tokens=max_tokens,
            chain=RunnableLambda(inspect_history),
        )
        config = {"configurable": {"session_id": "session-limit"}}

        history = get_session_history("session-limit")
        messages = [
            (word + " ") * 55
            for word in (
                "historico um",
                "historico dois",
                "historico tres",
                "historico quatro",
                "historico cinco",
                "historico seis",
            )
        ]
        history.add_user_message(messages[0])
        history.add_ai_message(messages[1])
        history.add_user_message(messages[2])
        history.add_ai_message(messages[3])
        history.add_user_message(messages[4])
        history.add_ai_message(messages[5])

        chain.invoke(
            {"input": "nova mensagem"},
            config=config,
        )

        self.assertLessEqual(observed_history_tokens[0], max_tokens)
        self.assertLessEqual(
            count_tokens_approximately(history.messages),
            max_tokens,
        )
        trimmed_messages = observed_history_messages[0]
        self.assertTrue(trimmed_messages)
        self.assertEqual(trimmed_messages[0].type, "human")
        self.assertTrue(
            all(message.content in messages for message in trimmed_messages)
        )
        retained_contents = [message.content for message in trimmed_messages]
        self.assertNotIn(messages[0], retained_contents)
        self.assertEqual(history.messages[0].type, "human")


if __name__ == "__main__":
    unittest.main()