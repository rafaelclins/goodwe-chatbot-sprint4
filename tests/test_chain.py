import shutil
import subprocess
import unittest


class ChainTest(unittest.TestCase):
    def test_chain_can_be_built_and_optionally_invoked(self):
        try:
            from src.chain.builder import build_chain
        except ModuleNotFoundError as error:
            if error.name in {"langchain_core", "langchain_ollama"}:
                self.skipTest(f"Dependency not installed: {error.name}")
            raise

        chain = build_chain()
        self.assertIsNotNone(chain)

        if shutil.which("ollama") is None:
            self.skipTest("Ollama CLI not available")

        models = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "gpt-oss:120b" not in models.stdout:
            self.skipTest("Model gpt-oss:120b not available")

        response = chain.invoke({"input": "O que é kWh?"})
        self.assertIsInstance(response, str)
        self.assertTrue(response.strip())


if __name__ == "__main__":
    unittest.main()