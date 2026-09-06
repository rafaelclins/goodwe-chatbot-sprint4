import unittest

from scripts.measure_prompt import DEFAULT_PROMPT_PATH, measure_prompt


class PromptMeasurementTest(unittest.TestCase):
	def test_current_prompt_has_reproducible_measurement(self):
		measurement = measure_prompt()

		self.assertEqual(measurement["file"], "prompts/system_prompt.txt")
		self.assertEqual(measurement["encoding"], "cl100k_base")
		self.assertGreater(measurement["characters"], 0)
		self.assertGreater(measurement["words"], 0)
		self.assertGreater(measurement["tokens"], 0)
		self.assertTrue(DEFAULT_PROMPT_PATH.exists())


if __name__ == "__main__":
	unittest.main()