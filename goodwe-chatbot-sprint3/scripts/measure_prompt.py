import argparse
import json
import re
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
DEFAULT_ENCODING = "cl100k_base"


def measure_prompt(
	prompt_path: Path = DEFAULT_PROMPT_PATH,
	encoding_name: str = DEFAULT_ENCODING,
) -> dict[str, int | str]:
	text = prompt_path.read_text(encoding="utf-8")
	encoding = tiktoken.get_encoding(encoding_name)

	return {
		"file": prompt_path.relative_to(PROJECT_ROOT).as_posix(),
		"encoding": encoding_name,
		"characters": len(text),
		"words": len(re.findall(r"\S+", text)),
		"tokens": len(encoding.encode(text)),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Measure system prompt size.")
	parser.add_argument(
		"--prompt",
		type=Path,
		default=DEFAULT_PROMPT_PATH,
		help="prompt path, default: prompts/system_prompt.txt",
	)
	parser.add_argument(
		"--encoding",
		default=DEFAULT_ENCODING,
		help="tiktoken encoding name, default: cl100k_base",
	)
	args = parser.parse_args()
	prompt_path = args.prompt if args.prompt.is_absolute() else PROJECT_ROOT / args.prompt
	print(json.dumps(measure_prompt(prompt_path, args.encoding), ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()