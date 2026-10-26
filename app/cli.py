
"""Terminal version of the next-word suggester (for Colab or machines without a browser).

Run: python app/cli.py --model-dir data/gpac --order 3

Type words, then press Enter. Enter a number to append that suggestion, an empty line to
start a new sentence, or `q` to quit.
"""

import argparse
import sys
from pathlib import Path

# Add the project root so local ngram modules can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ngram import pipeline  # noqa: E402
from ngram.generate import Suggester  # noqa: E402
from ngram.models import make_model  # noqa: E402
from ngram.normalize import normalize_chars, tokenize  # noqa: E402


def main():
    # Parse command-line arguments for the model and suggestion settings.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/gpac")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    # Load the trained n-gram counts and initialize the suggester.
    vocab, counts = pipeline.load_counts(args.model_dir, args.order)
    suggester = Suggester(make_model(counts, args.order, "kn"), vocab)

    words = []

    # Continue suggesting words until the user chooses to quit.
    while True:
        suggestions = suggester.suggest(words, k=args.k)

        # Display the current sentence and its next-word suggestions.
        print("\n" + (" ".join(words) or "(new sentence)"))
        for i, (w, p) in enumerate(suggestions, 1):
            print(f"  {i}. {w}  ({p:.4f})")

        line = input("> ").strip()

        if line == "q":
            break
        if not line:
            # Clear the current sentence and start over.
            words = []
        elif line.isdigit() and 1 <= int(line) <= len(suggestions):
            # Append the word corresponding to the selected suggestion.
            words.append(suggestions[int(line) - 1][0])
        else:
            # Normalize and tokenize manually entered text before adding it.
            words.extend(tokenize(normalize_chars(line)))


if __name__ == "__main__":
    main()
