
"""Amharic next-word prediction using an n-gram language model.

Run the application with:
    streamlit run app/streamlit_app.py -- --model-dir data/gpac --order 3

The application predicts the next word based on the preceding words.
If the input ends without a space, the last token is treated as a partial
word, and suggestions are filtered to match that prefix.

Selecting a suggestion adds it to the input and updates the predictions.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ngram import pipeline  # noqa: E402
from ngram.generate import Suggester  # noqa: E402
from ngram.models import make_model  # noqa: E402
from ngram.normalize import NUM, normalize_chars, tokenize  # noqa: E402


def parse_args():
    """Read the model directory and maximum n-gram order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="data/gpac")
    parser.add_argument("--order", type=int, default=3)
    return parser.parse_args(sys.argv[1:])


@st.cache_resource
def load_suggesters(model_dir, max_order):
    """Load vocabulary and build language models for each n-gram order."""
    vocab, counts = pipeline.load_counts(model_dir, max_order)

    # Create a separate suggester for each available model order.
    models = {
        order: Suggester(make_model(counts, order, "kn"), vocab)
        for order in range(1, max_order + 1)
    }

    return vocab, models


def split_input(text):
    """Separate completed words from a potentially unfinished last word."""
    text = normalize_chars(text)
    has_trailing_space = text[-1:].isspace()
    words = tokenize(text)

    # Without a trailing space, the last token may still be incomplete.
    if words and not has_trailing_space:
        return words[:-1], words[-1]

    return words, ""


def accept(word):
    """Append a selected suggestion to the text input."""
    completed_words, _ = split_input(st.session_state.text)

    # Add a space so the user can continue typing the next word.
    st.session_state.text = " ".join(completed_words + [word]) + " "


def main():
    """Initialize the interface and display word predictions."""
    args = parse_args()

    st.set_page_config(
        page_title="Amharic Next-Word Suggestion",
        layout="centered"
    )
    st.title("Amharic Next-Word Suggestion")

    # Load the vocabulary and trained n-gram models.
    vocab, suggesters = load_suggesters(args.model_dir, args.order)

    # Allow users to choose the model order and number of suggestions.
    n = st.sidebar.select_slider(
        "N-gram order",
        options=list(range(1, args.order + 1)),
        value=args.order
    )
    k = st.sidebar.slider("Number of suggestions", 1, 10, 5)

    st.sidebar.caption(
        f"Vocabulary: {len(vocab):,} words · Interpolated Kneser-Ney"
    )

    # Keep the input value between Streamlit reruns.
    st.session_state.setdefault("text", "")

    st.text_input(
        "Type an incomplete Amharic sentence",
        key="text",
        placeholder="ኢትዮጵያ ውስጥ የቡና "
    )

    # Determine the context words and any partial word being typed.
    words, partial = split_input(st.session_state.text)

    # An n-gram model uses at most n-1 preceding words as context.
    history = words[-(n - 1):] if n > 1 else []

    context_message = f"History used: {' '.join(history) or '(none)'}"
    if partial:
        context_message += f" · Completing: “{partial}”"

    st.caption(context_message)

    # Generate candidates and exclude special number placeholders.
    candidates = suggesters[n].suggest(
        words,
        k=k + 5,
        partial=partial
    )
    suggestions = [
        (word, probability)
        for word, probability in candidates
        if NUM not in word
    ][:k]

    # Display a message if no matching words are available.
    if not suggestions:
        st.info("No candidates found for this prefix.")
        return

    # Each suggestion is clickable and updates the input text.
    columns = st.columns(len(suggestions))

    for column, (word, _) in zip(columns, suggestions):
        column.button(
            word,
            on_click=accept,
            args=(word,),
            width="stretch"
        )

    # Show the suggested words and their corresponding probabilities.
    results = pd.DataFrame(
        suggestions,
        columns=["Word", "P(word | history)"]
    )
    st.dataframe(results, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
