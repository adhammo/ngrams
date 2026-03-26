import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports work when Streamlit
# launches this file directly (e.g. `streamlit run src/ui/app.py`).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


class PredictorUI:
    """Browser-based interface for the n-gram next-word predictor."""

    def __init__(self, predictor: Predictor, top_k: int):
        self.predictor = predictor
        self.top_k = top_k

    def run(self) -> None:
        st.title("N-Gram Next-Word Predictor")
        st.write(
            "Type a sentence fragment below and the model will suggest "
            "the most likely next words."
        )

        user_input = st.text_input(
            "Input sentence:",
            placeholder="e.g. it was the best of",
        )

        predictions = self.predictor.predict_next(user_input, self.top_k)
        if predictions:
            st.subheader("Top predictions")
            for rank, word in enumerate(predictions, start=1):
                st.write(f"{rank}. **{word}**")
        else:
            st.info("No predictions available for the given input.")


@st.cache_resource
def _load_predictor() -> tuple[Predictor, int]:
    """Load model and normalizer once; cache across Streamlit reruns."""
    # Resolve config/.env relative to the project root regardless of cwd.
    env_path = _PROJECT_ROOT / "config" / ".env"
    load_dotenv(str(env_path))

    ngram_model = NGramModel()
    ngram_model.load(
        str(_PROJECT_ROOT / os.environ["MODEL"]),
        str(_PROJECT_ROOT / os.environ["VOCAB"]),
    )
    normalizer = Normalizer()
    predictor = Predictor(ngram_model, normalizer)
    return predictor, int(os.getenv("TOP_K", 3))


# Streamlit executes the module at the top level on every rerun.
predictor, top_k = _load_predictor()
PredictorUI(predictor, top_k).run()
