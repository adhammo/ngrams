import argparse
import os
from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

STEPS = ["dataprep", "model", "inference", "all"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Gram Language Model Pipeline")
    parser.add_argument(
        "--step",
        choices=STEPS,
        nargs="+",
        default=STEPS,
        help="Pipeline step(s) to run: dataprep, model, inference, all (default: all)",
    )
    args = parser.parse_args()
    steps = set(args.step)

    load_dotenv("config/.env")

    normalizer = Normalizer()

    if "dataprep" in steps or "all" in steps:
        for mode in ["TRAIN", "EVAL"]:
            sentences = []
            books = normalizer.load(os.environ[f"{mode}_RAW_DIR"])
            for book in books:
                normalized_book = normalizer.normalize(normalizer.strip_gutenberg(book))
                sentences.append(normalizer.sentence_tokenize(normalized_book))
            normalizer.save("\n".join(sentences), f"{os.environ[f'{mode}_TOKENS']}")

    ngram_model = NGramModel()

    if "model" in steps or "all" in steps:
        ngram_model.build_vocab(os.environ["TRAIN_TOKENS"])
        ngram_model.build_counts_and_probabilities(os.environ["TRAIN_TOKENS"])
        ngram_model.save_vocab(f"{os.environ['VOCAB']}")
        ngram_model.save_model(f"{os.environ['MODEL']}")
    else:
        ngram_model.load(f"{os.environ['MODEL']}", f"{os.environ['VOCAB']}")

    if "inference" in steps or "all" in steps:
        predictor = Predictor(ngram_model, normalizer)

        while True:
            user_input = input("Enter a sentence (or type 'quit' to quit): ")
            if user_input.lower() == "quit":
                break
            prediction = predictor.predict_next(user_input, int(os.getenv("TOP_K")))
            print(f"Predicted next word: {prediction}")
