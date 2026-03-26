import os


class Predictor:

    # Accept a pre-loaded NGramModel and Normalizer instance. Do not load files here.
    def __init__(self, model, normalizer):
        self.model = model
        self.normalizer = normalizer

    # Call Normalizer.normalize(text);
    # * extract last NGRAM_ORDER − 1 words as context
    def normalize(self, text):
        lines = self.normalizer.normalize(text).splitlines()
        if len(lines) == 0:
            return []
        tokens = lines[-1].split(" ")
        if len(tokens) < (int(os.getenv("NGRAM_ORDER")) - 1):
            return tokens
        return tokens[-(int(os.getenv("NGRAM_ORDER")) - 1) :]

    # Replace out-of-vocabulary words with <UNK>
    def map_oov(self, context):
        return [word if word in self.model.vocab else "<UNK>" for word in context]

    # Orchestrate normalize → map_oov → NGramModel.lookup() → return top-k words sorted by probability
    def predict_next(self, text, k):
        normalized_text = self.normalize(text)
        context = self.map_oov(normalized_text)
        predictions = self.model.lookup(context)
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [pred[0] for pred in sorted_predictions[:k]]
