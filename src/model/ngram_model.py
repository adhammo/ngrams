import os
import json


class NGramModel:
    def __init__(self):
        self.vocab = []
        self.model = {}

    # Build the vocabulary
    # * collect all unique words; replace any word appearing fewer than UNK_THRESHOLD times (from config/.env) with <UNK>;
    # * add <UNK> to the vocabulary.
    def build_vocab(self, token_file):
        token_freq = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for sentence in f.read().splitlines():
                for token in sentence.split(" "):
                    token_freq[token] = token_freq.get(token, 0) + 1
        self.vocab = [
            token
            for token, freq in token_freq.items()
            if freq >= int(os.getenv("UNK_THRESHOLD"))
        ]
        self.vocab.append("<UNK>")

    # Build counts at all orders
    # * slide a window across every sentence and count all unique n-grams from 1-gram up to NGRAM_ORDER-gram.
    # * The number of orders is read from config/.env.
    def build_counts_and_probabilities(self, token_file):
        NGRAM_ORDER = int(os.getenv("NGRAM_ORDER"))
        self.model = {f"{i+1}gram": {} for i in range(NGRAM_ORDER)}
        with open(token_file, "r", encoding="utf-8") as f:
            for sentence in f.read().splitlines():
                tokens = [
                    tk if tk in self.vocab else "<UNK>" for tk in sentence.split(" ")
                ]
                for i in range(len(tokens)):
                    unigram = tokens[i]
                    self.model["1gram"][unigram] = (
                        self.model["1gram"].get(unigram, 0) + 1
                    )
                    for n in range(2, NGRAM_ORDER + 1):
                        if n - 1 > i:
                            continue
                        ngram = " ".join(tokens[i - n + 1 : i])
                        if ngram not in self.model[f"{n}gram"]:
                            self.model[f"{n}gram"][ngram] = {}
                        self.model[f"{n}gram"][ngram][tokens[i]] = (
                            self.model[f"{n}gram"][ngram].get(tokens[i], 0) + 1
                        )
            total_count = sum(self.model["1gram"].values())
            for unigram in self.model["1gram"]:
                self.model["1gram"][unigram] /= total_count
            for n in range(2, NGRAM_ORDER + 1):
                for ngram in self.model[f"{n}gram"]:
                    total_count = sum(self.model[f"{n}gram"][ngram].values())
                    for token in self.model[f"{n}gram"][ngram]:
                        self.model[f"{n}gram"][ngram][token] /= total_count

    # Backoff lookup
    # * try the highest-order context first, fall back to lower orders down to 1-gram.
    # * Return a dict of {word: probability} from the highest order that matches.
    # * Return empty dict if no match at any order.
    # * This is the single source of backoff logic in the project.
    def lookup(self, context):
        for n in range(int(os.getenv("NGRAM_ORDER")), 0, -1):
            if n == 1:
                return self.model["1gram"]
            tokens = context[-n + 1 :]
            ngram = " ".join(tokens)
            if ngram in self.model[f"{n}gram"]:
                return self.model[f"{n}gram"][ngram]
        return {}

    # Save all probability tables to model.json
    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.model, f, ensure_ascii=False, indent=4)

    # Save vocabulary list to vocab.json
    def save_vocab(self, vocab_path):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

    def load(self, model_path, vocab_path):
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
