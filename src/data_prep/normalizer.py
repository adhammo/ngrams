import os
import string
import re
from unidecode import unidecode


class Normalizer:
    def __init__(self):
        self.punctuation_table = str.maketrans(
            "",
            "",
            string.punctuation.replace(".", "").replace("?", "").replace("!", ""),
        )
        self.digits_table = str.maketrans("", "", string.digits)
        self.titles_regex = re.compile(r"\b(mr|mrs|ms)\.\s*")

    # Load raw text
    # * call load() on the training folder (TRAIN_RAW_DIR). Load all .txt files found in the folder.
    def load(self, folder_path):
        books = []
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                books.append(unidecode(f.read()))
        return books

    # Strip Gutenberg header and footer
    # * Remove all text before and including: *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    # * Remove all text from and including: *** END OF THE PROJECT GUTENBERG EBOOK ... ***
    def strip_gutenberg(self, text):
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        start_index = text.find(start_marker)
        # move to the end of the line after the start marker
        start_index = 0 if start_index == -1 else text.find("\n", start_index)
        end_index = text.find(end_marker)
        end_index = len(text) if end_index == -1 else end_index
        return text[start_index:end_index].strip()

    # Lowercase all text
    def lowercase(self, text):
        return text.lower()

    # Remove all punctuation
    def remove_punctuation(self, text):
        return text.translate(self.punctuation_table)

    # Remove all numbers
    def remove_numbers(self, text):
        return text.translate(self.digits_table)

    # Remove extra whitespace and blank lines
    def remove_whitespace(self, text):
        return "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    # Normalize
    # * call normalize(text) which applies lowercase, remove punctuation, remove numbers, and remove extra whitespace in order.
    # * Apply all normalization steps in order: lowercase → remove punctuation → remove numbers → remove whitespace.
    # * This is the single method that other modules call to normalize text consistently.
    def normalize(self, text):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    # Word tokenize
    # * call word_tokenize(sentence) on each sentence to split it into tokens separated by a single space.
    # * Split a single sentence into a list of tokens
    def word_tokenize(self, sentence):
        return " ".join(word.strip() for word in sentence.split(" ") if word.strip())

    # Sentence tokenize
    # * split text into sentences; each becomes one line in the output file.
    def sentence_tokenize(self, text):
        text = text.replace("\n", " ")
        text = text.replace("?", ".")
        text = text.replace("!", ".")
        return "\n".join(
            [
                self.word_tokenize(sentence.strip())
                for sentence in re.sub(self.titles_regex, r"\1 ", text).split(".")
                if sentence.strip()
            ]
        )

    # Write output file
    # * concatenate all training books and write to train_tokens.txt.
    # * Format: one sentence per line, tokens separated by spaces.
    # * If implementing the Model Evaluator extra credit, also process EVAL_RAW_DIR and write eval_tokens.txt using the same pipeline.
    def save(self, sentences, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sentences)
