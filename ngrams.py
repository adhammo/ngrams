import string
import numpy as np

debug_mode = False


def get_sentences(text):
    """Clean the text by lowercasing, removing punctuation, and splitting into sentences."""
    # lowercase text
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation.replace(".", "") + "”“’‘"))
    # remove lines
    text = text.replace("\n", " ")
    # split into sentences
    sentences = [sentence.strip() for sentence in text.split(".")]
    # return sentences
    return sentences


def get_ngrams(text):
    """Generate ngrams from text."""
    # create ngrams table
    ngrams = {
        "unigrams": {},
        "bigrams": {},
        "trigrams": {},
        "fourgrams": {},
    }
    # get sentences from text
    sentences = get_sentences(text)
    # iterate through sentences
    for sentence in sentences:
        # skip empty sentences
        if sentence == "":
            continue
        # append <BOS> and <EOS> tokens
        words = ["<BOS>"] * 4 + sentence.split() + ["<EOS>"]
        # iterate through words
        for i in range(4, len(words)):
            # populate ngrams ngrams
            unigram = words[i]
            bigram = (words[i - 1], words[i])
            trigram = (words[i - 2], words[i - 1], words[i])
            fourgram = (words[i - 3], words[i - 2], words[i - 1], words[i])
            ngrams["unigrams"][unigram] = ngrams["unigrams"].get(unigram, 0) + 1
            ngrams["bigrams"][bigram] = ngrams["bigrams"].get(bigram, 0) + 1
            ngrams["trigrams"][trigram] = ngrams["trigrams"].get(trigram, 0) + 1
            ngrams["fourgrams"][fourgram] = ngrams["fourgrams"].get(fourgram, 0) + 1
    # return ngrams
    return ngrams


def get_next_word(ngrams, text):
    """Predict the next word based on the last words in the text."""
    # get sentences from text
    sentences = get_sentences(text)
    # get the last sentence
    sentence = sentences[-1] if sentences else ""
    # handle empty sentence
    sentence = "<BOS>" if sentence == "" else sentence
    # append <BOS> and <EOS> tokens
    words = ["<BOS>"] * 3 + sentence.split()
    # calculate probabilities for each unigram
    keys = list(ngrams["unigrams"].keys())
    probs = np.zeros((4, len(keys))).astype(np.uint64)
    probs[0] = np.array(list(ngrams["unigrams"].values())).astype(np.uint64)
    for bigram, count in ngrams["bigrams"].items():
        if tuple(words[-1:]) == bigram[:1]:
            probs[1, keys.index(bigram[1])] = count
    for trigram, count in ngrams["trigrams"].items():
        if tuple(words[-2:]) == trigram[:2]:
            probs[2, keys.index(trigram[2])] = count
    for fourgram, count in ngrams["fourgrams"].items():
        if tuple(words[-3:]) == fourgram[:3]:
            probs[3, keys.index(fourgram[3])] = count
    # combine probabilities
    probs = (
        probs[0]
        * (probs[1] if np.any(probs[1]) else 1)
        * (probs[2] if np.any(probs[2]) else 1)
        * (probs[3] if np.any(probs[3]) else 1)
    )
    # print debug information
    if debug_mode:
        print("words: ", words)
        print(
            "predictions: ",
            [
                unigram.replace("<EOS>", ".")
                for unigram in sorted(keys, key=lambda x: probs[keys.index(x)], reverse=True)[:10]
            ],
        )
    # return the unigram with the highest probability
    return max(keys, key=lambda x: probs[keys.index(x)]).replace("<EOS>", ".")


if __name__ == "__main__":
    import streamlit as st

    @st.cache_resource
    def load_ngrams():
        learn_text = ""
        with open("ngrams/sherlock.txt", "r", encoding="utf-8") as f:
            learn_text += f.read() + "\n"
        ngrams = get_ngrams(learn_text[:len(learn_text) // 2])
        return ngrams

    @st.cache_resource
    def verify_ngrams(ngrams):
        verify_text = ""
        with open("ngrams/sherlock.txt", "r", encoding="utf-8") as f:
            verify_text += f.read() + "\n"
        accuarcy = 0
        total = 0
        count = 0
        sentences = get_sentences(verify_text[len(verify_text) // 2:])
        for sentence in sentences:
            count += 1
            words = sentence.split()
            for i in range(len(words)):
                next_word = get_next_word(ngrams, " ".join(words[:i]))
                if next_word == words[i]:
                    accuarcy += 1
                total += 1
            print(f"{count}/{len(sentences)}", accuarcy / total * 100)
        return accuarcy / total * 100

    ngrams = load_ngrams()
    accuarcy = verify_ngrams(ngrams)
    print(accuarcy)

    st.title("N-gram Next Word Predictor")
    user_input = st.text_input("Enter a sentence:", "")
    predicted_word = get_next_word(ngrams, user_input)
    st.write(f"Predicted next word: **{predicted_word}**")
