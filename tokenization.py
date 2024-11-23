import os
import json
import pickle

# Unicode
print(ord("H"))  # Convert to unicode
print(chr(72))  # Convert to character
# Byte-Pair Encoding
desired_vocab_size = 600
with open("input.txt", "r") as f:
    text = f.read()


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    def train(self, text, vocab_size=600, verbose=False):
        tokens = list(text.encode("utf-8"))
        initial_vocab_size = len(tokens)
        if verbose:
            print(f"Initial vocab size: {initial_vocab_size}")
        while len(tokens) > vocab_size:
            stats, most_frequent_bigram = self.get_stats(tokens)
            tokens, merge_idx = self.merge_tokens(
                tokens, most_frequent_bigram, stats, verbose=verbose
            )
            self.merges[most_frequent_bigram] = merge_idx
        compression_ratio = initial_vocab_size / len(tokens)
        if verbose:
            print(f"Compression ratio: {compression_ratio:.2f}x")
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for bigram, merge_idx in self.merges.items():
            self.vocab[merge_idx] = self.vocab[bigram[0]] + self.vocab[bigram[1]]
        print(len(self.merges))

    def get_stats(self, tokens):
        counts = {}
        max_count = 0
        most_frequent_bigram = None
        for ch1, ch2 in zip(tokens, tokens[1:]):
            bigram = (ch1, ch2)
            counts[bigram] = counts.get(bigram, 0) + 1
            if counts[bigram] > max_count:
                max_count = counts[bigram]
                most_frequent_bigram = bigram
        return counts, most_frequent_bigram

    def merge_tokens(self, tokens, most_frequent_bigram, stats, verbose=False):
        new_token = max(256, max(tokens) + 1)
        if verbose:
            print(f"Merging {most_frequent_bigram} into {new_token}")
        ix = 0
        while ix < len(tokens) - 1:
            if (tokens[ix], tokens[ix + 1]) == most_frequent_bigram:
                tokens = tokens[:ix] + [new_token] + tokens[ix + 2 :]
                ix = 0
            else:
                ix += 1
        return tokens, new_token

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        merged_tokens = []
        ix = 0
        while ix < len(tokens):
            if ix == len(tokens) - 1:
                merged_tokens.append(tokens[ix])
                break
            else:
                bigram = (tokens[ix], tokens[ix + 1])
                if bigram in self.merges:
                    merged_tokens.append(self.merges[bigram])
                    ix += 2
                else:
                    merged_tokens.append(tokens[ix])
                    ix += 1
        return merged_tokens

    def decode(self, ids):
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        merges = {self.decode(list(k)): v for k, v in self.merges.items()}
        print(len(merges))
        with open(f"{path}/merges.json", "w") as f:
            json.dump(merges, f)

    def load(self, path):
        with open(f"{path}/vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        with open(f"{path}/merges.json", "r") as f:
            self.merges = json.load(f)


test = open("taylorswift.txt", "r").read()

tokenizer = BPETokenizer()
tokenizer.train(text[:10000], vocab_size=5000)
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)
tokenizer.save("tokenizer_bpe")

assert decoded == test, "Decoding failed"
