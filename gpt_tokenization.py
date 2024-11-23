import os
import json
import pickle
import regex as re
import tiktoken

# Unicode
print(ord("H"))  # Convert to unicode
print(chr(72))  # Convert to character
# Byte-Pair Encoding
desired_vocab_size = 600
with open("input.txt", "r") as f:
    text = f.read()


class RegexTokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def train(self, text, vocab_size=600, verbose=False):
        splits = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens = [list(split.encode("utf-8")) for split in splits]
        initial_vocab_size = sum(len(token) for token in tokens)
        if verbose:
            print(f"Initial vocab size: {initial_vocab_size}")
        vocab_size = initial_vocab_size
        while vocab_size > desired_vocab_size:
            stats, most_frequent_bigram = self.get_stats(tokens)
            if most_frequent_bigram is None:
                break
            tokens, merge_idx = self.merge_tokens(
                tokens, most_frequent_bigram, stats, verbose=verbose
            )
            vocab_size = sum(len(token) for token in tokens)
            self.merges[most_frequent_bigram] = merge_idx
        compression_ratio = initial_vocab_size / vocab_size
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
        for ids in tokens:
            for ch1, ch2 in zip(ids, ids[1:]):
                bigram = (ch1, ch2)
                counts[bigram] = counts.get(bigram, 0) + 1
                if counts[bigram] > max_count:
                    max_count = counts[bigram]
                    most_frequent_bigram = bigram
        return counts, most_frequent_bigram

    def merge_tokens(self, tokens, most_frequent_bigram, stats, verbose=False):
        current_max = max(max(ids) for ids in tokens)
        new_token = max(256, current_max + 1)
        if verbose:
            print(f"Merging {most_frequent_bigram} into {new_token}")
        for i in range(len(tokens)):
            ix = 0
            while ix < len(tokens[i]) - 1:
                if (tokens[i][ix], tokens[i][ix + 1]) == most_frequent_bigram:
                    tokens[i] = tokens[i][:ix] + [new_token] + tokens[i][ix + 2 :]
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

tokenizer = RegexTokenizer()
tokenizer.train(text[:10000], vocab_size=5000, verbose=True)
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)
tokenizer.save("tokenizer_regex")

assert decoded == test, "Decoding failed"

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
len(enc.encode("hello world!!!? (안녕하세요!) lol123 😉"))
len(tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉"))


parts = [bytes([b]) for b in b"CHILD"]
for i, pair in enumerate(zip(parts[:-1], parts[1:])):
    print(pair[0] + pair[1])
    rank = enc._mergeable_ranks.get(pair[0] + pair[1])
    print(i, rank)
