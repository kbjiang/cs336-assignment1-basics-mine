from typing import Iterable, Iterator
from ast import literal_eval
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Iterable[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab if vocab else {}
        self.vocab_reversed = {v:k for k,v in vocab.items()}
        self.merges = merges if merges else []
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] | None=None):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {int(k): v.encode() for k,v in vocab.items()}
        merges = []
        with open(merges_filepath) as f:
            for line in f.readlines():
                line = line.strip("\n")
                merge = literal_eval(line)
                merges.append((merge[0].encode(), merge[1].encode()))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokens = re.findall(PAT, text)
        tokens = []
        for pretoken in pretokens:
            pretoken  = [bytes([b]) for b in pretoken.encode()]
            token = pretoken[0]
            i = 1
            while i < len(pretoken):
                if (token, pretoken[i]) in self.merges:
                    token = b"".join((token, pretoken[i]))
                    i += 1
                else:
                    tokens.append(token)
                    token = pretoken[i]
                    i += 1
            tokens.append(token)
        return [self.vocab_reversed[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]):
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8")