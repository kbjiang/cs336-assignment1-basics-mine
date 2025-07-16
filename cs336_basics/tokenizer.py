from typing import Iterable, Iterator
from ast import literal_eval
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def update_pretoken(pretoken: Iterable[bytes], pair: tuple[bytes, bytes]):
    result = []
    i = 0
    while i < len(pretoken):
        if (i < len(pretoken) - 1 and 
            pretoken[i] == pair[0] and 
            pretoken[i + 1] == pair[1]):
            # Merge the pair
            result.append(b"".join(pair))
            i += 2
        else:
            result.append(pretoken[i])
            i += 1
    
    return result

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Iterable[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab if vocab else {}
        self.merges = merges if merges else []
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] | None=None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        vocab_reversed = {v:k for k,v in self.vocab.items()}
        pretokens = re.findall(PAT, text)
        tokens = []
        for pretoken in pretokens:
            pretoken  = [bytes([b]) for b in pretoken.encode()]
            while len(pretoken) >= 2:
                pairs = list(zip(pretoken[:-1], pretoken[1:]))
                try:
                    pid = min([self.merges.index(p) for p in pairs if p in self.merges])
                    pair = self.merges[pid]
                    pretoken = update_pretoken(pretoken, pair)
                except ValueError:
                    break
            tokens.extend(pretoken)
        return [vocab_reversed[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]):
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8")