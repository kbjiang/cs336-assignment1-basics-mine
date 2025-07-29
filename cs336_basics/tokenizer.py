import os
import json
import regex as re
from typing import Iterable, Iterator, BinaryIO
from ast import literal_eval

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def save_vocab_and_merge(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                            vocab_path: str, merges_path: str):
    byte_to_unicode = gpt2_bytes_to_unicode()

    # Convert the byte tokens in the vocab back to string tokens using the unicode mapping
    reversed_vocab = {''.join([byte_to_unicode[b] for b in bytes_token]):k
                      for k, bytes_token in vocab.items()}

    # Convert the byte sequences in merges back to string tokens
    reversed_merges = [' '.join([''.join([byte_to_unicode[b] for b in merge[0]]),
                                 ''.join([byte_to_unicode[b] for b in merge[1]])])
                       for merge in merges]

    # Save the vocab dictionary as a JSON file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(reversed_vocab, f, ensure_ascii=False)
    
    # Save the merges list to a file
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in reversed_merges:
            f.write(merge + '\n')

def get_vocab_and_merges_from_files(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike
    ):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
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

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

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

def split_by_special_tokens(special_tokens, text):
    if not special_tokens:
        return [text]
    escaped_patterns = [re.escape(p) for p in sorted(special_tokens, key=len, reverse=True)]
    pattern = f"({'|'.join(escaped_patterns)})"
    return re.split(pattern, text)

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
        chunks = split_by_special_tokens(self.special_tokens, text)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(chunk.encode())
                continue
            pretokens = re.findall(PAT, chunk)
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
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")