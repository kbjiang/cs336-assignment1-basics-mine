import os
import multiprocessing as mp
from typing import BinaryIO

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

def process_chunk(args):
    file_path, start, end = args  # Unpack the arguments properly
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pretoken_counts_ = {}
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    for doc in chunk.split("<|endoftext|>"):
        _ = get_initial_pretoken_counts(doc, PAT)
        for k, v in _.items():
            pretoken_counts_[k] = pretoken_counts_.get(k, 0) + v
    return pretoken_counts_
    
def get_initial_pretoken_counts(doc, pat):
    import regex as re
    pretokens = re.finditer(pat, doc)
    pretoken_counts = {}
    for ptok in pretokens:
        pretoken = ptok.group().encode("utf-8")
        pretoken = tuple(list(bytes([idx]) for idx in list(pretoken)))
        pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1
    return pretoken_counts

def get_pair_counts(pretoken_counts):
    """Optimized version using dict.get() with default"""
    pair_counts = {}
    for byte_tup, byte_tup_count in pretoken_counts.items():
        for i in range(len(byte_tup) - 1):
            pair = (byte_tup[i], byte_tup[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + byte_tup_count
    return pair_counts

def get_max_pair(pair_counts):
    max_counts = max(pair_counts.values())
    max_pair = max([byte_tup for byte_tup, byte_tup_count in pair_counts.items() if byte_tup_count==max_counts])
    return max_pair

def merge_one_tuple(byte_tup, max_pair):
    """Optimized version using list operations instead of tuple concatenation"""
    if len(byte_tup) < 2:
        return byte_tup
    
    result = []
    i = 0
    max_pair_0, max_pair_1 = max_pair  # Unpack once
    merged_token = max_pair_0 + max_pair_1  # Pre-compute joined bytes
    
    while i < len(byte_tup):
        if (i < len(byte_tup) - 1 and 
            byte_tup[i] == max_pair_0 and 
            byte_tup[i + 1] == max_pair_1):
            # Merge the pair
            result.append(merged_token)
            i += 2
        else:
            result.append(byte_tup[i])
            i += 1
    
    return tuple(result)

def merge_pretoken_counts(pretoken_counts, max_pair):
    new_pretoken_counts = {}
    for byte_tup, byte_tup_count in pretoken_counts.items():
        new_byte_tup = merge_one_tuple(byte_tup, max_pair)
        new_pretoken_counts[new_byte_tup] = byte_tup_count
    return new_pretoken_counts

def merge_pretoken_counts_optimized(pretoken_counts, pair_counts, max_pair, new_token):
    """
    Optimized version that incrementally updates pair counts instead of 
    recalculating everything from scratch. Only pairs that overlap with 
    the merged pair need to have their counts updated.
    """
    new_pretoken_counts = {}
    new_pair_counts = dict(pair_counts)  # Faster than .copy()
    
    # Remove the merged pair from pair counts
    new_pair_counts.pop(max_pair, None)
    
    for byte_tup, byte_tup_count in pretoken_counts.items():
        new_byte_tup = merge_one_tuple(byte_tup, max_pair)
        new_pretoken_counts[new_byte_tup] = byte_tup_count
        
        # Only update pair counts for sequences that actually changed
        if new_byte_tup != byte_tup:
            # Remove old pair counts for this sequence
            for i in range(len(byte_tup) - 1):
                old_pair = (byte_tup[i], byte_tup[i+1])
                count = new_pair_counts.get(old_pair, 0) - byte_tup_count
                if count <= 0:
                    new_pair_counts.pop(old_pair, None)
                else:
                    new_pair_counts[old_pair] = count
            
            # Add new pair counts for the merged sequence
            for i in range(len(new_byte_tup) - 1):
                new_pair = (new_byte_tup[i], new_byte_tup[i+1])
                new_pair_counts[new_pair] = new_pair_counts.get(new_pair, 0) + byte_tup_count
    
    return new_pretoken_counts, new_pair_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = [sp_tok.encode("utf-8") for sp_tok in special_tokens] + [bytes([i]) for i in range(256)]
    merges = []

    # pre-tokenization in parallel
    pretoken_counts = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens[0].encode("utf-8")
        )
        # Create arguments for each worker process
        chunk_args = [(input_path, start, end) 
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
        # Process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            pretoken_countss = pool.map(process_chunk, chunk_args)

    for pretoken_counts_ in pretoken_countss:
        for k, v in pretoken_counts_.items():
            pretoken_counts[k] = pretoken_counts.get(k, 0) + v

    # Optimized BPE training with incremental updates
    num_merges = vocab_size - len(vocab)
    
    # Build initial pair counts index
    pair_counts = get_pair_counts(pretoken_counts)
    
    for _ in range(num_merges):
        if not pair_counts:
            break
            
        # Find the most frequent pair
        max_pair = get_max_pair(pair_counts)
        
        # Create new merged token and add to vocab
        new_token = b"".join(max_pair)
        vocab.append(new_token)
        merges.append(max_pair)
        
        # Update pretoken_counts and pair_counts incrementally
        pretoken_counts, pair_counts = merge_pretoken_counts_optimized(
            pretoken_counts, pair_counts, max_pair, new_token
        )

    vocab = {i: v for i, v in enumerate(vocab)}
    return vocab, merges

if __name__ == "__main__":
    # input_path = "/home/azureuser/02-fun/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "/home/azureuser/02-fun/assignment1-basics/tests/fixtures/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    num_processes = 40
    vocab, merges = train_bpe(
        input_path, vocab_size, special_tokens, num_processes
    )
    # print(merges)
    # print(vocab)
