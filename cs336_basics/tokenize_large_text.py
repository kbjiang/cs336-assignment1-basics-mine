import os
import numpy as np
import multiprocessing as mp
import time
from typing import TextIO
from tokenizer import find_chunk_boundaries, Tokenizer

def find_chunk_boundaries(
    file: TextIO, 
    desired_num_chunks: int, 
    split_special_token: str
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, str), (
        "Must represent special token as a string"
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

def encode_chunk(args):
    tokenizer, file_path, start, end = args  # Unpack the arguments properly
    with open(file_path, "r") as f:
        f.seek(start)
        chunk = f.read(end - start)
    return tokenizer.encode(chunk)

if __name__ == "__main__":
    input_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_vocab_ts.json"
    merges_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_merges_ts.txt"
    save_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/TinyStoriesV2-train.npy"

    special_tokens = ["<|endoftext|>"]
    num_workers = 40

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens = special_tokens,
    )

    with open(input_path, encoding="utf-8") as f:
        boundaries = find_chunk_boundaries(
            f, num_workers, special_tokens[0]
        )
        # Create arguments for each worker process
        chunk_args = [(tokenizer, input_path, start, end) 
                    for start, end in zip(boundaries[:-1], boundaries[1:])]

    # Process chunks in parallel
    print(f"Parallel tokenization in progress...")
    processing_start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        token_idss = pool.map(encode_chunk, chunk_args)
    processing_time = time.time() - processing_start
    print(f"Parallel tokenization completed in {processing_time:.2f} seconds")

    token_ids = [tid for tids in token_idss for tid in tids]
    np.save(save_path, np.array(token_ids, dtype=np.int32))

    
