import os
import numpy as np
import multiprocessing as mp
import time
from typing import TextIO
from tokenizer import find_chunk_boundaries, Tokenizer
import threading
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed. Install with: pip install tqdm")
    exit(1)

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
            try:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            except UnicodeDecodeError:  # byte not valid, go back one position
                file.seek(initial_position - 1)
                continue

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == "":
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

def encode_chunk_with_progress(args):
    tokenizer, file_path, start, end, worker_id, progress_dict, chunk_size_bytes = args
    
    # Initialize progress for this worker
    progress_dict[worker_id] = 0
    
    with open(file_path, "r") as f:
        f.seek(start)
        chunk = f.read(end - start)
    
    # Process the chunk in smaller pieces to show real progress
    chunk_length = len(chunk)
    sub_chunk_size = max(1e4, chunk_length // 100)  # Process in ~2% increments
    
    all_tokens = []
    
    for i in range(0, chunk_length, sub_chunk_size):
        sub_chunk = chunk[i:i + sub_chunk_size]
        
        # Actually tokenize this sub-chunk
        sub_tokens = tokenizer.encode(sub_chunk)
        all_tokens.extend(sub_tokens)
        
        # Update progress based on characters processed
        chars_processed = min(i + len(sub_chunk), chunk_length)
        progress_percent = int((chars_processed / chunk_length) * 100)
        progress_dict[worker_id] = progress_percent
    
    progress_dict[worker_id] = 100  # Mark as complete
    return all_tokens

def monitor_progress(progress_dict, num_workers, total_chunks):
    """Monitor and display progress bars for all workers"""
    # Create progress bars for each worker
    progress_bars = {}
    for worker_id in range(num_workers):
        progress_bars[worker_id] = tqdm(
            total=100, 
            desc=f"Worker {worker_id:2d}", 
            position=worker_id,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:3.0f}/100'
        )
    
    # Monitor progress until all chunks are done
    completed_chunks = 0
    last_progress = {i: 0 for i in range(num_workers)}
    
    while completed_chunks < total_chunks:
        time.sleep(0.1)  # Update every 100ms
        
        completed_chunks = 0
        for worker_id in range(num_workers):
            current_progress = progress_dict.get(worker_id, 0)
            
            # Update progress bar
            if current_progress > last_progress[worker_id]:
                increment = current_progress - last_progress[worker_id]
                progress_bars[worker_id].update(increment)
                last_progress[worker_id] = current_progress
            
            if current_progress >= 100:
                completed_chunks += 1
    
    # Close all progress bars
    for bar in progress_bars.values():
        bar.close()

if __name__ == "__main__":
    # input_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # save_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/TinyStoriesV2-train.npy"
    # vocab_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_vocab_ts.json"
    # merges_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_merges_ts.txt"
    input_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/owt_train.txt"
    save_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/owt_train.npy"
    vocab_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_vocab_owt.json"
    merges_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_merges_owt.txt"

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
        
        # Calculate chunk sizes for progress tracking
        chunk_sizes = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # Create shared dictionary for progress tracking
        manager = mp.Manager()
        progress_dict = manager.dict()
        
        # Create arguments for each worker process with worker IDs
        chunk_args = [
            (tokenizer, input_path, start, end, worker_id, progress_dict, chunk_size) 
            for worker_id, ((start, end), chunk_size) in enumerate(zip(
                zip(boundaries[:-1], boundaries[1:]), chunk_sizes
            ))
        ]

    print(f"Starting parallel tokenization with individual worker progress...")
    print(f"Processing {len(chunk_args)} chunks with {num_workers} workers...")
    print()  # Add space before progress bars
    
    processing_start = time.time()
    
    # Start progress monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_progress, 
        args=(progress_dict, num_workers, len(chunk_args))
    )
    monitor_thread.start()
    
    # Process chunks in parallel
    with mp.Pool(processes=num_workers) as pool:
        token_idss = pool.map(encode_chunk_with_progress, chunk_args)
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    processing_time = time.time() - processing_start
    print(f"\nParallel tokenization completed in {processing_time:.2f} seconds")

    token_ids = [tid for tids in token_idss for tid in tids]
    np.save(save_path, np.array(token_ids, dtype=np.int32))
