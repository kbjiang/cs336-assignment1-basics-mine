import os
import numpy as np
import multiprocessing as mp
import time
from typing import TextIO
from tokenizer import find_chunk_boundaries, Tokenizer
import threading
import tempfile
import shutil
import json
import gc
from tqdm import tqdm

def find_chunk_boundaries(
    file: TextIO, 
    desired_num_chunks: int, 
    split_special_token: str,
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
    tokenizer, file_path, start, end, worker_id, progress_dict, chunk_size_bytes, chunk_id = args
    
    # Initialize progress for this worker
    progress_dict[worker_id] = {'progress': 0, 'chunk_id': chunk_id, 'last_update': time.time()}
    
    try:
        with open(file_path, "r") as f:
            f.seek(start)
            chunk = f.read(end - start)
        
        # Process the chunk in smaller pieces to show real progress
        chunk_length = len(chunk)
        sub_chunk_size = max(int(1e4), chunk_length // 100)  # Process in ~2% increments
        
        all_tokens = []
        
        for i in range(0, chunk_length, sub_chunk_size):
            sub_chunk = chunk[i:i + sub_chunk_size]
            
            # Actually tokenize this sub-chunk
            sub_tokens = tokenizer.encode(sub_chunk)
            all_tokens.extend(sub_tokens)
            
            # Update progress based on characters processed
            chars_processed = min(i + len(sub_chunk), chunk_length)
            progress_percent = int((chars_processed / chunk_length) * 100)
            progress_dict[worker_id] = {
                'progress': progress_percent, 
                'chunk_id': chunk_id, 
                'last_update': time.time()
            }
        
        progress_dict[worker_id] = {
            'progress': 100, 
            'chunk_id': chunk_id, 
            'last_update': time.time()
        }
        return all_tokens
        
    except Exception as e:
        print(f"Error in worker {worker_id}, chunk {chunk_id}: {e}")
        progress_dict[worker_id] = {
            'progress': -1, 
            'chunk_id': chunk_id, 
            'last_update': time.time(),
            'error': str(e)
        }
        return []

def monitor_progress_with_timeout(progress_dict, num_workers, batch_num=None, stuck_timeout=300, stop_event=None):
    """Monitor and display progress bars for all workers, detect stuck workers"""
    # Create progress bars for each worker
    progress_bars = {}
    for worker_id in range(num_workers):
        desc = f"Worker {worker_id:2d}"
        if batch_num is not None:
            desc = f"Batch {batch_num} - Worker {worker_id:2d}"
        progress_bars[worker_id] = tqdm(
            total=100, 
            desc=desc, 
            position=worker_id,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:3.0f}/100 Chunk:{postfix}'
        )
    
    # Monitor progress until all chunks are done
    last_progress = {i: 0 for i in range(num_workers)}
    stuck_workers = set()
    completed_workers = set()
    
    while len(completed_workers) < num_workers:
        # Check if we should stop monitoring
        if stop_event and stop_event.is_set():
            break
            
        time.sleep(0.5)  # Update every 500ms
        
        current_time = time.time()
        
        for worker_id in range(num_workers):
            worker_info = progress_dict.get(worker_id, {'progress': 0, 'chunk_id': 'N/A', 'last_update': current_time})
            
            if isinstance(worker_info, dict):
                current_progress = worker_info.get('progress', 0)
                chunk_id = worker_info.get('chunk_id', 'N/A')
                last_update = worker_info.get('last_update', current_time)
            else:
                # Handle old format (just progress number)
                current_progress = worker_info
                chunk_id = 'N/A'
                last_update = current_time
            
            # Check if worker is stuck
            time_since_update = current_time - last_update
            if time_since_update > stuck_timeout and current_progress < 100 and worker_id not in stuck_workers:
                stuck_workers.add(worker_id)
                print(f"\n⚠️  Worker {worker_id} appears stuck on chunk {chunk_id} (no progress for {time_since_update:.1f}s)")
            
            # Update progress bar
            if current_progress > last_progress[worker_id]:
                increment = current_progress - last_progress[worker_id]
                progress_bars[worker_id].update(increment)
                last_progress[worker_id] = current_progress
            
            # Update postfix to show chunk ID
            progress_bars[worker_id].set_postfix_str(f"{chunk_id}")
            
            # Mark worker as completed if finished
            if current_progress >= 100 or current_progress == -1:  # Complete or error
                completed_workers.add(worker_id)
    
    # Close all progress bars
    for bar in progress_bars.values():
        bar.close()
    
    return stuck_workers

def split_file_into_batches(input_path, num_batches, special_token, temp_dir):
    """Split the large file into smaller batch files for sequential processing"""
    print(f"Splitting file into {num_batches} batches...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        boundaries = find_chunk_boundaries(f, num_batches, special_token)
    
    batch_files = []
    batch_info = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # Create temporary file for this batch
            batch_file = os.path.join(temp_dir, f"batch_{i:03d}.txt")
            batch_files.append(batch_file)
            
            # Read and write this batch
            f.seek(start)
            while True:
                try:
                    batch_content = f.read(end - start)
                    break
                except UnicodeDecodeError:
                    print("hey hey")
                    start -= 1
                    continue
            
            with open(batch_file, 'w', encoding='utf-8') as batch_f:
                batch_f.write(batch_content)
            
            batch_size = len(batch_content)
            batch_info.append({
                'file': batch_file,
                'size': batch_size,
                'original_start': start,
                'original_end': end
            })
            
            print(f"  Batch {i+1}/{num_batches}: {batch_size:,} characters -> {batch_file}")
    
    return batch_files, batch_info

def process_batch_file_with_timeout(batch_file, tokenizer, num_workers, special_token, batch_num, timeout_minutes=10):
    """Process a single batch file with multiprocessing and timeout handling"""
    print(f"\nProcessing batch {batch_num}...")
    
    with open(batch_file, encoding="utf-8") as f:
        boundaries = find_chunk_boundaries(f, num_workers, special_token)
        
        # Calculate chunk sizes for progress tracking
        chunk_sizes = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # Create shared dictionary for progress tracking
        manager = mp.Manager()
        progress_dict = manager.dict()
        
        # Create arguments for each worker process with worker IDs and chunk IDs
        chunk_args = [
            (tokenizer, batch_file, start, end, worker_id, progress_dict, chunk_size, f"B{batch_num}C{worker_id}") 
            for worker_id, ((start, end), chunk_size) in enumerate(zip(
                zip(boundaries[:-1], boundaries[1:]), chunk_sizes
            ))
        ]

    print(f"  Processing {len(chunk_args)} chunks with {num_workers} workers (timeout: {timeout_minutes}min)...")
    
    processing_start = time.time()
    
    # Create stop event for monitor thread
    stop_event = threading.Event()
    
    # Start progress monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_progress_with_timeout, 
        args=(progress_dict, num_workers, batch_num, 300, stop_event),
        daemon=True  # Make it a daemon thread
    )
    monitor_thread.start()
    
    # Process chunks in parallel with timeout
    pool = None
    try:
        pool = mp.Pool(processes=num_workers)
        # Use map_async with timeout
        result = pool.map_async(encode_chunk_with_progress, chunk_args)
        token_idss = result.get(timeout=timeout_minutes * 60)  # Convert to seconds
        
        # Close pool immediately after getting results
        pool.close()
        pool.join()
        
    except mp.TimeoutError:
        print(f"\n❌ Batch {batch_num} timed out after {timeout_minutes} minutes!")
        # Terminate stuck workers
        if pool:
            pool.terminate()
            pool.join()
        
        # Collect information about stuck chunks
        stuck_chunks = []
        for worker_id in range(num_workers):
            worker_info = progress_dict.get(worker_id, {})
            if isinstance(worker_info, dict) and worker_info.get('progress', 0) < 100:
                stuck_chunks.append({
                    'batch': batch_num,
                    'worker_id': worker_id,
                    'chunk_id': worker_info.get('chunk_id', f'B{batch_num}C{worker_id}'),
                    'progress': worker_info.get('progress', 0),
                    'chunk_start': boundaries[worker_id] if worker_id < len(boundaries) - 1 else 0,
                    'chunk_end': boundaries[worker_id + 1] if worker_id + 1 < len(boundaries) else 0,
                    'chunk_size': chunk_sizes[worker_id] if worker_id < len(chunk_sizes) else 0
                })
        
        # Save stuck chunk information
        stuck_log_path = f"stuck_chunks_batch_{batch_num}.json"
        with open(stuck_log_path, 'w') as f:
            json.dump(stuck_chunks, f, indent=2)
        print(f"  Saved stuck chunk info to: {stuck_log_path}")
        
        # Return empty results for this batch
        token_idss = [[] for _ in chunk_args]
    
    finally:
        # Signal monitor thread to stop
        stop_event.set()
        
        # Clean up pool if it wasn't closed yet
        if pool:
            try:
                if hasattr(pool, '_state') and pool._state == mp.pool.RUN:
                    pool.close()
                    pool.join()
                elif hasattr(pool, '_state'):
                    # Pool was terminated, just join to ensure cleanup
                    pool.join()
            except Exception as e:
                print(f"Warning: Pool cleanup error: {e}")
    
    # Wait for monitoring to complete with timeout
    print(f"  Waiting for monitor thread to complete...")
    monitor_thread.join(timeout=5)  # Give it 5 seconds max
    if monitor_thread.is_alive():
        print(f"Warning: Monitor thread for batch {batch_num} did not terminate cleanly")
    else:
        print(f"  Monitor thread completed successfully")
    
    processing_time = time.time() - processing_start
    print(f"  Calculating results for batch {batch_num}...")
    
    # Count successful chunks
    successful_chunks = sum(1 for tokens in token_idss if len(tokens) > 0)
    print(f"  Batch {batch_num} completed in {processing_time:.2f} seconds ({successful_chunks}/{len(chunk_args)} chunks successful)")
    
    print(f"  Flattening tokens for batch {batch_num}...")
    # Flatten the token lists
    batch_tokens = [tid for tids in token_idss for tid in tids]
    print(f"  Batch {batch_num} produced {len(batch_tokens):,} tokens")
    
    # Force cleanup of multiprocessing manager
    print(f"  Cleaning up manager for batch {batch_num}...")
    try:
        manager.shutdown()
    except:
        pass
    
    # Force garbage collection
    del progress_dict, manager, chunk_args, token_idss
    gc.collect()
    
    print(f"  Returning from process_batch_file_with_timeout for batch {batch_num}")
    return batch_tokens

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Configuration
    input_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # save_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/TinyStoriesV2-train.npy"
    # vocab_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_vocab_ts.json"
    # merges_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_merges_ts.txt"
    # input_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/owt_train.txt"
    save_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/owt_train_batched.npy"
    vocab_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_vocab_owt.json"
    merges_path = "/home/azureuser/02-fun/cs336-assignment1-basics/data/train_bpe_merges_owt.txt"

    special_tokens = ["<|endoftext|>"]
    num_workers = 40  # Workers per batch
    num_batches = 80 # num of batches of files
    timeout_minutes = 40  # Timeout per batch in minutes
    
    # Create temporary directory for batch files
    temp_dir = tempfile.mkdtemp(prefix="tokenize_batches_")
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Initialize tokenizer
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=special_tokens,
        )

        # Split the file into batches
        total_start = time.time()
        batch_files, batch_info = split_file_into_batches(
            input_path, num_batches, special_tokens[0], temp_dir
        )
        
        split_time = time.time() - total_start
        print(f"File splitting completed in {split_time:.2f} seconds\n")
        
        # Process each batch sequentially
        all_tokens = []
        all_stuck_chunks = []
        
        for i, (batch_file, info) in enumerate(zip(batch_files, batch_info)):
            print(f"\n{'='*50}")
            print(f"Starting batch {i+1}/{num_batches}")
            print(f"{'='*50}")
            
            batch_tokens = process_batch_file_with_timeout(
                batch_file, tokenizer, num_workers, special_tokens[0], i + 1, timeout_minutes
            )
            
            print(f"  Extending main token list with {len(batch_tokens):,} tokens...")
            all_tokens.extend(batch_tokens)
            
            # Clean up this batch file immediately to save disk space
            print(f"  Removing batch file: {batch_file}")
            os.remove(batch_file)
            print(f"  Cleaned up batch file: {batch_file}")
            
            print(f"  Total tokens so far: {len(all_tokens):,}")
            
            # Force garbage collection between batches
            print(f"  Running garbage collection...")
            gc.collect()
            print(f"  Batch {i+1} fully completed, moving to next batch")
        
        total_time = time.time() - total_start
        print(f"\nAll batches completed in {total_time:.2f} seconds")
        print(f"Total tokens: {len(all_tokens):,}")
        
        # Check for stuck chunk logs
        stuck_log_files = [f for f in os.listdir('.') if f.startswith('stuck_chunks_batch_')]
        if stuck_log_files:
            print(f"\n⚠️  Found {len(stuck_log_files)} stuck chunk log files:")
            for log_file in stuck_log_files:
                print(f"  - {log_file}")
            print("Review these files to identify problematic chunks for manual inspection.")
        
        # Save the final result
        print(f"Saving tokens to {save_path}...")
        np.save(save_path, np.array(all_tokens, dtype=np.int32))
        print("Tokenization complete!")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
