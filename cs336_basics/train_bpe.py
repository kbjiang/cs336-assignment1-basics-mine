import os
import multiprocessing as mp
from typing import BinaryIO
import json
import time
import logging
from contextlib import contextmanager
import cProfile
import pstats
from io import StringIO
import argparse

from utils import save_vocab_and_merge, find_chunk_boundaries

import tracemalloc
from tqdm import tqdm
from collections import defaultdict

# Set up logging for profiling information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('bpe_cprofile.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Global file for all profiling results
PROFILE_OUTPUT_FILE = "bpe_complete_profile.txt"

@contextmanager
def CProfiler(name: str):
    """Context manager for profiling code sections with cProfile - writes to single file"""
    profiler = cProfile.Profile()
    logger.info(f"Starting cProfile: {name}")
    start_time = time.perf_counter()
    
    try:
        profiler.enable()
        yield
    finally:
        profiler.disable()
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Create stats object
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Append results to single profile file
        with open(PROFILE_OUTPUT_FILE, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"=== cProfile Results for: {name} ===\n")
            f.write(f"=== Total time: {elapsed:.4f} seconds ===\n")
            f.write(f"{'='*80}\n\n")
            
            # Redirect stats output to file
            old_stdout = stats.stream
            stats.stream = f
            stats.print_stats(30)  # Top 30 functions
            f.write("\n\n=== Callers for top functions ===\n")
            stats.print_callers(10)  # Top 10 functions with their callers
            stats.stream = old_stdout
            f.write(f"\n{'='*80}\n")
            f.write(f"=== End of {name} ===\n")
            f.write(f"{'='*80}\n\n")
        
        logger.info(f"Completed cProfile: {name} - Time: {elapsed:.4f} seconds - Results appended to: {PROFILE_OUTPUT_FILE}")

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
        pretoken = tuple(bytes([b]) for b in pretoken)
        # pretoken = tuple(list(bytes([idx]) for idx in list(pretoken)))
        pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1
    return pretoken_counts

def get_pair_counts(pretoken_counts):
    """Optimized version using dict.get() with default"""
    pair_counts = defaultdict(int)
    for byte_tup, byte_tup_count in pretoken_counts.items():
        for i in range(len(byte_tup) - 1):
            pair = (byte_tup[i], byte_tup[i+1])
            pair_counts[pair] += byte_tup_count
    return pair_counts

def get_max_pair(pair_counts):
    max_count = 0
    max_pair = None
    for pair, count in pair_counts.items():
        if count > max_count or (count == max_count and pair > max_pair):
            max_count = count
            max_pair = pair
    return max_pair

def merge_one_tuple(byte_tup, max_pair, max_pair_merged):
    """Optimized version using list operations instead of tuple concatenation"""
    if len(byte_tup) < 2:
        return byte_tup
    
    merged_byte_tup = b"".join(byte_tup)
    if max_pair_merged not in merged_byte_tup:
        return byte_tup
    
    result = []
    i = 0
    while i < len(byte_tup):
        if (i < len(byte_tup) - 1 and 
            byte_tup[i] == max_pair[0] and 
            byte_tup[i + 1] == max_pair[1]):
            # Merge the pair
            result.append(max_pair_merged)
            i += 2
        else:
            result.append(byte_tup[i])
            i += 1
    
    return tuple(result)

def merge_pretoken_counts(pretoken_counts, pair_counts, max_pair):
    """
    Optimized version that incrementally updates pair counts instead of 
    recalculating everything from scratch. Only pairs that overlap with 
    the merged pair need to have their counts updated.
    """
    # new_pretoken_counts = {}
    new_pretoken_counts = defaultdict(int)
    new_pair_counts = dict(pair_counts)  # Faster than .copy()
    max_pair_merged = b"".join(max_pair)
    
    # Remove the merged pair from pair counts
    new_pair_counts.pop(max_pair, None)
    
    # Filter: only process pretokens that could contain the target pair
    # This is the key optimization - avoid 90%+ of merge_one_tuple calls
    relevant_pretokens = []
    for byte_tup, count in pretoken_counts.items():
        if len(byte_tup) >= 2:
            # Quick scan for the pair
            for i in range(len(byte_tup) - 1):
                if byte_tup[i] == max_pair[0] and byte_tup[i + 1] == max_pair[1]:
                    relevant_pretokens.append((byte_tup, count))
                    break
            else:
                # No pair found, copy unchanged
                new_pretoken_counts[byte_tup] += count
        else:
            # Single byte, copy unchanged
            new_pretoken_counts[byte_tup] += count

    for byte_tup, byte_tup_count in relevant_pretokens:
        new_byte_tup = merge_one_tuple(byte_tup, max_pair, max_pair_merged)
        new_pretoken_counts[new_byte_tup] += byte_tup_count
        
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
    num_workers: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = [sp_tok.encode("utf-8") for sp_tok in special_tokens] + [bytes([i]) for i in range(256)]
    merges = []

    # pre-tokenization in parallel - NO PROFILING HERE (multiprocessing doesn't work with cProfile)
    pretoken_counts = {}
    logger.info("Starting file chunking and boundary detection")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_workers, special_tokens[0].encode("utf-8")
        )
        # Create arguments for each worker process
        chunk_args = [(input_path, start, end) 
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    logger.info(f"Starting parallel pretokenization ({num_workers} processes) - NO cProfile")
    # Process chunks in parallel - cProfile doesn't work here
    with mp.Pool(processes=num_workers) as pool:
        pretoken_countss = pool.map(process_chunk, chunk_args)
    logger.info("Completed parallel pretokenization")

    # with CProfiler("Merging pretoken counts from all processes"):
    for pretoken_counts_ in pretoken_countss:
        for k, v in pretoken_counts_.items():
            pretoken_counts[k] = pretoken_counts.get(k, 0) + v

    # Optimized BPE training with incremental updates
    num_merges = vocab_size - len(vocab)
    
    # with CProfiler("Initial pair counts calculation"):
    # Build initial pair counts index
    pair_counts = get_pair_counts(pretoken_counts)
    
    logger.info(f"Starting BPE training with {num_merges} merges")
    
    # Profile the BPE training loop
    # with CProfiler(f"BPE training loop ({num_merges} merges)"):
    for _ in tqdm(range(num_merges), total=num_merges):
        if not pair_counts:
            break
            
        # Find the most frequent pair
        max_pair = get_max_pair(pair_counts)
        
        # Create new merged token and add to vocab
        new_token = b"".join(max_pair)
        vocab.append(new_token)
        merges.append(max_pair)
        
        # Update pretoken_counts and pair_counts incrementally
        pretoken_counts, pair_counts = merge_pretoken_counts(
            pretoken_counts, pair_counts, max_pair,
        )

    # with CProfiler("Converting vocab to final format"):
    vocab = {i: v for i, v in enumerate(vocab)}
    return vocab, merges

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input text file')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Target vocabulary size (default: 10000)')
    parser.add_argument('--special_tokens', nargs='+', default=["<|endoftext|>"],
                        help='List of special tokens (default: ["<|endoftext|>"])')
    parser.add_argument('--num_workers', type=int, default=40,
                        help='Number of processes for parallel processing (default: 40)')
    parser.add_argument('--vocab_path', type=str, default="train_bpe_vocab.json",
                        help='Output path for vocabulary file (default: train_bpe_vocab.json)')
    parser.add_argument('--merges_path', type=str, default="train_bpe_merges.txt",
                        help='Output path for merges file (default: train_bpe_merges.txt)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use parsed arguments
    input_path = args.input_path
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    num_workers = args.num_workers
    vocab_path = args.vocab_path
    merges_path = args.merges_path
    
    logger.info("Starting BPE training script with cProfile")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Target vocab size: {vocab_size}")
    logger.info(f"Special tokens: {special_tokens}")
    logger.info(f"Number of processes: {num_workers}")
    logger.info(f"Vocab output path: {vocab_path}")
    logger.info(f"Merges output path: {merges_path}")
    
    # Initialize the profile output file
    with open(PROFILE_OUTPUT_FILE, 'w') as f:
        f.write(f"BPE Training Complete Profile Results\n")
        f.write(f"=====================================\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Target vocab size: {vocab_size}\n")
        f.write(f"Number of processes: {num_workers}\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=====================================\n\n")
    
    logger.info("Starting memory tracking setup")
    # tracemalloc.start()
    # print("Tracemalloc started.")
    
    # Run BPE training without nested profilers
    vocab, merges = train_bpe(
        input_path, vocab_size, special_tokens, num_workers
    )
    
    logger.info("Starting memory usage reporting")
    current, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / (1024 * 1024)
    print(f"Peak memory usage: {peak_mb:.2f} MB")
    logger.info(f"Peak memory usage: {peak_mb:.2f} MB")
    # tracemalloc.stop()

    save_vocab_and_merge(vocab, merges, vocab_path, merges_path)
    
    logger.info("BPE training script with cProfile completed successfully")
    logger.info(f"Complete profile results saved to: {PROFILE_OUTPUT_FILE}")
    logger.info(f"Vocabulary saved to: {vocab_path}")
    logger.info(f"Merges saved to: {merges_path}")


# python train_bpe.py --input_path /home/azureuser/02-fun/cs336-assignment1-basics/data/owt_train.txt --vocab_size 32000 --num_workers 16 --vocab_path train_bpe_vocab_owt.json --merges_path train_bpe_merges_owt.txt