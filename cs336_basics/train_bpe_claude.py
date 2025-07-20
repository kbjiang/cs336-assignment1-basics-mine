import os
import multiprocessing as mp
from typing import Dict, List, Tuple, Set
import json
import time
import logging
from collections import defaultdict, Counter
import argparse
import regex as re
from tqdm import tqdm
from utils import save_vocab_and_merge

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-tokenization pattern (GPT-4 style)
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(args: Tuple[str, int, int, str]) -> Dict[Tuple[bytes, ...], int]:
    """Process a chunk of the file and return pretoken counts."""
    file_path, start, end, special_token = args
    
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    pretoken_counts = defaultdict(int)
    
    for doc in chunk.split(special_token):
        if not doc.strip():
            continue
            
        # Use regex to find all pretokens
        for match in re.finditer(PATTERN, doc):
            pretoken_bytes = match.group().encode("utf-8")
            # Convert to tuple of individual bytes
            pretoken_tuple = tuple(bytes([b]) for b in pretoken_bytes)
            pretoken_counts[pretoken_tuple] += 1
            
    return dict(pretoken_counts)

def find_chunk_boundaries(file_path: str, num_chunks: int, special_token: str) -> List[int]:
    """Find optimal chunk boundaries that don't split documents."""
    with open(file_path, "rb") as f:
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        
    chunk_size = file_size // num_chunks
    boundaries = [0]
    
    with open(file_path, "rb") as f:
        for i in range(1, num_chunks):
            # Start at approximate position
            pos = i * chunk_size
            f.seek(pos)
            
            # Find next document boundary
            separator = special_token.encode("utf-8")
            
            # Read ahead to find separator
            buffer = f.read(min(10000, file_size - pos))
            sep_pos = buffer.find(separator)
            
            if sep_pos != -1:
                boundaries.append(pos + sep_pos + len(separator))
            else:
                boundaries.append(pos)
    
    boundaries.append(file_size)
    return boundaries

def get_initial_pretoken_counts(file_path: str, special_tokens: List[str], num_workers: int = 1) -> Dict[Tuple[bytes, ...], int]:
    """Get initial pretoken counts using parallel processing."""
    logger.info("Finding chunk boundaries...")
    boundaries = find_chunk_boundaries(file_path, num_workers, special_tokens[0])
    
    # Create chunk arguments
    chunk_args = [(file_path, boundaries[i], boundaries[i+1], special_tokens[0]) 
                 for i in range(len(boundaries)-1)]
    
    logger.info(f"Processing {len(chunk_args)} chunks with {num_workers} workers...")
    
    # Process chunks in parallel
    with mp.Pool(processes=num_workers) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)
    
    # Merge results
    logger.info("Merging chunk results...")
    final_counts = defaultdict(int)
    for chunk_counts in chunk_results:
        for pretoken, count in chunk_counts.items():
            final_counts[pretoken] += count
    
    return dict(final_counts)

def build_pair_index(pretoken_counts: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]]:
    """Build an index mapping pairs to pretokens that contain them."""
    pair_to_pretokens = defaultdict(set)
    
    for pretoken in pretoken_counts:
        if len(pretoken) >= 2:
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_to_pretokens[pair].add(pretoken)
    
    return dict(pair_to_pretokens)

def get_pair_counts(pretoken_counts: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """Get counts for all adjacent pairs."""
    pair_counts = defaultdict(int)
    
    for pretoken, count in pretoken_counts.items():
        if len(pretoken) >= 2:
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counts[pair] += count
    
    return dict(pair_counts)

def get_max_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    """Find the most frequent pair (with tie-breaking)."""
    if not pair_counts:
        return None
    
    # Use max with custom key for tie-breaking
    return max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

def merge_pretoken(pretoken: Tuple[bytes, ...], target_pair: Tuple[bytes, bytes], merged_token: bytes) -> Tuple[bytes, ...]:
    """Merge a specific pair in a pretoken."""
    if len(pretoken) < 2:
        return pretoken
    
    # Quick check: does pretoken contain the target pair?
    contains_pair = False
    for i in range(len(pretoken) - 1):
        if pretoken[i] == target_pair[0] and pretoken[i + 1] == target_pair[1]:
            contains_pair = True
            break
    
    if not contains_pair:
        return pretoken
    
    # Perform the merge
    result = []
    i = 0
    while i < len(pretoken):
        if (i < len(pretoken) - 1 and 
            pretoken[i] == target_pair[0] and 
            pretoken[i + 1] == target_pair[1]):
            result.append(merged_token)
            i += 2
        else:
            result.append(pretoken[i])
            i += 1
    
    return tuple(result)

def update_pair_counts_efficiently(
    pretoken_counts: Dict[Tuple[bytes, ...], int],
    pair_counts: Dict[Tuple[bytes, bytes], int],
    pair_index: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]],
    target_pair: Tuple[bytes, bytes]
) -> Tuple[Dict[Tuple[bytes, ...], int], Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]]]:
    """Efficiently update counts after merging a pair."""
    
    merged_token = b"".join(target_pair)
    new_pretoken_counts = {}
    new_pair_counts = dict(pair_counts)
    new_pair_index = defaultdict(set)
    
    # Copy existing pair index
    for pair, pretokens in pair_index.items():
        new_pair_index[pair] = pretokens.copy()
    
    # Remove the merged pair from counts and index
    new_pair_counts.pop(target_pair, None)
    new_pair_index.pop(target_pair, None)
    
    # Get pretokens that contain the target pair
    affected_pretokens = pair_index.get(target_pair, set())
    
    # Process unaffected pretokens (just copy)
    for pretoken, count in pretoken_counts.items():
        if pretoken not in affected_pretokens:
            new_pretoken_counts[pretoken] = count
    
    # Process affected pretokens
    for pretoken in affected_pretokens:
        count = pretoken_counts[pretoken]
        new_pretoken = merge_pretoken(pretoken, target_pair, merged_token)
        new_pretoken_counts[new_pretoken] = new_pretoken_counts.get(new_pretoken, 0) + count
        
        # Update pair counts: remove old pairs, add new pairs
        
        # Remove old pair counts
        for i in range(len(pretoken) - 1):
            old_pair = (pretoken[i], pretoken[i + 1])
            if old_pair in new_pair_counts:
                new_pair_counts[old_pair] -= count
                if new_pair_counts[old_pair] <= 0:
                    new_pair_counts.pop(old_pair)
            
            # Update pair index
            if old_pair in new_pair_index:
                new_pair_index[old_pair].discard(pretoken)
                if not new_pair_index[old_pair]:
                    new_pair_index.pop(old_pair, None)
        
        # Add new pair counts
        for i in range(len(new_pretoken) - 1):
            new_pair = (new_pretoken[i], new_pretoken[i + 1])
            new_pair_counts[new_pair] = new_pair_counts.get(new_pair, 0) + count
            new_pair_index[new_pair].add(new_pretoken)
    
    return new_pretoken_counts, new_pair_counts, dict(new_pair_index)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_workers: int = 1
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train BPE tokenizer."""
    logger.info("Starting BPE training...")
    
    # Initialize vocab with special tokens + all bytes
    vocab = [tok.encode("utf-8") for tok in special_tokens]
    vocab.extend([bytes([i]) for i in range(256)])
    merges = []
    
    # Step 1: Get initial pretoken counts
    logger.info("Step 1: Getting initial pretoken counts...")
    pretoken_counts = get_initial_pretoken_counts(input_path, special_tokens, num_workers)
    logger.info(f"Found {len(pretoken_counts)} unique pretokens")
    
    # Step 2: Build initial pair counts and index
    logger.info("Step 2: Building initial pair counts and index...")
    pair_counts = get_pair_counts(pretoken_counts)
    pair_index = build_pair_index(pretoken_counts)
    logger.info(f"Found {len(pair_counts)} unique pairs")
    
    # Step 3: Perform merges
    num_merges = vocab_size - len(vocab)
    logger.info(f"Step 3: Performing {num_merges} merges...")
    
    for merge_idx in tqdm(range(num_merges), desc="BPE merges"):
        if not pair_counts:
            logger.warning(f"No more pairs to merge at iteration {merge_idx}")
            break
        
        # Find most frequent pair
        max_pair = get_max_pair(pair_counts)
        if max_pair is None:
            break
        
        # Create merged token
        merged_token = b"".join(max_pair)
        vocab.append(merged_token)
        merges.append(max_pair)
        
        # Update counts efficiently
        pretoken_counts, pair_counts, pair_index = update_pair_counts_efficiently(
            pretoken_counts, pair_counts, pair_index, max_pair
        )
    
    # Convert vocab to final format
    vocab_dict = {i: token for i, token in enumerate(vocab)}
    
    logger.info(f"Training completed. Final vocab size: {len(vocab_dict)}")
    return vocab_dict, merges

def main():
    parser = argparse.ArgumentParser(description='Optimal BPE Tokenizer Training')
    parser.add_argument('--input_path', type=str, required=True, help='Input text file path')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Target vocabulary size')
    parser.add_argument('--special_tokens', nargs='+', default=["<|endoftext|>"], help='Special tokens')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(), help='Number of worker processes')
    parser.add_argument('--vocab_path', type=str, default="bpe_vocab.json", help='Output vocabulary path')
    parser.add_argument('--merges_path', type=str, default="bpe_merges.txt", help='Output merges path')
    
    args = parser.parse_args()
    
    # Train
    start_time = time.time()
    vocab, merges = train_bpe(
        args.input_path, 
        args.vocab_size, 
        args.special_tokens, 
        args.num_workers
    )
    end_time = time.time()
    
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    save_vocab_and_merge(vocab, merges, args.vocab_path, args.merges_path)
    logger.info(f"Saved vocabulary to {args.vocab_path}")
    logger.info(f"Saved merges to {args.merges_path}")

if __name__ == "__main__":
    main()