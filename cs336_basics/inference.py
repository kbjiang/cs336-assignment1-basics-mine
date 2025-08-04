import argparse
import torch
from jaxtyping import Int
from model import *
# from nn_utils import *
# from data import *
from tokenizer import *
from serialization import load_checkpoint

def get_next_id(model, current_prompt, temperature=1.0, top_p=None):
    if len(current_prompt.size()) == 1:
        current_prompt = current_prompt.unsqueeze(0)
    with torch.no_grad():
        logits = model(current_prompt)[0]
        probs = softmax(logits[-1, :], dim=-1, temperature=temperature)
        indices = torch.arange(len(probs)).to(probs.device)
        if top_p:
            probs_, indices = torch.sort(probs, descending=True)

            p = 0
            i = 0
            while p < top_p:
                p += probs_[i]
                i += 1

            probs = probs_[:i] / torch.sum(probs_[:i])

        next_id = probs.multinomial(num_samples=1, replacement=True)
        return indices[next_id]

def decoding(model, current_prompt: Int[torch.Tensor, "length"], max_new_tokens: int, eos_id: int, temperature: float = 1.0, top_p: float | None = None):
    count = 0
    while count < max_new_tokens:
        next_id = get_next_id(model, current_prompt, temperature, top_p)
        current_prompt = torch.cat((current_prompt, next_id))
        count += 1
        if next_id == eos_id: 
            break
    return current_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation with transformer model")
    
    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward dimension")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=1e4, help="RoPE theta parameter")
    
    # Tokenizer and model paths
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file")
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="/home/azureuser/02-fun/cs336-assignment1-basics/data/checkpoint_ts.pt",
        help="Path to save checkpoint"
    )

    # Generation parameters
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling parameter")
    
    args = parser.parse_args()
    
    EOS_TOKEN = "<|endoftext|>"
    SPECIAL_TOKENS = [EOS_TOKEN]

    # Load tokenizer first to get vocab size
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, SPECIAL_TOKENS)
    vocab_size = len(tokenizer.vocab)
    
    # Load model
    model = transformer_lm(
        d_model = args.d_model,
        d_ff = args.d_ff,
        num_heads = args.num_heads,
        rope_theta = args.rope_theta,
        num_layers = args.num_layers,
        vocab_size = vocab_size,
        context_length = args.context_length
    )
    load_checkpoint(args.checkpoint_path, model, None)
    model.to(args.device)
    
    # Get user input for initial prompt
    print("Enter your initial prompt:")
    user_input = input("> ")
    
    if not user_input.strip():
        raise ValueError("Initial prompt cannot be empty!")

    # Tokenize the input
    current_prompt = torch.tensor(tokenizer.encode(user_input), dtype=torch.int32).to(args.device)
    
    # Get EOS token ID
    try:
        eos_id = tokenizer.encode(EOS_TOKEN)[0]
    except Exception as e:
        raise ValueError(f"EOS not valid: {e}")

    # Generate text
    print("Generated text:")
    print("-" * 50)

    generated_tokens = decoding(
        model=model,
        current_prompt=current_prompt,
        max_new_tokens=args.max_new_tokens,
        eos_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Decode and print the result
    generated_text = tokenizer.decode(generated_tokens.cpu().tolist())
    print(generated_text)




