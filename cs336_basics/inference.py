from jaxtyping import Int
from model import *
from nn_utils import *
from data import *
from tokenizer import *
from serialization import *

def get_next_id(model, current_prompt, temperature=1.0, top_p=None):
    if len(current_prompt.size()) == 1:
        current_prompt = current_prompt.unsqueeze(0)
    with torch.no_grad():
        logits = model(current_prompt)[0]
        probs = softmax(logits[-1, :], dim=-1, temperature=temperature)
        indices = range(len(probs))
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
    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=1e4, help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    
    # Data and model paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (.npy file)")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to evaluation data (.npy file)")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file")
    parser.add_argument("--checkpoint_path", type=str, default="/home/azureuser/02-fun/cs336-assignment1-basics/data/checkpoint.pt", help="Path to save checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # training code starts here
    vocab, _ = get_vocab_and_merges_from_files(args.vocab_path, args.merges_path)

    from serialization import load_checkpoint
    load_checkpoint("../data/checkpoint.pt", model, None)
