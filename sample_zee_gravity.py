import os
import torch
import sentencepiece as spm
from model import GPTConfig, GPT

# ----------- Config ------------------

out_dir = 'out-zee-gravity'  # change if needed
tokenizer_path = 'physics_tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_prompt = "The solution to the equation is given by"  # or "" to sample freely
num_samples = 3
max_new_tokens = 256
temperature = 0.9
top_k = 100

# -------------------------------------

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)

# Load model checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# Rebuild the GPT model
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Encode the prompt
if start_prompt.strip():
    start_ids = sp.encode(start_prompt, out_type=int)
else:
    start_ids = [sp.bos_id()]  # or any valid starting token

input_ids = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

# Generate text
for i in range(num_samples):
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    out_ids = out[0].tolist()
    decoded = sp.decode(out_ids)
    print(f"\n=== Sample {i+1} ===\n{decoded.strip()}\n")