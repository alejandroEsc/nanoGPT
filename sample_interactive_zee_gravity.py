import os
import torch
import sentencepiece as spm
from model import GPTConfig, GPT

# ----------- Config ------------------

out_dir = 'out-zee-gravity'  # change if needed
tokenizer_path = 'data/zee_gravity/physics_tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_prompt = "The solution to the equation is given by"  # or "" to sample freely
num_samples = 3
max_new_tokens = 256
temperature = 0.8
top_k = 100

exec(open('configurator.py').read())
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
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Encode the prompt
if start_prompt.strip():
    start_ids = sp.encode(start_prompt, out_type=int)
else:
    start_ids = [sp.bos_id()]  # or any valid starting token

input_ids = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

# --------- Interactive loop ----------

print("\n📐 PhysicsGPT Interactive Mode (type 'exit' to quit)\n")
while True:
    try:
        prompt = input("🧪 Prompt: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        # Encode the prompt
        input_ids = sp.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Generate
        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Decode
        output_ids = out[0].tolist()
        generated_text = sp.decode(output_ids)

        # Print only the new part
        new_text = generated_text[len(prompt):].strip()
        print(f"\n📤 Response:\n{new_text}\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        break