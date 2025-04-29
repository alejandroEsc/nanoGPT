import os
import re
import fitz  # PyMuPDF
import sentencepiece as spm
import numpy as np

pdf_name = "gravity_nutshell.pdf"
txt_name = "gravity_nutshell.txt"
pdf_file_path = os.path.join(os.path.dirname(__file__), pdf_name)
txt_file_path = os.path.join(os.path.dirname(__file__), txt_name)

train_file = "train.bin"
val_file = "val.bin"
train_file_path = os.path.join(os.path.dirname(__file__), train_file)
val_file_path = os.path.join(os.path.dirname(__file__), val_file)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text("text")  # Extract as plain text
        full_text += text + "\n"
    return full_text


if not os.path.exists(txt_file_path):
    physics_text = extract_text_from_pdf(pdf_file_path)
    # Optional: Clean a bit
    # physics_text = physics_text.replace("-\n", "")  # fix hyphenated line breaks
    # physics_text = physics_text.replace("\n", " ")

    # physics_text = re.sub(r'(\\[a-zA-Z]+)', r' \1 ', physics_text)
    #
    # # 3. Add spaces around math environments
    # # Example: "$ ... $" or "\[ ... \]" or "\begin{equation} ... \end{equation}"
    # physics_text = re.sub(r'(\$[^$]+\$)', r' \1 ', physics_text)
    # physics_text = re.sub(r'(\\\[.*?\\\])', r' \1 ', physics_text, flags=re.DOTALL)
    # physics_text = re.sub(r'(\\begin\{.*?\}.*?\\end\{.*?\})', r' \1 ', physics_text, flags=re.DOTALL)
    #
    # # 4. Remove multiple spaces
    # physics_text = re.sub(r'\s+', ' ', physics_text)

    # Save it to a .txt file (SentencePiece needs a file)
    with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(physics_text)


spm.SentencePieceTrainer.train(
    input=txt_file_path,    # input file
    model_prefix="physics_tokenizer",  # output name: will create physics_tokenizer.model + .vocab
    vocab_size=8000,               # typical size; adjust if you want
    model_type="unigram",           # you can also try "bpe", "word", or "char"
    character_coverage=1.0,         # 1.0 if full coverage (for English+math); lower for pure English
    bos_id=-1,                      # no special BOS token
    eos_id=-1,                      # no special EOS token
    pad_id=-1,                      # no special PAD token
    unk_id=0                        # keep UNK token for weird symbols
)

# Load the tokenizer
sp = spm.SentencePieceProcessor()
sp.load("physics_tokenizer.model")

# Load your cleaned text
with open(txt_file_path, "r", encoding="utf-8") as f:
    data = f.read()

# Encode text into integer token IDs
ids = sp.encode(data, out_type=int)

# Split into train/val (e.g., 90% train, 10% val)
n = len(ids)
train_ids = ids[:int(n * 0.9)]
val_ids = ids[int(n * 0.9):]

# Save as binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(train_file_path)
val_ids.tofile(val_file_path)
