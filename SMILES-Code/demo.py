
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ZINC_Dataset, load_Sime_data, DEMO
from model import bulid_transformer
from tokenizer import SMILES_Tokenizer
from generate import sample_molecules

CFG = dict(
    seq_len=64,
    d_model=128,
    N=2,
    h=4,
    d_ff=256,
    dropout=0.1,
    batch_size=4,
    epochs=40,
    lr=5e-4,
    num_gen=8,
    temperature=1.0,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# DATA & TOKENIZER
smiles = list(DEMO)
tokenizer = SMILES_Tokenizer()
tokenizer.vocab_bulider(smiles)
print(f"Vocabulary size: {tokenizer.Size()}")

dataset = ZINC_Dataset(smiles, tokenizer, CFG["seq_len"])
loader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True)


# Model:

vs = tokenizer.Size()
model = bulid_transformer(
    src_vocab_size=vs,
    tgt_vocab_size=vs,
    src_seq_len=CFG["seq_len"],
    tgt_seq_len=CFG["seq_len"],
    d_model=CFG["d_model"],
    N=CFG["N"],
    h=CFG["h"],
    d_ff=CFG["d_ff"],
    dropout=CFG["dropout"],
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


# Training

optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_id, label_smoothing=0.1)

for epoch in range(CFG["epochs"]):
    model.train()
    total = 0.0
    for batch in loader:
        enc_input = batch["encoder_input"].to(device)
        dec_input = batch["decoder_input"].to(device)
        label = batch["label"].to(device)
        enc_mask = batch["encoder_Mask"].to(device)
        dec_mask = batch["decoder_Mask"].to(device)

        enc_output = model.encode(enc_input, enc_mask)
        dec_output = model.decode(enc_output, enc_mask, dec_input, dec_mask)
        logits = model.project(dec_output)

        loss = loss_fn(logits.view(-1, vs), label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()

    avg = total / len(loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch +1:02d}/{CFG['epochs']} loss={avg:.4f}")


# Geneartion

print("\n -------------GENERATE RIRI MOLECULES----------")
results = sample_molecules(
    model, tokenizer, CFG["num_gen"],
    CFG["seq_len"], device, CFG["temperature"]
)

for i, r in enumerate(results, 1):
    tag = "v" if r["valid"] else "x"
    print(f" {tag} {r['smiles']}")

valid = sum(1 for r in results if r["valid"])
print(
    f"\n Validity : {valid} / {len(results)} ({valid/len(results)* 100:.0f}%)")
