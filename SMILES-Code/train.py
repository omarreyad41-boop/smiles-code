import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ZINC_Dataset, load_Sime_data
from model import bulid_transformer
from tokenizer import SMILES_Tokenizer
from generate import sample_molecules

CFG = dict(
    seq_len=64,
    d_model=256,
    N=4,
    h=8,
    d_ff=512,
    dropout=0.1,
    batch_size=32,
    epochs=50,
    lr=5e-4,
    num_gen=10,
    temperature=1.0,
)


def main():
    parser = argparse.ArgumentParser(description="Train SMILES transformer")
    parser.add_argument("--data", default="zinc_250k.csv")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(CFG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_path = Path(__file__).parent / args.data
    smiles = load_Sime_data(str(data_path), max_samples=args.max_samples)

    tokenizer = SMILES_Tokenizer()
    tokenizer.vocab_bulider(smiles)
    tokenizer.save(args.tokenizer)
    print(f"Vocabulary size: {tokenizer.Size()}")

    dataset = ZINC_Dataset(smiles, tokenizer, cfg["seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    vs = tokenizer.Size()
    model = bulid_transformer(
        src_vocab_size=vs,
        tgt_vocab_size=vs,
        src_seq_len=cfg["seq_len"],
        tgt_seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        N=cfg["N"],
        h=cfg["h"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=0.1)

    for epoch in range(cfg["epochs"]):
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:02d}/{cfg['epochs']} loss={avg:.4f}")

    torch.save(
        {
            "config": cfg,
            "model_state": model.state_dict(),
            "epoch": cfg["epochs"],
        },
        args.checkpoint,
    )
    print(f"\nSaved checkpoint: {args.checkpoint}")
    print(f"Saved tokenizer: {args.tokenizer}")

    print("\n-------------GENERATE MOLECULES-------------")
    results = sample_molecules(
        model, tokenizer, cfg["num_gen"],
        cfg["seq_len"], device, cfg["temperature"],
    )

    for i, r in enumerate(results, 1):
        tag = "v" if r["valid"] else "x"
        print(f" {tag} {r['smiles']}")

    valid = sum(1 for r in results if r["valid"])
    print(
        f"\n Validity : {valid} / {len(results)} ({valid / len(results) * 100:.0f}%)")


if __name__ == "__main__":
    main()
