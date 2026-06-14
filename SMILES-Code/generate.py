
import argparse
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from tokenizer import SMILES_Tokenizer
from model import bulid_transformer


# Validity Check

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def is_valid_smiles(smi: str) -> bool:
        return smi and Chem.MolFromSmiles(smi) is not None

    def canonicalize(smi: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None

    RDKIT_AVAILABLE = True

except ImportError:
    def is_valid_smiles(smi: str) -> bool:
        if not smi:
            return False
        return smi.count("(") == smi.count(")") and smi.count("[") == smi.count("]")

    def canonicalize(smi: str) -> Optional[str]:
        return smi

    RDKIT_AVAILABLE = False
    print("[generate] RDKit not found - using basic validity checks only. Install RDKit for better validation and canonicalization.")


# Decoding:

@torch.no_grad()
def greedy_decode(
    model, tokenizer: SMILES_Tokenizer, max_len: int, device: str,
) -> str:
    src = torch.tensor([[tokenizer.sos_id]], dtype=torch.long, device=device)
    dec_input = src.clone()
    output_ids = []

    for _ in range(max_len):
        src_mask = (src != tokenizer.pad_id).unsqueeze(0).unsqueeze(0).int()
        dec_mask = (
            torch.triu(torch.ones(1, dec_input.size(1), dec_input.size(
                1)), diagonal=1).type(torch.int).to(device) == 0
        )
        enc_output = model.encode(src, src_mask)
        dec_output = model.decode(enc_output, src_mask, dec_input, dec_mask)
        logits = model.project(dec_output[:, -1, :])
        next_id = logits.argmax(dim=-1).item()

        if next_id == tokenizer.eos_id:
            break
        output_ids.append(next_id)
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        src = torch.cat([src, next_token], dim=1)
        dec_input = torch.cat([dec_input, next_token], dim=1)

    return tokenizer.decode(output_ids)


@torch.no_grad()
def temperature_sample(
    model, tokenizer: SMILES_Tokenizer, max_len: int, device: str, tempetature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
) -> str:
    src = torch.tensor([[tokenizer.sos_id]], dtype=torch.long, device=device)
    dec_input = src.clone()
    output_ids = []

    for _ in range(max_len):
        src_mask = (src != tokenizer.pad_id).unsqueeze(0).unsqueeze(0).int()
        dec_mask = (
            torch.triu(torch.ones(1, dec_input.size(1), dec_input.size(
                1)), diagonal=1).type(torch.int).to(device) == 0
        )
        enc_output = model.encode(src, src_mask)
        dec_output = model.decode(enc_output, src_mask, dec_input, dec_mask)
        logits = model.project(dec_output[:, -1, :]) / max(tempetature, 1e-8)

        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            kth = torch.topk(logits, top_k_val).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, -float("inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = -float("inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == tokenizer.eos_id:
            break

        output_ids.append(next_id)
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        src = torch.cat([src, next_token], dim=1)
        dec_input = torch.cat([dec_input, next_token], dim=1)

    return tokenizer.decode(output_ids)


# Batch generation

@torch.no_grad()
def sample_molecules(
    model, tokenizer: SMILES_Tokenizer, n: int, seq_len: int, device: str, temperature: float = 1.0, top_k: Optional[int] = 50, top_p: Optional[float] = 0.9,
) -> list[dict]:

    model.eval()
    results = []

    for _ in range(n):
        smi = temperature_sample(
            model, tokenizer, seq_len, device, temperature, top_k, top_p,
        )
        valid = is_valid_smiles(smi)
        canon = canonicalize(smi) if valid else None
        results.append({"smiles": smi, "valid": valid, "canonical": canon})

    model.train()
    return results


def load_model_from_checkpoint(checkpoint_path: str, tokenizer: SMILES_Tokenizer, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
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
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate molecules ")
    parser.add_argument("-CheckPoint", required=True)
    parser.add_argument("-Tokenizer", default="tokenizer.json")
    parser.add_argument("-num", default=20, type=int)
    parser.add_argument("-temperature", default=1.0, type=float)
    parser.add_argument("-top_k", default=50, type=int)
    parser.add_argument("-top_p", default=0.9, type=float)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok = SMILES_Tokenizer()
    tok.load(args.Tokenizer)
    model, cfg = load_model_from_checkpoint(args.CheckPoint, tok, device)
    print(f"MODEL loadeded")

    print(
        f"\n Generating {args.num} molecules (temperature={args.temperature}) .... \n")
    results = sample_molecules(
        model, tok, args.num, cfg["seq_len"], device, args.temperature, args.top_k, args.top_p
    )

    valid_smiles = [r for r in results if r["valid"]]
    print(f"{'#':<4} {'SMILES':<55} {'Status'}")
    print("-" * 75)
    for i, rw in enumerate(results, 1):
        tag = "v valid" if rw["valid"] else "x invalid"
        print(f"{i:<4} {rw['smiles']:<55} {tag}")

    validity_rate = len(valid_smiles) / len(results) * 100
    print(
        f"\n Validity: {len(valid_smiles)} / {len(results)} ({validity_rate: .1f}%)")

    unique = {r["canonical"] for r in valid_smiles if r["canonical"]}
    print(f"Unique valid: {len(unique)}")
