
import csv
import random
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset


DEMO = [
    "CC(=O)Oc1ccccc1C(=O)O",               # aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",        # caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",       # ibuprofen
    "OC(=O)c1ccccc1O",                      # salicylic acid
    "c1ccc2ccccc2c1",                       # naphthalene
    "C1=CC=C(C=C1)O",                       # phenol
    "CC(=O)N1CCC[C@H]1C(=O)O",
    "CN(C)c1ccc(cc1)C=CC(=O)O",
    "O=C(O)c1ccc(F)cc1",
    "CC1=CC=C(C=C1)S(N)(=O)=O",
    "Nc1ccc(cc1)C(=O)O",
    "ClC(Cl)(Cl)c1ccccc1",
    "O=C(N)c1ccccc1",
    "c1ccncc1",
]


def load_Sime_data(file_path: str, max_samples: Optional[int] = None, shuffle: bool = True, seed: int = 42) -> list[str]:

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"csu File not Found: {file_path} \n"
            f"Make sure file exist in this location"
        )
    smiles_list = []

    with open(path, newline="", encoding="utf-8") as SmiFile:
        reader = csv.DictReader(SmiFile)
        for row in reader:
            smi = row["smiles"].strip()
            if smi:
                smiles_list.append(smi)

    if shuffle:
        random.seed(seed)
        random.shuffle(smiles_list)

    if max_samples is not None:
        smiles_list = smiles_list[:max_samples]

    print(f"[dataset] Loaded {len(smiles_list):,} molecules.")
    return smiles_list


class ZINC_Dataset(Dataset):
    def __init__(self, smiles_list: list[str], tokenizer, seq_len: int):
        self.data = smiles_list
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx: int):
        smi = self.data[indx]
        tok = self.tokenizer
        tokens = tok.tokenize(smi)
        token_id = [tok.token_to_id.get(t, tok.unk_id) for t in tokens]
        max_tok = self.seq_len - 2
        token_id = token_id[:max_tok]
        pad_count = self.seq_len - len(token_id) - 2

        enc_input = torch.tensor(
            [tok.sos_id] + token_id + [tok.pad_id] * (pad_count + 1), dtype=torch.long,)

        dec_input = torch.tensor(
            [tok.sos_id] + token_id + [tok.pad_id] * (pad_count + 1), dtype=torch.long, )

        label = torch.tensor(
            token_id + [tok.eos_id] + [tok.pad_id] * (pad_count + 1), dtype=torch.long,)

        enc_mask = (enc_input != tok.pad_id).unsqueeze(0).unsqueeze(0).int()
        dec_mask = ((dec_input != tok.pad_id).unsqueeze(
            0).unsqueeze(0).int() & causal_mask(self.seq_len))

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "label": label,
            "encoder_Mask": enc_mask,
            "decoder_Mask": dec_mask,
            "Smiles": smi,
        }


def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).int()
    return mask == 0
