# smiles-code

SMILES molecular generation with a Transformer (PyTorch).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy rdkit
```

## Run

```bash
cd SMILES-Code
python demo.py
python train.py --max_samples 10000 --epochs 50
python generate.py -CheckPoint checkpoint.pt -Tokenizer tokenizer.json -num 20
```
