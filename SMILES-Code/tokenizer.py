
import re
import json
from pathlib import Path


# CODE:
# TOKENIZER : SPLIT SMILES INTO UNITE
# ENCODE: convert this unit(str) into int AI can understand it
# DECODE: rewrite this int to SMILES str so human can read it and understand it
# SAVE&LOAD: SAVE AND GET DATA FROM THE FILE INSTAD OF BULID THE VOCAB IN EACH TIME WE RUN THE CODE AND TO KEEP THE RELATION AS WE TRAIN THE MODEL


Smiles_Token_Re = re.compile(
    # الذرات بين أقواس مربعة
    r"(\[[^\]]+\]"
    # ذرات بحرفين
    r"|Br|Cl|Si|Se|se|As"
    # كيرالية وحلقات
    r"|@@|%\d{2}"
    # أي حرف منفرد
    r"|.)"
)


Special_Tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]


class SMILES_Tokenizer:
    def __init__(self):
        # str to int
        self.token_to_id: dict[str, int] = {}
        # int to str
        self.id_to_token: dict[int, str] = {}

    # Vocabulary
    def vocab_bulider(self, smiles_list: list[str]) -> None:
        # نستخدم سيت لاننا مانبغى تكرار
        tokens = set()
        for smi in smiles_list:
            tokens.update(Smiles_Token_Re.findall(smi))

        vocab = Special_Tokens+sorted(tokens)

        # All str to int:
        self.token_to_id = {t: i for i, t in enumerate(vocab)}

        # All int to str:
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def Size(self) -> int:
        return len(self.token_to_id)

# Special tokens id

    @property
    def pad_id(self): return self.token_to_id['[PAD]']
    @property
    def sos_id(self): return self.token_to_id["[SOS]"]
    @property
    def eos_id(self): return self.token_to_id["[EOS]"]
    @property
    def unk_id(self): return self.token_to_id["[UNK]"]

    def tokenize(self, smiles: str) -> list[str]:
        return Smiles_Token_Re.findall(smiles)

    def encoder(self, smiles: str, max_len: str | None = None) -> list[int]:
        tokens = [self.sos_id]
        # يمر على القاموس ياخد قيمة تي لو مالقيها يعطي غير معروف  ويضيف كل مرة القيمة للتوكين
        for t in self.tokenize(smiles):
            tokens.append(self.token_to_id.get(t, self.unk_id))
        tokens.append(self.eos_id)

        # لو المستخدم ماأعطى طول نتجاهل لو لأ ندخل لهذا الشرط نكمل
        if max_len is not None:
            # الشرط الأول لو طول التوكين اكثر من الماكس لينث نقص القائمة
            if len(tokens) > max_len:
                return tokens[:max_len]
            # لو لأ لو القائمة أقصر احنا مثلا نبغى 10 وهي طولها 6 نضيف بادينق
            tokens += [self.pad_id] * (max_len - len(tokens))
        return tokens

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        tokens = [self.id_to_token.get(i, '[UNK]') for i in ids]
        # إزالة الرموز الخاصة
        if strip_special:
            tokens = [t for t in tokens if t not in Special_Tokens]
        # جمع العناصر للحصول على القائمة النهائية
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as SmiFile:
            json.dump(self.token_to_id, SmiFile, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as SmiFile:
            self.token_to_id = json.load(SmiFile)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
