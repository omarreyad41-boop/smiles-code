# الأخطاء المكتشفة في مشروع SMILES

## model.py

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 1 | `nn.parameter` (حرف p صغير) | `nn.Parameter` | ✅ مصلّح |
| 2 | `forwoard` بدل `forward` في `InputEmbedding` | `forward` | ✅ مصلّح |
| 3 | `nn.Droupout` | `nn.Dropout` | ✅ مصلّح |
| 4 | `softmax(dim=1)` في Attention | `softmax(dim=-1)` | ✅ مصلّح |

---

## tokenizer.py

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 5 | `dict[str:int]` و `dict[int:str]` | `dict[str, int]` و `dict[int, str]` | ✅ مصلّح |
| 6 | Regex مكسور: `[^\]] +` ومسافات زائدة في `% \d{2}` و ` Br` | `[^\]]+` و `%\d{2}` بدون مسافات | ✅ مصلّح |
| 7 | `id_to_token` معكوس `{t: i}` | `{i: t for t, i in ...}` | ✅ مصلّح |
| 8 | `json.dumb(self, self.token_to_id, ...)` | `json.dump(self.token_to_id, SmiFile, ...)` | ✅ مصلّح |

---

## generate.py

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 9 | استدعاء `model.encoder` / `model.decoder` بدون embedding | `model.encode` / `model.decode` | ✅ مصلّح |
| 10 | `- model.project(...)` في `greedy_decode` | حذف الإشارة السالبة | ✅ مصلّح |
| 11 | `return` داخل حلقة `for` في `greedy_decode` | نقل `return` بعد انتهاء الحلقة | ✅ مصلّح |
| 12 | استخدام `temperature` والباراميتر اسمه `tempetature` | استخدام `tempetature` | ✅ مصلّح |
| 13 | `src_seg_len` / `tgt_seg_len` في `load_model_from_checkpoint` | `src_seq_len` / `tgt_seq_len` | ✅ مصلّح |
| 14 | `args.tokenizer` | `args.Tokenizer` | ✅ مصلّح |
| 15 | `model` و `cfg` غير معرّفين في `__main__` | إضافة `load_model_from_checkpoint` | ✅ مصلّح |
| 16 | استخدام `r` بدل `rw` في حلقة الطباعة | `rw` | ✅ مصلّح |
| 17 | التوليد يستخدم encoder ثابت (`[SOS]` + padding) | encoder و decoder يكبران مع بعض أثناء التوليد | ✅ مصلّح |

---

## demo.py

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 18 | `epochs` غير معرّف في حلقة التدريب | `epoch` | ✅ مصلّح |
| 19 | `{sum(...): , }` صيغة format غلط | `{sum(...):,}` | ✅ مصلّح |

---

## dataset.py

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 20 | `{len(...): , }` صيغة format غلط | `{len(...):,}` | ✅ مصلّح |
| 21 | `enc_input` فيه `<EOS>` ومختلف عن `dec_input` | جعل `enc_input` = `dec_input` | ✅ مصلّح |

---

## train.py

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 22 | الملف كان فارغ | كتابة سكربت تدريب كامل مع حفظ checkpoint | ✅ مصلّح |

---

## البيئة (Environment)

| # | الخطأ | الإصلاح | الحالة |
|---|--------|---------|--------|
| 23 | `pip3 install torch` → `externally-managed-environment` | إنشاء `.venv` واستخدام `pip install` داخلها | ✅ مصلّح |
| 24 | `No module named 'numpy'` | `pip install numpy` | ✅ مصلّح |
| 25 | `RDKit not found` | `pip install rdkit` (اختياري لكن مُثبّت) | ✅ مصلّح |

---

## مشاكل تصميم / سلوك (مو أخطاء syntax)

| # | المشكلة | التأثير | الحالة |
|---|---------|---------|--------|
| 26 | التدريب على 15 جزيء فقط في `demo.py` | Validity منخفض جداً | ⚠️ متوقع — استخدم `train.py` |
| 27 | `--max_samples 100 --epochs 10` قليل جداً | Validity 0% حتى بعد إصلاح التوليد | ⚠️ زِد البيانات والـ epochs |
| 28 | النموذج يولّد أحياناً تكرار (`CCCC...`، `3333...`) | SMILES صحيح تقنياً لكن مو جزيء واقعي | ⚠️ يحتاج تدريب أطول + temperature أقل |
| 29 | `generate.py` يحتاج checkpoint — `demo.py` ما يحفظ واحد | ما تقدر تستخدم `generate.py` بعد demo | ⚠️ استخدم `train.py` اللي يحفظ `checkpoint.pt` |
| 30 | `load_Sime_data` typo في الاسم (`Sime` بدل `Smiles`) | يشتغل لكن الاسم غلط إملائياً | ⚠️ ما تغيّر (حسب طلب المستخدم) |
| 31 | أسماء دوال فيها typos: `bulid_transformer`, `vocab_bulider`, `PositionalEncodeing` | تشتغل لكن الأسماء غير قياسية | ⚠️ ما تغيّر (حسب طلب المستخدم) |

---

## أوامر التشغيل الصحيحة

```bash
cd /Users/omarreyad/Downloads/SMILES-main
source .venv/bin/activate
cd SMILES-Code

python demo.py
python train.py --max_samples 10000 --epochs 50
python generate.py -CheckPoint checkpoint.pt -Tokenizer tokenizer.json -num 20
```

---

## نتائج بعد الإصلاحات

| التجربة | Validity |
|---------|----------|
| demo.py (15 جزيء، 40 epoch) قبل إصلاح التوليد | 0% |
| train.py (100 جزيء، 10 epoch) قبل إصلاح التوليد | 0% |
| train.py (1000 جزيء، 20 epoch) بعد إصلاح التوليد | 60% |
