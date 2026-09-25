"""L2-01 实测探针：分词器公共契约的每一步都跑一遍，不靠读代码推断。

用法（仓库根目录）：
    ./.venv/bin/python courseware/_data/recon/probe_l201.py

为什么不用真模型：本机没有可离线加载的 HF 分词器缓存，而 `PreTrainedTokenizerFast`
可以**直接用 tokenizers 库的对象构造**（`tokenizer_object=`），全程零网络。
构造出来的对象就是真实的 `TokenizersBackend`，走的是 tokenization_utils_base.py
里那条公共路径，所以测出来的形状与错误信息都是真的。

本探针回答四个问题：
  A. input_ids 与 attention_mask 到底在哪一步被造出来
  B. return_tensors 影响的是哪一步（数值会不会变）
  C. padding / truncation 的枚举与归一化（True 是什么、max_length 什么时候失效）
  D. token 属性体系：bos_token / bos_token_id 是不是同一份数据
"""
import warnings

warnings.filterwarnings("ignore")

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

VOCAB = {
    "[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3,
    "hello": 4, "world": 5, "the": 6, "quick": 7, "brown": 8, "fox": 9,
    "jumps": 10, "over": 11, "lazy": 12, "dog": 13,
}


def build(model_max_length=32, with_pad=True):
    tk = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    # 有 post_processor，[CLS]/[SEP] 才会真的出现在序列里
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    kw = dict(unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    if model_max_length is not None:
        kw["model_max_length"] = model_max_length
    if with_pad:
        kw["pad_token"] = "[PAD]"
    return PreTrainedTokenizerFast(tokenizer_object=tk, **kw)


def line(k, v):
    print(f"   {k:<34s}: {v}")


def step(title):
    print("=" * 78)
    print(title)


tok = build()

# ===========================================================================
step("A input_ids / attention_mask 的产生时机")
# ===========================================================================
line("类", type(tok).__name__)
line("model_input_names", tok.model_input_names)
line("padding_side", tok.padding_side)
line("truncation_side", tok.truncation_side)

print("\n   A1 单条、不 padding：")
o = tok("hello world the quick brown fox")
line("keys", list(o.keys()))
line("input_ids", o["input_ids"])
line("attention_mask", o["attention_mask"])
line("is_fast（_encodings 是否为 None）", o.is_fast)

print("\n   A2 直接调 base.pad()（这条路径由 tokenization_utils_base.py 的 _pad 实现）：")
before = {"input_ids": [4, 5, 6]}
after = tok.pad(dict(before), padding="max_length", max_length=5)
line("输入", before)
line("输出类型", type(after).__name__)
line("输出", dict(after))
line("right 侧补：新位是 0，原位是 1",
     [1] * 3 + [0] * 2 == after["attention_mask"])
line("left 侧补",
     dict(tok.pad({"input_ids": [4, 5, 6]}, padding="max_length",
                  max_length=5, padding_side="left")))

print("\n   A3 空输入也会被 _pad 拦下来：")
line("pad({'input_ids': []}, return_attention_mask=True)",
     dict(tok.pad({"input_ids": []}, return_attention_mask=True)))

# ===========================================================================
step("B return_tensors 影响的是哪一步")
# ===========================================================================
texts = ["hello world", "the quick brown fox jumps"]
a = tok(texts, padding=True)
b = tok(texts, padding=True, return_tensors="pt")
line("list  input_ids", a["input_ids"])
line("pt    input_ids", b["input_ids"].tolist())
line("数值是否相同", a["input_ids"] == b["input_ids"].tolist())
line("list 类型", type(a["input_ids"]).__name__)
line("pt   类型/形状/dtype",
     f"{type(b['input_ids']).__name__} {tuple(b['input_ids'].shape)} {b['input_ids'].dtype}")

print("\n   B2 单条文本 + return_tensors='pt' —— batch 轴是转换时补上的：")
for rt in (None, "pt", "np"):
    v = tok("hello world", return_tensors=rt)["input_ids"]
    line(f"return_tensors={rt}", f"{type(v).__name__} shape={getattr(v, 'shape', None)}")

print("\n   B3 转换点就在 BatchEncoding.__init__ 里，可以手工复现：")
line("BatchEncoding({...}, tensor_type='pt')",
     f"{BatchEncoding({'input_ids': [4, 5, 6]}, tensor_type='pt')['input_ids'].shape}")
line("... prepend_batch_axis=True",
     f"{BatchEncoding({'input_ids': [4, 5, 6]}, tensor_type='pt', prepend_batch_axis=True)['input_ids'].shape}")

print("\n   B4 不 padding 的 ragged 批次转张量会报错（错误信息自己点明了这一点）：")
try:
    tok(["hello world", "the quick brown fox jumps"], return_tensors="pt")
except Exception as ex:
    line("->", f"{type(ex).__name__}: {str(ex)[:100]}")

# ===========================================================================
step("C padding / truncation 的枚举与归一化")
# ===========================================================================
line("PaddingStrategy", [m.value for m in PaddingStrategy])
line("TruncationStrategy", [m.value for m in TruncationStrategy])

print("\n   C1 padding=True 与 padding='longest' 是不是同一个东西：")
line("padding=True", tok(texts, padding=True)["input_ids"])
line("padding='longest'", tok(texts, padding="longest")["input_ids"])

print("\n   C2 padding='max_length' + 默认 model_max_length（极大值）会静默降级：")
tok3 = build(model_max_length=None)
line("model_max_length", tok3.model_max_length)
line("padding='max_length'", tok3(texts, padding="max_length")["input_ids"])
line("padding='max_length' + max_length=8",
     tok3(texts, padding="max_length", max_length=8, truncation=True)["input_ids"])

print("\n   C3 只给 max_length、不给 truncation —— 会自己打开截断：")
line("max_length=4", tok("hello world the quick brown fox", max_length=4)["input_ids"])
line("（不截断时应为）", tok("hello world the quick brown fox")["input_ids"])

print("\n   C4 四种 truncation 在成对输入上的真实差别（max_length=14）：")
q, p = "hello world the quick brown fox", "the quick brown fox jumps over the lazy dog"
line("不截断", tok(q, p)["input_ids"])
for t in ["only_first", "only_second", "longest_first", "do_not_truncate"]:
    line(t, tok(q, p, truncation=t, max_length=14)["input_ids"])

print("\n   C5 padding 与 truncation 的互查：")
tok_nopad = build(with_pad=False)
try:
    tok_nopad(texts, padding=True)
except Exception as ex:
    line("->", f"{type(ex).__name__}: {str(ex)[:80]}")
try:
    tok(texts, padding="max_length", truncation=True, max_length=7, pad_to_multiple_of=4)
except Exception as ex:
    line("max_length=7 与 pad_to_multiple_of=4 ->", f"{type(ex).__name__}: {str(ex)[:80]}")

# ===========================================================================
step("D token 属性体系：_token 与 _token_id 是同一份数据")
# ===========================================================================
line("SPECIAL_TOKENS_ATTRIBUTES", tok.SPECIAL_TOKENS_ATTRIBUTES)
line("pad_token", repr(tok.pad_token))
line("pad_token_id", tok.pad_token_id)
line("cls_token / cls_token_id", f"{tok.cls_token!r} / {tok.cls_token_id}")
line("special_tokens_map", tok.special_tokens_map)
line("all_special_tokens", tok.all_special_tokens)
line("all_special_ids", tok.all_special_ids)

print("\n   D2 赋值走 __setattr__：给 id 就等于给 token，给新名字会自动进 map：")
tok.pad_token_id = 3
line("tok.pad_token_id = 3 之后 pad_token", repr(tok.pad_token))
tok.add_special_tokens({"mask_token": "[MASK]"})
line("add_special_tokens 后 vocab_size", tok.vocab_size)
line("mask_token / mask_token_id", f"{tok.mask_token!r} / {tok.mask_token_id}")
line("新名字会加进 SPECIAL_TOKENS_ATTRIBUTES",
     "mask_token" in tok.SPECIAL_TOKENS_ATTRIBUTES)

print("\n   D3 未设置的 token 返回 None，而不是抛错：")
tok2 = build()
line("bos_token", repr(tok2.bos_token), )
line("bos_token_id", tok2.bos_token_id)

print("\n   D4 赋 None 是「清空」，不是「报错」：")
tok4 = build()
line("清空前 special_tokens_map", tok4.special_tokens_map)
tok4.pad_token = None
line("tok.pad_token = None 之后 pad_token", repr(tok4.pad_token))
line("pad_token_id", tok4.pad_token_id)
line("special_tokens_map", tok4.special_tokens_map)

# ===========================================================================
step("E 值必须精确：padding 用的是 is 比较，不是真假值比较")
# ===========================================================================
for v in (1, "longest", True):
    try:
        r = tok(["hello world", "the quick brown fox jumps"], padding=v)["input_ids"]
        line(f"padding={v!r}", f"ok  {r}")
    except Exception as ex:
        line(f"padding={v!r}", f"{type(ex).__name__}: {str(ex)[:70]}")

print("\n" + "=" * 78)
print("探针结束。上面每一行都是本次运行的真实输出。")
