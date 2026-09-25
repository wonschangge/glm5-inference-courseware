# -*- coding: utf-8 -*-
"""L2-02 实测探针：GLM-5.3-Flash 真实 tokenizer 的 fast 路径、形状与视觉 token 对齐。

准备（一次即可）：
    mkdir -p /tmp/glm5tok && cd /tmp/glm5tok
    for f in tokenizer.json tokenizer_config.json processor_config.json; do
      curl -L -o $f https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/$f
    done

运行（用项目自带的 venv，见 courseware/tools/check_params.py 的取法）：
    /data/WORKSPACE/transformer-project/.venv/bin/python courseware/_data/recon/probe_l202.py

本脚本的每一个数字都会被 L2-02 的 source.md / lesson.js 引用；改脚本必须重跑。
"""
import json
import os
import sys
import time
import types
import warnings

warnings.filterwarnings("ignore")

TOK_DIR = os.environ.get("GLM5_TOKENIZER_DIR", "/tmp/glm5tok")
UPSTREAM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "..", "upstream-transformers", "src"))
sys.path.insert(0, UPSTREAM)

import torch  # noqa: E402
from transformers.tokenization_utils_tokenizers import TokenizersBackend  # noqa: E402
from transformers.tokenization_python import PythonBackend, Trie, ExtensionsTrie  # noqa: E402
from transformers.convert_slow_tokenizer import (  # noqa: E402
    SLOW_TO_FAST_CONVERTERS,
    SpmConverter,
    generate_merges,
)
from transformers.tokenization_utils_base import generate_merges as generate_merges_base  # noqa: E402

IMG_ID = 154854  # Glm5NextConfig.image_token_id 的默认值


def sec(title):
    print("\n=== " + title + " ===")


def main():
    if not os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")):
        print("SKIP: 没找到 %s/tokenizer.json（见文件头部的下载命令）" % TOK_DIR)
        return 1

    sec("1 直方图：GLM-5.3-Flash 的 checkpoint 里有什么")
    files = sorted(os.listdir(TOK_DIR))
    print("tokenizer 目录文件:", files)
    cfg = json.load(open(os.path.join(TOK_DIR, "tokenizer_config.json"), encoding="utf-8"))
    print("tokenizer_class=%r backend=%r model_max_length=%s padding_side=%r" % (
        cfg.get("tokenizer_class"), cfg.get("backend"), cfg.get("model_max_length"), cfg.get("padding_side")))
    print("extra_special_tokens 条数 =", len(cfg.get("extra_special_tokens") or []))

    sec("2 加载：走的是哪条分支")
    t0 = time.time()
    tok = TokenizersBackend.from_pretrained(TOK_DIR)
    print("%s  %.2fs  is_fast=%s  len=%d  vocab_size=%d  model=%s" % (
        type(tok).__name__, time.time() - t0, tok.is_fast, len(tok), tok.vocab_size,
        type(tok.backend_tokenizer.model).__name__))
    print("model_input_names=%s  padding_side=%s" % (tok.model_input_names, tok.padding_side))
    print("pad=%r/%s  eos=%r/%s  bos=%r/%s  unk=%r/%s" % (
        tok.pad_token, tok.pad_token_id, tok.eos_token, tok.eos_token_id,
        tok.bos_token, tok.bos_token_id, tok.unk_token, tok.unk_token_id))
    print("can_save_slow_tokenizer=%s  hasattr(vocab_file)=%s  vocab_files_names=%s" % (
        tok.can_save_slow_tokenizer, hasattr(tok, "vocab_file"), tok.vocab_files_names))
    print("pipeline: normalizer=%s pre_tokenizer=%s post_processor=%s decoder=%s" % (
        type(tok.backend_tokenizer.normalizer).__name__,
        type(tok.backend_tokenizer.pre_tokenizer).__name__,
        type(tok.backend_tokenizer.post_processor).__name__,
        type(tok.backend_tokenizer.decoder).__name__))
    print("num_special_tokens_to_add(pair=False) =", tok.num_special_tokens_to_add())
    from transformers import Glm5NextConfig, Glm5NextTextConfig

    print("模型侧 Glm5NextTextConfig.vocab_size = %d（比 len(tok) 多 %d 个 embedding 槽位）" % (
        Glm5NextTextConfig().vocab_size, Glm5NextTextConfig().vocab_size - len(tok)))
    print("Glm5NextConfig.image_token_id = %d ; 与 154854 一致: %s" % (
        Glm5NextConfig().image_token_id, Glm5NextConfig().image_token_id == 154854))

    sec("3 关键 id：谁占着 154820 / 154854 / 154855")
    for tid in (154820, 154854, 154855, 154856):
        print("  %d -> %r" % (tid, tok.convert_ids_to_tokens(tid)))
    for piece in ("<|begin_of_image|>", "<|end_of_image|>", "<|begin_of_video|>", "[gMASK]", "<sop>", "<eop>"):
        print("  %-20r -> %s" % (piece, tok.convert_tokens_to_ids(piece)))

    sec("4 单条：中文文本的真实编码")
    s = "你好，世界！GLM-5 是一个混合架构模型。"
    e = tok(s)
    print("text=%r  len(ids)=%d" % (s, len(e["input_ids"])))
    print("tokens[:8]=%r" % (e.tokens()[:8],))
    e2 = tok(s, return_offsets_mapping=True)
    print("offset_mapping[:4]=%r  （只有 fast 路给得出这个）" % (e2["offset_mapping"][:4],))

    sec("5 批量：形状演算")
    texts = ["短", "你好，世界！", "GLM-5 是一个混合架构模型，" * 4]
    b = tok(texts, padding=True, return_tensors="pt")
    print("3 条不等长 + padding=True -> %s" % {k: tuple(v.shape) for k, v in b.items()})
    print("  每行真实长度 =", b["attention_mask"].sum(-1).tolist())
    print("  第 0 行前 6 个 id =", b["input_ids"][0].tolist()[:6], "（左侧补 pad）")
    c = tok(texts, padding="max_length", max_length=20, truncation=True, return_tensors="pt")
    print("padding=max_length/20 + truncation -> %s" % {k: tuple(v.shape) for k, v in c.items()})
    print("  第 0 行 mask =", c["attention_mask"][0].tolist())
    print("  第 2 行（被截断）mask.sum =", int(c["attention_mask"][2].sum()))
    o = tok(texts, padding=True, return_offsets_mapping=True, return_tensors="pt")
    print("3 条 + return_offsets_mapping -> offset_mapping %s" % (tuple(o["offset_mapping"].shape),))
    single = tok(s)
    print("tok(单条)            -> list[int]，%d 个" % len(single["input_ids"]))
    print("tok([单条])          -> list[list[int]]，%d 条" % len(tok([s])["input_ids"]))
    print("tok(单条, pt)        -> %s" % (tuple(tok(s, return_tensors="pt")["input_ids"].shape),))
    print("tok([单条], pt)      -> %s" % (tuple(tok([s], return_tensors="pt")["input_ids"].shape),))
    big = ["第 %d 条测试文本，用来测批量编码。" % i for i in range(256)]
    t0 = time.time()
    tok(big, padding=True, return_tensors="pt")
    print("256 条一次 encode_batch = %.0f ms" % ((time.time() - t0) * 1000))
    t0 = time.time()
    [tok(x) for x in big]
    print("256 次单条调用         = %.0f ms" % ((time.time() - t0) * 1000))

    sec("6 慢路径：GLM-5 能不能用纯 Python 后端加载")
    try:
        slow = PythonBackend.from_pretrained(TOK_DIR)
        print("UNEXPECTED 加载成功:", type(slow).__name__, slow.is_fast)
    except Exception as ex:
        print("PythonBackend.from_pretrained -> %s: %s" % (
            type(ex).__name__, str(ex).replace("\n", " ")[:160]))
    out = "/tmp/glm5tok_saved"
    os.makedirs(out, exist_ok=True)
    print("save_pretrained(默认)      ->", sorted(os.path.basename(f) for f in tok.save_pretrained(out)))
    print("save_pretrained(legacy=T)  ->", sorted(os.path.basename(f)
                                                for f in tok.save_pretrained(out, legacy_format=True)))

    sec("7 视觉 token：真实图像处理器的输出")
    from PIL import Image
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor

    ip = Glm5NextImageProcessor()
    print("patch_size=%s temporal_patch_size=%s merge_size=%s" % (
        ip.patch_size, ip.temporal_patch_size, ip.merge_size))
    grids = {}
    for name, wh in (("448x448", (448, 448)), ("672x336", (672, 336)), ("224x224", (224, 224))):
        out_ip = ip(images=Image.new("RGB", wh, (200, 60, 60)), return_tensors="pt")
        g = out_ip["image_grid_thw"]
        ntok = int(g.prod()) // (ip.merge_size ** 2)
        grids[name] = (g.tolist(), ntok)
        print("  %-8s grid_thw=%-14s pixel_values=%-14s 视觉 token = %s.prod()/%d = %d" % (
            name, g.tolist(), tuple(out_ip["pixel_values"].shape), g.tolist(), ip.merge_size ** 2, ntok))

    sec("8 文本与视觉在一条序列上对齐")
    for name in ("448x448", "224x224"):
        grid, ntok = grids[name]
        prompt = "<|begin_of_image|>" + "<|image|>" * ntok + "<|end_of_image|>你好，描述这张图。"
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"][0]
        pos = (ids == IMG_ID).nonzero().flatten()
        print("  %-8s n_vision=%-4d 总长=%-4d vision 槽位=[%d..%d] 共 %d 个" % (
            name, ntok, ids.shape[-1], pos[0], pos[-1], pos.numel()))
    n1 = grids["448x448"][1]
    s1 = "<|begin_of_image|>" + "<|image|>" * n1 + "<|end_of_image|>图里是什么？"
    s2 = "只用文字的一句话。"
    b2 = tok([s1, s2], padding=True, return_tensors="pt")
    print("  两条样本 batch -> %s" % {k: tuple(v.shape) for k, v in b2.items()})
    print("  每行长度 =", b2["attention_mask"].sum(-1).tolist(),
          " vision 槽位数 =", int((b2["input_ids"] == IMG_ID).sum()))
    emb = torch.zeros(b2["input_ids"].shape[0], b2["input_ids"].shape[1], 4096)
    print("  inputs_embeds%s 里：视觉槽 %d + 文本槽 %d" % (
        tuple(emb.shape), int((b2["input_ids"] == IMG_ID).sum()),
        int(b2["input_ids"].numel() - (b2["input_ids"] == IMG_ID).sum())))

    sec("9 纯 Python 那条路上的零件（不需要 GLM-5 权重）")
    t = Trie()
    for piece in ("[gMASK]", "<sop>", "<|image|>"):
        t.add(piece)
    text = "前缀[gMASK]<sop>中间<|image|>后缀[gMASK]"
    print("Trie.split(%r)" % text)
    print("  ->", t.split(text))
    t2 = ExtensionsTrie()
    for w in ("apple", "app", "application"):
        t2.add(w)
    print("ExtensionsTrie.extensions('app') ->", t2.extensions("app"))
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5}
    scores = {"a": -1.0, "b": -2.0, "c": -3.0, "ab": -4.0, "bc": -5.0, "abc": -6.0}
    print("generate_merges(按词表序)  ->", generate_merges_base(vocab))
    print("generate_merges(按 piece 分数) ->", generate_merges(vocab, scores))

    sec("10 SentencePiece proto -> fast tokenizer 的组装函数")
    proto = types.SimpleNamespace(
        trainer_spec=types.SimpleNamespace(byte_fallback=True, unk_piece="<unk>", unk_id=0, model_type=1),
        normalizer_spec=types.SimpleNamespace(precompiled_charsmap=None),
    )
    pieces = [("<unk>", 0.0), ("▁你", -1.0), ("好", -2.0), ("▁世界", -3.0), ("<0x0A>", -4.0)]
    tk = SpmConverter.build_tokenizer_from_spm_proto(proto=proto, vocab=pieces, merges=None)
    print("组装结果: %s normalizer=%s pre_tokenizer=%s decoder=%s vocab_size=%d" % (
        type(tk).__name__, type(tk.normalizer).__name__, type(tk.pre_tokenizer).__name__,
        type(tk.decoder).__name__, tk.get_vocab_size()))
    one = tk.encode("▁你好▁世界")
    print("  encode('▁你好▁世界') -> ids=%s tokens=%s decode=%r" % (one.ids, one.tokens, tk.decode(one.ids)))
    from collections import Counter
    cnt = Counter(v.__name__ for v in SLOW_TO_FAST_CONVERTERS.values())
    print("SLOW_TO_FAST_CONVERTERS 条目 %d 个，复用最多: %s" % (len(SLOW_TO_FAST_CONVERTERS), cnt.most_common(4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
