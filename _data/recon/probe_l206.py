"""L2-06 实测探针：多模态处理器装配的实际行为。

用法（仓库根目录 /data/WORKSPACE/transformer-project）：
    ./.venv/bin/python courseware/_data/recon/probe_l206.py

本脚本是**工装**，不属于覆盖域（覆盖域只统计 src/transformers/**/*.py），
不计入覆盖率。它的作用是：把 L2-06 里每一句断言变成一次真实运行。
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402

from transformers import (  # noqa: E402
    Glm5NextImageProcessorPil,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    PreTrainedTokenizerFast,
)


def line(k, v):
    print(f"   {k:<44s}: {v}")


def step(t):
    print("=" * 78)
    print(t)


# ---------------------------------------------------------------------------
# ① 造一个最小可用 tokenizer（真实 GLM-5 词表不在本地，用 WordLevel 顶替）
# ---------------------------------------------------------------------------
step("① 装配：tokenizer + image_processor + video_processor -> Glm5NextProcessor")

SPECIALS = [
    "<|endoftext|>", "<|image|>", "<|video|>", "<|begin_of_video|>", "<|end_of_video|>",
    "<|begin_of_image|>", "<|end_of_image|>", "user", "assistant", "hello", "describe",
]
vocab = {tok: i for i, tok in enumerate(SPECIALS)}
vocab["<unk>"] = len(vocab)
tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
tok_obj.add_special_tokens(SPECIALS)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok_obj, unk_token="<unk>", pad_token="<|endoftext|>", eos_token="<|endoftext|>"
)

image_processor = Glm5NextImageProcessorPil()
video_processor = Glm5NextVideoProcessor()

line("image_processor 类", type(image_processor).__name__)
line("image_processor.model_input_names", image_processor.model_input_names)
line("video_processor.model_input_names", video_processor.model_input_names)
line("image_processor.merge_size / patch_size", (image_processor.merge_size, image_processor.patch_size))

processor = Glm5NextProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor)
line("processor.image_token / image_token_id", (processor.image_token, processor.image_token_id))
line("processor.video_start_id / video_end_id", (processor.video_start_id, processor.video_end_id))
line("processor.model_input_names", processor.model_input_names)
line("processor.get_attributes()", processor.get_attributes())
line("processor.all_special_multimodal_tokens", processor.all_special_multimodal_tokens)

# ---------------------------------------------------------------------------
# ② smart_resize：一张图会被 resize 到多大的画布
# ---------------------------------------------------------------------------
step("② smart_resize：把 (h, w) 对齐到 28 的倍数并卡进 token 预算")

for (h, w) in [(336, 336), (672, 672), (512, 512), (1024, 768), (2048, 1536)]:
    n_patch = image_processor.get_number_of_image_patches(h, w)
    line(f"{h}x{w} -> patches / tokens", f"{n_patch} / {n_patch // image_processor.merge_size ** 2}")

# ---------------------------------------------------------------------------
# ③ 一次真实图像调用：inputs 字典到底有哪几个 key
# ---------------------------------------------------------------------------
step("③ processor(text=..., images=...) 的返回字典")

img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
inputs = processor(text="<|image|> describe", images=[img], return_tensors="pt")
line("keys()", list(inputs.keys()))
for k, v in inputs.items():
    shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
    dtype = getattr(v, "dtype", "-")
    line(f"  {k}", f"shape={shape} dtype={dtype}")

# ---------------------------------------------------------------------------
# ④ 占位符展开：1 个 <|image|> 变成多少个 token
# ---------------------------------------------------------------------------
step("④ 占位符展开（prompt 侧 1 个 -> 序列侧 N 个）")

one = processor.replace_image_token(inputs, image_idx=0)
line("replace_image_token 返回的字符串长度", len(one))
line("prompt 里 <|image|> 出现次数", "1")
line("展开后 input_ids 长度", int(inputs["input_ids"].shape[1]))
line("mm_token_type_ids[0][:6]", inputs["mm_token_type_ids"][0][:6].tolist())
line("mm_token_type_ids 中 1 的个数", int((inputs["mm_token_type_ids"] == 1).sum()))
line("input_ids 中 image_token_id 的个数", int((inputs["input_ids"] == processor.image_token_id).sum()))

# ---------------------------------------------------------------------------
# ⑤ 图像 / 视频共用同一个 <|image|>：mm_token_type_ids 靠 BOE/EOV 区分
# ---------------------------------------------------------------------------
step("⑤ create_mm_token_type_ids：图像=1，视频帧=2")

ids = [
    [processor.video_start_id, processor.image_token_id, processor.image_token_id,
     processor.video_end_id, processor.image_token_id, 5],
]
line("输入 input_ids", ids[0])
line("输出 mm_token_type_ids", processor.create_mm_token_type_ids(ids)[0])

# ---------------------------------------------------------------------------
# ⑥ chat template：模板只写 1 个占位符，展开交给 processor
# ---------------------------------------------------------------------------
step("⑥ apply_chat_template(tokenize=True) 的装配顺序")

TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}:"
    "{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}<|image|>{% else %}{{ c['text'] }}{% endif %}"
    "{% endfor %}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)
processor.chat_template = TEMPLATE
conversation = [
    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "describe"}]},
]
out = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
line("keys()", list(out.keys()))
line("input_ids", out["input_ids"][0].tolist())
line("input_ids 长度", int(out["input_ids"].shape[1]))
line("image_token_id 个数", int((out["input_ids"] == processor.image_token_id).sum()))

text_only = processor.apply_chat_template(conversation)
line("未 tokenize 的 prompt", repr(text_only))

# ---------------------------------------------------------------------------
# ⑦ 视频：frame token 的文本结构
# ---------------------------------------------------------------------------
step("⑦ replace_frame_token_id：每一帧展开成什么")

line("replace_frame_token_id(0.5, 4)", repr(processor.replace_frame_token_id(0.5, num_image_tokens=4)))

# ---------------------------------------------------------------------------
# ⑧ smart_resize 给出的画布本身（而不只是 patch 数）
# ---------------------------------------------------------------------------
step("⑧ smart_resize 返回的对齐画布")

from transformers.models.glm5_next.image_processing_pil_glm5_next import smart_resize as sr_pil  # noqa: E402

for (h, w) in [(336, 336), (512, 512), (1024, 768), (2048, 1536), (100, 100)]:
    th, tw = sr_pil(num_frames=2, height=h, width=w, factor=28, temporal_factor=2,
                    min_pixels=16, max_pixels=8000)
    gh, gw = th // 14, tw // 14
    line(f"{h}x{w} -> canvas / grid / tokens", f"{th}x{tw} / {gh}x{gw} / {gh * gw // 4}")

# ---------------------------------------------------------------------------
# ⑨ 一批不同尺寸的图：pixel_values 是"拼起来的一条"
# ---------------------------------------------------------------------------
step("⑨ 两张不同尺寸的图一起进 PIL 后端")

imgs = [Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))]
b = processor(text="<|image|> <|image|>", images=imgs, return_tensors="pt")
line("pixel_values.shape", tuple(b["pixel_values"].shape))
line("image_grid_thw", b["image_grid_thw"].tolist())
line("input_ids 长度", int(b["input_ids"].shape[1]))
line("image_token_id 个数", int((b["input_ids"] == processor.image_token_id).sum()))

# ---------------------------------------------------------------------------
# ⑩ 视频 patchify：temporal_patch_size 把 2 帧并成 1 个时间 patch
# ---------------------------------------------------------------------------
step("⑩ 视频 patchify 与每帧占位符数")

import torch  # noqa: E402

frames = torch.zeros(1, 16, 3, 336, 336)
fp, gt, gh, gw = video_processor.patchify(frames, patch_size=14, merge_size=2, temporal_patch_size=2)
line("输入 videos.shape", tuple(frames.shape))
line("flatten_patches.shape", tuple(fp.shape))
line("(grid_t, grid_h, grid_w)", (gt, gh, gw))
line("get_num_of_video_patches(16,336,336)", video_processor.get_num_of_video_patches(16, 336, 336))
line("video token 总数 = prod//merge^2", (gt * gh * gw) // 4)
line("每帧占位符 = tokens // num_frames", ((gt * gh * gw) // 4) // 16)
line("每帧占位符 = gh*gw//(merge^2*temporal)", (gh * gw) // (4 * 2))

# ---------------------------------------------------------------------------
# ⑪ 奇数帧：pad 一帧后时间维才对得上
# ---------------------------------------------------------------------------
step("⑪ 15 帧（奇数）时的 temporal padding")

frames15 = torch.zeros(1, 15, 3, 336, 336)
fp15, gt15, gh15, gw15 = video_processor.patchify(frames15, patch_size=14, merge_size=2, temporal_patch_size=2)
line("输入帧数 / (-15 % 2) 的 pad", (15, -15 % 2))
line("flatten_patches.shape", tuple(fp15.shape))
line("grid_t", gt15)
line("get_num_of_video_patches(15,336,336)", video_processor.get_num_of_video_patches(15, 336, 336))

# ---------------------------------------------------------------------------
# ⑫ torchvision 后端：同一张图的 pixel_values 逐元素相等吗
# ---------------------------------------------------------------------------
step("⑫ 两套实现的结果对比")

try:
    from transformers import Glm5NextImageProcessor as TV  # noqa: E402

    tvp = TV()
    out_tv = tvp(images=[Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))], return_tensors="pt")
    out_pil = image_processor(images=[Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))],
                              return_tensors="pt")
    line("torchvision pixel_values.shape", tuple(out_tv["pixel_values"].shape))
    line("PIL         pixel_values.shape", tuple(out_pil["pixel_values"].shape))
    line("两者 allclose", bool(torch.allclose(out_tv["pixel_values"].float(),
                                              torch.from_numpy(out_pil["pixel_values"]).float(), atol=1e-5)))
    line("torchvision image_grid_thw", out_tv["image_grid_thw"].tolist())
except Exception as e:  # noqa: BLE001
    line("torchvision 后端不可用", repr(e)[:120])

# ---------------------------------------------------------------------------
# ⑬ 极小的图：min_image_tokens 的下限是怎么落地的
# ---------------------------------------------------------------------------
step("⑬ 小于预算的图会被放大到最小 token 数")

for (h, w) in [(50, 50), (30, 30), (14, 14), (200, 100)]:
    th, tw = sr_pil(num_frames=2, height=h, width=w, factor=28, temporal_factor=2,
                    min_pixels=16, max_pixels=8000)
    n = (th // 14) * (tw // 14)
    line(f"{h}x{w} -> canvas / tokens", f"{th}x{tw} / {n // 4}")
