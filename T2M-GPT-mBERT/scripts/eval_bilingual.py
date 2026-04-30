#!/usr/bin/env python3
import argparse
import csv
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.motion_process import recover_from_ric

TEXT2MOTION_MBERT_PATH = CURRENT_DIR / "text2motion_prompt_mbert.py"
MBERT_SPEC = importlib.util.spec_from_file_location("text2motion_prompt_mbert", TEXT2MOTION_MBERT_PATH)
if MBERT_SPEC is None or MBERT_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module: {TEXT2MOTION_MBERT_PATH}")
text2motion_mbert = importlib.util.module_from_spec(MBERT_SPEC)
MBERT_SPEC.loader.exec_module(text2motion_mbert)

TEXT2MOTION_CLIP_PATH = CURRENT_DIR / "text2motion_prompt.py"
CLIP_SPEC = importlib.util.spec_from_file_location("text2motion_prompt", TEXT2MOTION_CLIP_PATH)
if CLIP_SPEC is None or CLIP_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module: {TEXT2MOTION_CLIP_PATH}")
text2motion_clip = importlib.util.module_from_spec(CLIP_SPEC)
CLIP_SPEC.loader.exec_module(text2motion_clip)


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path not found: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_proj_ckpt(proj_ckpt_arg: str) -> str:
    if proj_ckpt_arg:
        return proj_ckpt_arg

    preferred = [
        os.path.join("output_projection", "HML3D_mBERT_last2_proj", "proj_best.pth"),
        os.path.join("output_projection", "HML3D_mBERT_last2_proj", "proj_last.pth"),
        os.path.join("output_projection", "HML3D_mBERT_proj", "proj_best.pth"),
        os.path.join("output_projection", "HML3D_mBERT_proj", "proj_last.pth"),
        os.path.join("output_projection", "KITALL_mBERT_proj", "proj_best.pth"),
        os.path.join("output_projection", "KITALL_mBERT_proj", "proj_last.pth"),
        os.path.join("output_projection", "KIT2K_mBERT_proj", "proj_best.pth"),
        os.path.join("output_projection", "KIT2K_mBERT_proj", "proj_last.pth"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    auto_candidates = sorted(
        glob.glob(os.path.join("output_projection", "*", "proj_best.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    if auto_candidates:
        return auto_candidates[0]

    raise FileNotFoundError(
        "Could not auto-resolve projection checkpoint. "
        "Pass --proj-ckpt with an explicit path."
    )


PAIR_SAMPLE_FIELDNAMES = [
    "pair_id",
    "motion_id",
    "en_prompt",
    "zh_prompt",
    "f_tag",
    "to_tag",
    "text_path",
    "caption_idx",
]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(out):
        return float(default)
    return float(out)


def _to_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def read_bilingual_pairs(csv_path: str, max_pairs: int) -> List[Dict[str, object]]:
    ensure_exists(csv_path)
    pairs: List[Dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"motion_id", "en_prompt", "zh_prompt", "f_tag", "to_tag"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "Sample CSV missing required columns. "
                f"Required={sorted(required)}, got={reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, 1):
            motion_id = str(row.get("motion_id", "")).strip()
            en_prompt = str(row.get("en_prompt", "")).strip()
            zh_prompt = str(row.get("zh_prompt", "")).strip()
            if not motion_id or not en_prompt or not zh_prompt:
                continue

            pairs.append(
                {
                    "pair_id": str(row.get("pair_id", "")).strip() or str(row_idx),
                    "motion_id": motion_id,
                    "en_prompt": en_prompt,
                    "zh_prompt": zh_prompt,
                    "f_tag": _to_float(row.get("f_tag", 0.0), 0.0),
                    "to_tag": _to_float(row.get("to_tag", 0.0), 0.0),
                    "text_path": str(row.get("text_path", "")).strip(),
                    "caption_idx": _to_int(row.get("caption_idx", -1), -1),
                }
            )
            if len(pairs) >= max_pairs:
                break

    for idx, row in enumerate(pairs, 1):
        row["pair_id"] = str(idx)
    return pairs


def write_bilingual_pairs(csv_path: str, rows: List[Dict[str, object]]) -> None:
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_SAMPLE_FIELDNAMES)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "pair_id": str(idx),
                    "motion_id": str(row.get("motion_id", "")),
                    "en_prompt": str(row.get("en_prompt", "")),
                    "zh_prompt": str(row.get("zh_prompt", "")),
                    "f_tag": _to_float(row.get("f_tag", 0.0), 0.0),
                    "to_tag": _to_float(row.get("to_tag", 0.0), 0.0),
                    "text_path": str(row.get("text_path", "")),
                    "caption_idx": _to_int(row.get("caption_idx", -1), -1),
                }
            )


def parse_hml3d_caption_line(line: str) -> Optional[Tuple[str, float, float]]:
    parts = line.strip().split("#")
    if not parts:
        return None

    caption = parts[0].strip()
    if not caption:
        return None

    f_tag = 0.0
    to_tag = 0.0
    if len(parts) >= 4:
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            f_tag = 0.0
            to_tag = 0.0

    if np.isnan(f_tag):
        f_tag = 0.0
    if np.isnan(to_tag):
        to_tag = 0.0

    return caption, float(f_tag), float(to_tag)


def read_hml3d_test_pairs(
    split_path: str,
    text_dir: str,
    max_pairs: int,
    seed: int,
    shuffle_samples: bool,
) -> List[Dict[str, object]]:
    with open(split_path, "r", encoding="utf-8") as f:
        motion_ids = [line.strip() for line in f if line.strip()]

    candidates: List[Dict[str, object]] = []
    for motion_id in motion_ids:
        text_path = os.path.join(text_dir, f"{motion_id}.txt")
        if not os.path.exists(text_path):
            continue

        with open(text_path, "r", encoding="utf-8") as f:
            for caption_idx, line in enumerate(f):
                parsed = parse_hml3d_caption_line(line)
                if parsed is None:
                    continue
                en_prompt, f_tag, to_tag = parsed
                candidates.append(
                    {
                        "motion_id": motion_id,
                        "en_prompt": en_prompt,
                        "f_tag": f_tag,
                        "to_tag": to_tag,
                        "text_path": text_path,
                        "caption_idx": caption_idx,
                    }
                )

    if shuffle_samples:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    selected = candidates[:max_pairs]
    pairs: List[Dict[str, object]] = []
    for idx, row in enumerate(selected, 1):
        pairs.append(
            {
                "pair_id": str(idx),
                "motion_id": str(row["motion_id"]),
                "en_prompt": str(row["en_prompt"]),
                "zh_prompt": "",
                "f_tag": float(row["f_tag"]),
                "to_tag": float(row["to_tag"]),
                "text_path": str(row["text_path"]),
                "caption_idx": int(row["caption_idx"]),
            }
        )
    return pairs


def build_en_to_zh_translator(model_name: str, device: torch.device):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for EN->ZH translation. Install with: pip install transformers"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError as exc:
        msg = str(exc).lower()
        if "sentencepiece" in msg:
            raise ImportError(
                "sentencepiece is required by the selected translation tokenizer. "
                "Install with: pip install sentencepiece"
            ) from exc
        raise

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def translate_en_to_zh_prompts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_text_len: int,
) -> List[str]:
    translated: List[str] = []
    if not texts:
        return translated

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=128,
            num_beams=4,
        )
        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for out in outputs:
            text = re.sub(r"\s+", " ", out).strip()
            translated.append(text)

    return translated


def load_or_create_sample_pairs(
    sample_csv_path: str,
    split_path: str,
    text_dir: str,
    max_pairs: int,
    seed: int,
    shuffle_samples: bool,
    regenerate_sample_csv: bool,
    translator_model_name: str,
    translator_batch_size: int,
    max_text_len: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    if os.path.exists(sample_csv_path) and not regenerate_sample_csv:
        pairs = read_bilingual_pairs(sample_csv_path, max_pairs)
        if len(pairs) < max_pairs:
            raise RuntimeError(
                f"Sample CSV has only {len(pairs)} rows, but max_pairs={max_pairs}. "
                f"Remove/regenerate CSV: {sample_csv_path}"
            )
        return pairs

    pairs = read_hml3d_test_pairs(
        split_path=split_path,
        text_dir=text_dir,
        max_pairs=max_pairs,
        seed=seed,
        shuffle_samples=shuffle_samples,
    )
    if len(pairs) < max_pairs:
        raise RuntimeError(
            f"HumanML3D test candidates insufficient: got {len(pairs)}, required {max_pairs}."
        )

    translator_tokenizer, translator_model = build_en_to_zh_translator(
        model_name=translator_model_name,
        device=device,
    )
    zh_prompts = translate_en_to_zh_prompts(
        texts=[str(row["en_prompt"]) for row in pairs],
        tokenizer=translator_tokenizer,
        model=translator_model,
        device=device,
        batch_size=translator_batch_size,
        max_text_len=max_text_len,
    )
    if len(zh_prompts) != len(pairs):
        raise RuntimeError(
            f"Translation count mismatch: expected {len(pairs)}, got {len(zh_prompts)}"
        )

    for row, zh_prompt in zip(pairs, zh_prompts):
        zh = zh_prompt.strip()
        row["zh_prompt"] = zh if zh else str(row["en_prompt"])

    write_bilingual_pairs(sample_csv_path, pairs)

    del translator_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pairs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_mean_l2(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return 0.0
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(features.shape[0], 1)
    return float(np.mean(dist[iu]))


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_std(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64)))


def normalize_caption_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .,!?:;，。！？；：\"'`“”’‘")
    return t


def sequence_similarity_deviation(ref_seq: np.ndarray, tgt_seq: np.ndarray) -> Tuple[float, float, float]:
    compare_len = min(int(ref_seq.shape[0]), int(tgt_seq.shape[0]))
    ref_use = ref_seq[:compare_len]
    tgt_use = tgt_seq[:compare_len]
    sim_cos = cosine_sim(ref_use.reshape(-1), tgt_use.reshape(-1))
    sim_mse = float(np.mean((ref_use - tgt_use) ** 2))
    frame_gap = float(abs(int(ref_seq.shape[0]) - int(tgt_seq.shape[0])))
    return sim_cos, sim_mse, frame_gap


def load_gt_motion_segment(
    motion_dir: str,
    motion_id: str,
    f_tag: float,
    to_tag: float,
    fps: float,
) -> Dict[str, object]:
    motion_path = os.path.join(motion_dir, f"{motion_id}.npy")
    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Cannot find HumanML3D motion file for motion_id='{motion_id}': {motion_path}")
    motion_all = np.load(motion_path).astype(np.float32)

    if f_tag != 0.0 or to_tag != 0.0:
        start = int(f_tag * fps)
        end = int(to_tag * fps)
        if 0 <= start < end <= len(motion_all):
            motion_seg = motion_all[start:end]
        else:
            motion_seg = motion_all
            f_tag, to_tag = 0.0, 0.0
    else:
        motion_seg = motion_all
        f_tag, to_tag = 0.0, 0.0

    return {
        "motion": motion_seg,
        "frames": int(motion_seg.shape[0]),
        "motion_path": motion_path,
        "text_path": "",
        "matched_caption": "",
        "f_tag": float(f_tag),
        "to_tag": float(to_tag),
    }


@torch.no_grad()
def to_xyz_joints(motion_denorm: np.ndarray, joints_num: int, device: torch.device) -> np.ndarray:
    motion_t = torch.from_numpy(motion_denorm).float().unsqueeze(0).to(device)
    xyz = recover_from_ric(motion_t, joints_num)[0].detach().cpu().numpy().astype(np.float32)
    return xyz


def normalize_methods(methods: List[str]) -> List[str]:
    valid = {"mbert", "clip"}
    out: List[str] = []
    seen = set()
    for method in methods:
        m = method.strip().lower()
        if m not in valid:
            raise ValueError(f"Unsupported method: {method}. Supported: {sorted(valid)}")
        if m not in seen:
            out.append(m)
            seen.add(m)
    if not out:
        raise ValueError("No valid methods provided.")
    return out


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample HumanML3D test captions, translate EN->ZH, generate motions with selected "
            "methods (mBERT projection and/or original CLIP), then summarize metrics to CSV/JSON."
        )
    )
    parser.add_argument("--hml3d-root", type=str, default="dataset/HumanML3D")
    parser.add_argument("--split-file", type=str, default="test.txt", help="Split file under HumanML3D root.")
    parser.add_argument("--max-pairs", type=int, default=100, help="Number of EN-ZH prompt pairs to process.")
    parser.add_argument("--shuffle-samples", action="store_true", help="Shuffle test candidates before taking first N.")
    parser.add_argument(
        "--sample-csv",
        type=str,
        default="dataset/HumanML3D/test_bilingual_eval_100.csv",
        help=(
            "Fixed sample CSV path. If file exists, evaluator reads it directly. "
            "If missing, evaluator samples/translates and writes this file."
        ),
    )
    parser.add_argument(
        "--regenerate-sample-csv",
        action="store_true",
        help="Force regenerate fixed sample CSV from split and overwrite existing file.",
    )
    parser.add_argument("--out-dir", type=str, default="output_eval_bilingual/HML3D_TEST100_EN_ZH")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["mbert", "clip"],
        help="Methods to evaluate. Any subset of: mbert clip",
    )
    parser.add_argument(
        "--comparison-standard",
        type=str,
        default="clip",
        help="Standard baseline method for comparison deltas.",
    )

    parser.add_argument("--vq-ckpt", type=str, default="pretrained/VQVAE/net_last.pth")
    parser.add_argument("--trans-ckpt", type=str, default="pretrained/VQTransformer_corruption05/net_best_fid.pth")
    parser.add_argument("--meta-dir", type=str, default="checkpoints/t2m/Comp_v6_KLD005/meta")
    parser.add_argument("--evaluator-opt", type=str, default="checkpoints/t2m/Comp_v6_KLD005/opt.txt")
    parser.add_argument(
        "--hml3d-fps",
        type=float,
        default=20.0,
        help="HumanML3D fps used to map caption f_tag/to_tag to frame indices.",
    )
    parser.add_argument(
        "--translator-model-name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-zh",
        help="Hugging Face EN->ZH translation model name/path.",
    )
    parser.add_argument(
        "--translator-batch-size",
        type=int,
        default=16,
        help="Batch size for EN->ZH translation.",
    )

    parser.add_argument("--mbert-model-name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument(
        "--proj-ckpt",
        type=str,
        default="",
        help=(
            "Projection or joint mBERT+projection checkpoint path. "
            "If empty, auto-resolve with HML3D last2-proj priority."
        ),
    )

    parser.add_argument(
        "--save-motion-npy",
        action="store_true",
        help="Save generated denormalized motion arrays for each prompt.",
    )

    args = parser.parse_args()
    methods = normalize_methods(args.methods)
    comparison_standard = args.comparison_standard.strip().lower()
    if comparison_standard not in methods:
        raise ValueError(
            f"comparison standard '{args.comparison_standard}' must be in methods={methods}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on a GPU node/session or set --device cpu.")

    set_seed(args.seed)

    split_path = os.path.join(args.hml3d_root, args.split_file)
    hml_motion_dir = os.path.join(args.hml3d_root, "new_joint_vecs")
    hml_text_dir = os.path.join(args.hml3d_root, "texts")
    sample_csv_path = args.sample_csv

    ensure_exists(args.hml3d_root)
    ensure_exists(split_path)
    ensure_exists(hml_motion_dir)
    ensure_exists(hml_text_dir)
    ensure_exists(args.vq_ckpt)
    ensure_exists(args.trans_ckpt)
    ensure_exists(args.evaluator_opt)
    ensure_exists(os.path.join(args.meta_dir, "mean.npy"))
    ensure_exists(os.path.join(args.meta_dir, "std.npy"))

    proj_ckpt_path = ""
    if "mbert" in methods:
        proj_ckpt_path = resolve_proj_ckpt(args.proj_ckpt)
        ensure_exists(proj_ckpt_path)

    os.makedirs(args.out_dir, exist_ok=True)
    motions_dir = os.path.join(args.out_dir, "motions")
    if args.save_motion_npy:
        os.makedirs(motions_dir, exist_ok=True)

    device = torch.device(args.device)

    # Build shared generation backbone.
    vq_model = text2motion_clip.build_vq_model(device, args.vq_ckpt)
    trans_model = text2motion_clip.build_trans_model(device, args.trans_ckpt)

    method_resources: Dict[str, Dict[str, object]] = {}
    if "mbert" in methods:
        text_encoder = text2motion_mbert.MBertTextEncoder(
            model_name=args.mbert_model_name,
            out_dim=512,
            max_length=args.max_text_len,
            freeze_mbert=True,
        ).to(device)
        text2motion_mbert.load_projection_weights(text_encoder, proj_ckpt_path, device)
        text_encoder.eval()
        method_resources["mbert"] = {"text_encoder": text_encoder}

    if "clip" in methods:
        clip_model, _ = text2motion_clip.clip.load("ViT-B/32", device=device, jit=False)
        text2motion_clip.clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        method_resources["clip"] = {"clip_model": clip_model}

    # Build motion embedding evaluator for generated motions.
    eval_opt = get_opt(args.evaluator_opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)
    max_motion_len = int(eval_opt.max_motion_length)

    mean = np.load(os.path.join(args.meta_dir, "mean.npy")).astype(np.float32)
    std = np.load(os.path.join(args.meta_dir, "std.npy")).astype(np.float32)

    pairs = load_or_create_sample_pairs(
        sample_csv_path=sample_csv_path,
        split_path=split_path,
        text_dir=hml_text_dir,
        max_pairs=args.max_pairs,
        seed=args.seed,
        shuffle_samples=args.shuffle_samples,
        regenerate_sample_csv=args.regenerate_sample_csv,
        translator_model_name=args.translator_model_name,
        translator_batch_size=args.translator_batch_size,
        max_text_len=args.max_text_len,
        device=device,
    )
    if not pairs:
        raise RuntimeError("No valid prompt pairs found from fixed sample CSV / HumanML3D split.")

    method_pair_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}
    method_prompt_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}
    method_failed_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}

    method_motion_embs: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    method_text_cos_values: Dict[str, List[float]] = {m: [] for m in methods}
    method_motion_cos_values: Dict[str, List[float]] = {m: [] for m in methods}
    method_motion_mse_values: Dict[str, List[float]] = {m: [] for m in methods}
    method_frame_gap_values: Dict[str, List[float]] = {m: [] for m in methods}
    method_runtime_sec: Dict[str, float] = {m: 0.0 for m in methods}

    # Extra metrics with HumanML3D ground-truth motion as reference standard.
    gt_ref_metric_names = [
        "gt_ref_clip_en_xyz_cos",
        "gt_ref_clip_en_xyz_mse",
        "gt_ref_clip_en_frame_gap",
        "gt_ref_clip_zh_xyz_cos",
        "gt_ref_clip_zh_xyz_mse",
        "gt_ref_clip_zh_frame_gap",
        "gt_ref_mbert_en_xyz_cos",
        "gt_ref_mbert_en_xyz_mse",
        "gt_ref_mbert_en_frame_gap",
        "gt_ref_mbert_zh_xyz_cos",
        "gt_ref_mbert_zh_xyz_mse",
        "gt_ref_mbert_zh_frame_gap",
    ]
    gt_ref_metric_values: Dict[str, List[float]] = {name: [] for name in gt_ref_metric_names}
    gt_ref_pair_rows: List[Dict[str, object]] = []

    run_start = time.time()

    @torch.no_grad()
    def encode_text_feature(prompt: str, method: str) -> torch.Tensor:
        if method == "mbert":
            text_encoder = method_resources["mbert"]["text_encoder"]
            return text_encoder.encode_text([prompt], device=device)

        clip_model = method_resources["clip"]["clip_model"]
        text_tokens = text2motion_clip.clip.tokenize([prompt], truncate=True).to(device)
        return clip_model.encode_text(text_tokens).float()

    @torch.no_grad()
    def generate_motion(prompt: str, method: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        text_feat = encode_text_feature(prompt, method)
        motion_tokens = trans_model.sample(text_feat, if_categorial=False)
        pred_motion = vq_model.forward_decoder(motion_tokens)

        motion_norm = pred_motion[0].detach().cpu().numpy().astype(np.float32)
        motion_denorm = motion_norm * std + mean
        text_vec = text_feat[0].detach().cpu().numpy().astype(np.float32)
        return text_vec, motion_denorm, int(motion_denorm.shape[0]), int(motion_tokens.shape[1])

    @torch.no_grad()
    def to_motion_embedding(motion_denorm: np.ndarray) -> Tuple[np.ndarray, int]:
        motion_norm = (motion_denorm - mean) / std
        motion_dim = int(motion_norm.shape[1])
        use_len = min(max_motion_len, int(motion_norm.shape[0]))

        padded = np.zeros((max_motion_len, motion_dim), dtype=np.float32)
        padded[:use_len] = motion_norm[:use_len]

        motions = torch.from_numpy(padded).unsqueeze(0).to(device)
        m_lens = torch.tensor([use_len], dtype=torch.long, device=device)
        emb = eval_wrapper.get_motion_embeddings(motions, m_lens)[0].detach().cpu().numpy().astype(np.float32)
        return emb, use_len

    for row in pairs:
        pair_id = row["pair_id"]
        motion_id = row["motion_id"]
        en_prompt = row["en_prompt"]
        zh_prompt = row["zh_prompt"]
        f_tag = float(row["f_tag"])
        to_tag = float(row["to_tag"])
        source_text_path = str(row["text_path"])
        try:
            gt_info = load_gt_motion_segment(
                motion_dir=hml_motion_dir,
                motion_id=motion_id,
                f_tag=f_tag,
                to_tag=to_tag,
                fps=args.hml3d_fps,
            )
            gt_motion = gt_info["motion"]
            gt_xyz = to_xyz_joints(gt_motion, joints_num=22, device=device)
        except Exception as exc:
            err = f"GT motion lookup failed (HumanML3D): {exc}"
            for method in methods:
                method_failed_rows[method].append(
                    {
                        "method": method,
                        "pair_id": pair_id,
                        "motion_id": motion_id,
                        "en_prompt": en_prompt,
                        "zh_prompt": zh_prompt,
                        "error": err,
                    }
                )
            continue

        pair_generated: Dict[str, Dict[str, Dict[str, object]]] = {}

        for method in methods:
            method_start = time.time()
            try:
                en_text_vec, en_motion, en_frames, en_tokens = generate_motion(en_prompt, method)
                zh_text_vec, zh_motion, zh_frames, zh_tokens = generate_motion(zh_prompt, method)

                en_emb, en_eval_len = to_motion_embedding(en_motion)
                zh_emb, zh_eval_len = to_motion_embedding(zh_motion)
                en_xyz = to_xyz_joints(en_motion, joints_num=22, device=device)
                zh_xyz = to_xyz_joints(zh_motion, joints_num=22, device=device)

                text_cos = cosine_sim(en_text_vec, zh_text_vec)
                motion_cos = cosine_sim(en_emb, zh_emb)
                compare_len = min(en_frames, zh_frames)
                motion_mse = float(np.mean((en_motion[:compare_len] - zh_motion[:compare_len]) ** 2))
                frame_gap = float(abs(en_frames - zh_frames))

                en_motion_path = ""
                zh_motion_path = ""
                if args.save_motion_npy:
                    en_motion_path = os.path.join(
                        motions_dir,
                        f"pair{pair_id}_{method}_en_motion_263.npy",
                    )
                    zh_motion_path = os.path.join(
                        motions_dir,
                        f"pair{pair_id}_{method}_zh_motion_263.npy",
                    )
                    np.save(en_motion_path, en_motion)
                    np.save(zh_motion_path, zh_motion)

                method_prompt_rows[method].append(
                    {
                        "method": method,
                        "pair_id": pair_id,
                        "motion_id": motion_id,
                        "language": "en",
                        "prompt": en_prompt,
                        "frames": en_frames,
                        "token_count": en_tokens,
                        "motion_path": en_motion_path,
                    }
                )
                method_prompt_rows[method].append(
                    {
                        "method": method,
                        "pair_id": pair_id,
                        "motion_id": motion_id,
                        "language": "zh",
                        "prompt": zh_prompt,
                        "frames": zh_frames,
                        "token_count": zh_tokens,
                        "motion_path": zh_motion_path,
                    }
                )

                method_pair_rows[method].append(
                    {
                        "method": method,
                        "pair_id": pair_id,
                        "motion_id": motion_id,
                        "en_prompt": en_prompt,
                        "zh_prompt": zh_prompt,
                        "text_cos": text_cos,
                        "motion_cos": motion_cos,
                        "motion_mse": motion_mse,
                        "frame_gap": frame_gap,
                        "en_frames": en_frames,
                        "zh_frames": zh_frames,
                        "en_eval_frames": en_eval_len,
                        "zh_eval_frames": zh_eval_len,
                        "en_motion_path": en_motion_path,
                        "zh_motion_path": zh_motion_path,
                    }
                )

                method_text_cos_values[method].append(text_cos)
                method_motion_cos_values[method].append(motion_cos)
                method_motion_mse_values[method].append(motion_mse)
                method_frame_gap_values[method].append(frame_gap)
                method_motion_embs[method].append(en_emb)
                method_motion_embs[method].append(zh_emb)

                pair_generated.setdefault(method, {})
                pair_generated[method]["en"] = {
                    "motion": en_motion,
                    "emb": en_emb,
                    "xyz": en_xyz,
                    "frames": en_frames,
                    "eval_frames": en_eval_len,
                }
                pair_generated[method]["zh"] = {
                    "motion": zh_motion,
                    "emb": zh_emb,
                    "xyz": zh_xyz,
                    "frames": zh_frames,
                    "eval_frames": zh_eval_len,
                }
            except Exception as exc:
                method_failed_rows[method].append(
                    {
                        "method": method,
                        "pair_id": pair_id,
                        "motion_id": motion_id,
                        "en_prompt": en_prompt,
                        "zh_prompt": zh_prompt,
                        "error": str(exc),
                    }
                )
            finally:
                method_runtime_sec[method] += float(time.time() - method_start)

        target_lookup = {
            "clip_en": pair_generated.get("clip", {}).get("en"),
            "clip_zh": pair_generated.get("clip", {}).get("zh"),
            "mbert_en": pair_generated.get("mbert", {}).get("en"),
            "mbert_zh": pair_generated.get("mbert", {}).get("zh"),
        }

        pair_row: Dict[str, object] = {
            "pair_id": pair_id,
            "motion_id": motion_id,
            "en_prompt": en_prompt,
            "zh_prompt": zh_prompt,
            "gt_motion_path": str(gt_info["motion_path"]),
            "gt_text_path": source_text_path,
            "gt_caption_matched": en_prompt,
            "gt_f_tag": float(gt_info["f_tag"]),
            "gt_to_tag": float(gt_info["to_tag"]),
            "gt_frames": int(gt_info["frames"]),
            "gt_joints": 22,
            "gt_ref_clip_en_frames": "",
            "gt_ref_clip_zh_frames": "",
            "gt_ref_mbert_en_frames": "",
            "gt_ref_mbert_zh_frames": "",
            "gt_ref_clip_en_xyz_cos": "",
            "gt_ref_clip_en_xyz_mse": "",
            "gt_ref_clip_en_frame_gap": "",
            "gt_ref_clip_zh_xyz_cos": "",
            "gt_ref_clip_zh_xyz_mse": "",
            "gt_ref_clip_zh_frame_gap": "",
            "gt_ref_mbert_en_xyz_cos": "",
            "gt_ref_mbert_en_xyz_mse": "",
            "gt_ref_mbert_en_frame_gap": "",
            "gt_ref_mbert_zh_xyz_cos": "",
            "gt_ref_mbert_zh_xyz_mse": "",
            "gt_ref_mbert_zh_frame_gap": "",
        }

        for target_name, target_value in target_lookup.items():
            if target_value is None:
                continue

            tgt_xyz = target_value["xyz"][:, :22, :]
            xyz_cos, xyz_mse, frame_gap = sequence_similarity_deviation(gt_xyz, tgt_xyz)
            frame_name = f"gt_ref_{target_name}_frames"
            cos_name = f"gt_ref_{target_name}_xyz_cos"
            mse_name = f"gt_ref_{target_name}_xyz_mse"
            gap_name = f"gt_ref_{target_name}_frame_gap"

            pair_row[frame_name] = int(target_value["frames"])
            pair_row[cos_name] = xyz_cos
            pair_row[mse_name] = xyz_mse
            pair_row[gap_name] = frame_gap

            gt_ref_metric_values[cos_name].append(xyz_cos)
            gt_ref_metric_values[mse_name].append(xyz_mse)
            gt_ref_metric_values[gap_name].append(frame_gap)

        gt_ref_pair_rows.append(pair_row)

    if not any(method_pair_rows[m] for m in methods):
        raise RuntimeError("No successful pair generation for any method.")

    total_runtime = float(time.time() - run_start)

    gt_ref_summary_values: Dict[str, object] = {
        "gt_ref_num_pairs": len(gt_ref_pair_rows),
    }
    for metric_name in gt_ref_metric_names:
        gt_ref_summary_values[f"{metric_name}_mean"] = safe_mean(gt_ref_metric_values[metric_name])
        gt_ref_summary_values[f"{metric_name}_std"] = safe_std(gt_ref_metric_values[metric_name])

    summary_rows: List[Dict[str, object]] = []
    for method in methods:
        pair_rows = method_pair_rows[method]
        prompt_rows = method_prompt_rows[method]
        failed_rows = method_failed_rows[method]

        if pair_rows:
            all_motion_embs_np = np.stack(method_motion_embs[method], axis=0)
            summary_rows.append(
                {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_csv": sample_csv_path,
                    "out_dir": args.out_dir,
                    "method": method,
                    "status": "ok",
                    "comparison_standard": comparison_standard,
                    "proj_ckpt": proj_ckpt_path if method == "mbert" else "",
                    "num_pairs_requested": len(pairs),
                    "num_pairs_success": len(pair_rows),
                    "num_pairs_failed": len(failed_rows),
                    "num_prompts_success": len(prompt_rows),
                    "text_pair_cos_mean": safe_mean(method_text_cos_values[method]),
                    "text_pair_cos_std": safe_std(method_text_cos_values[method]),
                    "motion_pair_cos_mean": safe_mean(method_motion_cos_values[method]),
                    "motion_pair_cos_std": safe_std(method_motion_cos_values[method]),
                    "motion_pair_mse_mean": safe_mean(method_motion_mse_values[method]),
                    "motion_pair_mse_std": safe_std(method_motion_mse_values[method]),
                    "frame_gap_mean": safe_mean(method_frame_gap_values[method]),
                    "frame_gap_std": safe_std(method_frame_gap_values[method]),
                    "motion_embedding_diversity_l2": pairwise_mean_l2(all_motion_embs_np),
                    "runtime_sec_method": method_runtime_sec[method],
                    "runtime_sec_total": total_runtime,
                    **gt_ref_summary_values,
                }
            )
        else:
            summary_rows.append(
                {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_csv": sample_csv_path,
                    "out_dir": args.out_dir,
                    "method": method,
                    "status": "failed_all",
                    "comparison_standard": comparison_standard,
                    "proj_ckpt": proj_ckpt_path if method == "mbert" else "",
                    "num_pairs_requested": len(pairs),
                    "num_pairs_success": 0,
                    "num_pairs_failed": len(failed_rows),
                    "num_prompts_success": 0,
                    "text_pair_cos_mean": 0.0,
                    "text_pair_cos_std": 0.0,
                    "motion_pair_cos_mean": 0.0,
                    "motion_pair_cos_std": 0.0,
                    "motion_pair_mse_mean": 0.0,
                    "motion_pair_mse_std": 0.0,
                    "frame_gap_mean": 0.0,
                    "frame_gap_std": 0.0,
                    "motion_embedding_diversity_l2": 0.0,
                    "runtime_sec_method": method_runtime_sec[method],
                    "runtime_sec_total": total_runtime,
                    **gt_ref_summary_values,
                }
            )

    summary_by_method = {row["method"]: row for row in summary_rows if row["status"] == "ok"}
    comparison_rows: List[Dict[str, object]] = []
    if comparison_standard in summary_by_method:
        std = summary_by_method[comparison_standard]
        for method in methods:
            if method == comparison_standard or method not in summary_by_method:
                continue
            cur = summary_by_method[method]
            comparison_rows.append(
                {
                    "standard_method": comparison_standard,
                    "compare_method": method,
                    "delta_text_pair_cos_mean": float(cur["text_pair_cos_mean"] - std["text_pair_cos_mean"]),
                    "delta_motion_pair_cos_mean": float(cur["motion_pair_cos_mean"] - std["motion_pair_cos_mean"]),
                    "delta_motion_pair_mse_mean": float(cur["motion_pair_mse_mean"] - std["motion_pair_mse_mean"]),
                    "delta_frame_gap_mean": float(cur["frame_gap_mean"] - std["frame_gap_mean"]),
                    "delta_motion_embedding_diversity_l2": float(
                        cur["motion_embedding_diversity_l2"] - std["motion_embedding_diversity_l2"]
                    ),
                }
            )

    pair_rows_all = [row for method in methods for row in method_pair_rows[method]]
    prompt_rows_all = [row for method in methods for row in method_prompt_rows[method]]
    failed_rows_all = [row for method in methods for row in method_failed_rows[method]]

    pair_csv = os.path.join(args.out_dir, "pair_metrics.csv")
    prompt_csv = os.path.join(args.out_dir, "prompt_outputs.csv")
    summary_csv = os.path.join(args.out_dir, "evaluation_summary.csv")
    summary_json = os.path.join(args.out_dir, "evaluation_summary.json")
    comparison_csv = os.path.join(args.out_dir, "model_comparison.csv")
    gt_ref_pair_csv = os.path.join(args.out_dir, "gt_reference_pair_metrics.csv")
    failed_csv = os.path.join(args.out_dir, "failed_pairs.csv")

    write_csv(
        pair_csv,
        pair_rows_all,
        [
            "method",
            "pair_id",
            "motion_id",
            "en_prompt",
            "zh_prompt",
            "text_cos",
            "motion_cos",
            "motion_mse",
            "frame_gap",
            "en_frames",
            "zh_frames",
            "en_eval_frames",
            "zh_eval_frames",
            "en_motion_path",
            "zh_motion_path",
        ],
    )
    write_csv(
        prompt_csv,
        prompt_rows_all,
        ["method", "pair_id", "motion_id", "language", "prompt", "frames", "token_count", "motion_path"],
    )
    write_csv(
        summary_csv,
        summary_rows,
        [
            "created_at",
            "dataset_csv",
            "out_dir",
            "method",
            "status",
            "comparison_standard",
            "proj_ckpt",
            "num_pairs_requested",
            "num_pairs_success",
            "num_pairs_failed",
            "num_prompts_success",
            "text_pair_cos_mean",
            "text_pair_cos_std",
            "motion_pair_cos_mean",
            "motion_pair_cos_std",
            "motion_pair_mse_mean",
            "motion_pair_mse_std",
            "frame_gap_mean",
            "frame_gap_std",
            "motion_embedding_diversity_l2",
            "runtime_sec_method",
            "runtime_sec_total",
            "gt_ref_num_pairs",
            "gt_ref_clip_en_xyz_cos_mean",
            "gt_ref_clip_en_xyz_cos_std",
            "gt_ref_clip_en_xyz_mse_mean",
            "gt_ref_clip_en_xyz_mse_std",
            "gt_ref_clip_en_frame_gap_mean",
            "gt_ref_clip_en_frame_gap_std",
            "gt_ref_clip_zh_xyz_cos_mean",
            "gt_ref_clip_zh_xyz_cos_std",
            "gt_ref_clip_zh_xyz_mse_mean",
            "gt_ref_clip_zh_xyz_mse_std",
            "gt_ref_clip_zh_frame_gap_mean",
            "gt_ref_clip_zh_frame_gap_std",
            "gt_ref_mbert_en_xyz_cos_mean",
            "gt_ref_mbert_en_xyz_cos_std",
            "gt_ref_mbert_en_xyz_mse_mean",
            "gt_ref_mbert_en_xyz_mse_std",
            "gt_ref_mbert_en_frame_gap_mean",
            "gt_ref_mbert_en_frame_gap_std",
            "gt_ref_mbert_zh_xyz_cos_mean",
            "gt_ref_mbert_zh_xyz_cos_std",
            "gt_ref_mbert_zh_xyz_mse_mean",
            "gt_ref_mbert_zh_xyz_mse_std",
            "gt_ref_mbert_zh_frame_gap_mean",
            "gt_ref_mbert_zh_frame_gap_std",
        ],
    )

    write_csv(
        gt_ref_pair_csv,
        gt_ref_pair_rows,
        [
            "pair_id",
            "motion_id",
            "en_prompt",
            "zh_prompt",
            "gt_motion_path",
            "gt_text_path",
            "gt_caption_matched",
            "gt_f_tag",
            "gt_to_tag",
            "gt_frames",
            "gt_joints",
            "gt_ref_clip_en_frames",
            "gt_ref_clip_zh_frames",
            "gt_ref_mbert_en_frames",
            "gt_ref_mbert_zh_frames",
            "gt_ref_clip_en_xyz_cos",
            "gt_ref_clip_en_xyz_mse",
            "gt_ref_clip_en_frame_gap",
            "gt_ref_clip_zh_xyz_cos",
            "gt_ref_clip_zh_xyz_mse",
            "gt_ref_clip_zh_frame_gap",
            "gt_ref_mbert_en_xyz_cos",
            "gt_ref_mbert_en_xyz_mse",
            "gt_ref_mbert_en_frame_gap",
            "gt_ref_mbert_zh_xyz_cos",
            "gt_ref_mbert_zh_xyz_mse",
            "gt_ref_mbert_zh_frame_gap",
        ],
    )

    if comparison_rows:
        write_csv(
            comparison_csv,
            comparison_rows,
            [
                "standard_method",
                "compare_method",
                "delta_text_pair_cos_mean",
                "delta_motion_pair_cos_mean",
                "delta_motion_pair_mse_mean",
                "delta_frame_gap_mean",
                "delta_motion_embedding_diversity_l2",
            ],
        )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_rows,
                "comparison": comparison_rows,
                "gt_reference_summary": gt_ref_summary_values,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if failed_rows_all:
        write_csv(
            failed_csv,
            failed_rows_all,
            ["method", "pair_id", "motion_id", "en_prompt", "zh_prompt", "error"],
        )

    print("BILINGUAL_EVAL_DONE")
    for summary in summary_rows:
        print("method:", summary["method"])
        print("status:", summary["status"])
        print("pairs_success:", summary["num_pairs_success"])
        print("pairs_failed:", summary["num_pairs_failed"])
        print("prompts_success:", summary["num_prompts_success"])
        print("text_pair_cos_mean:", f"{summary['text_pair_cos_mean']:.6f}")
        print("motion_pair_cos_mean:", f"{summary['motion_pair_cos_mean']:.6f}")
        print("motion_pair_mse_mean:", f"{summary['motion_pair_mse_mean']:.6f}")
        print("motion_embedding_diversity_l2:", f"{summary['motion_embedding_diversity_l2']:.6f}")
    print("summary_csv:", summary_csv)
    print("summary_json:", summary_json)
    print("sample_pairs_csv:", sample_csv_path)
    print("pair_metrics_csv:", pair_csv)
    print("prompt_outputs_csv:", prompt_csv)
    print("gt_reference_pair_metrics_csv:", gt_ref_pair_csv)
    if comparison_rows:
        print("model_comparison_csv:", comparison_csv)
    if failed_rows_all:
        print("failed_pairs_csv:", failed_csv)


if __name__ == "__main__":
    main()
