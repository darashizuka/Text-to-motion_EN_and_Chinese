#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple

import clip
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def collate_text(batch: List[str]) -> List[str]:
    return batch


def parse_motion_texts(data_root: str, subset_size: int, split_file: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    split_path = os.path.join(data_root, split_file)
    text_dir = os.path.join(data_root, "texts")

    with open(split_path, "r", encoding="utf-8") as f:
        all_ids = [line.strip() for line in f if line.strip()]

    selected_ids = all_ids if subset_size <= 0 else all_ids[:subset_size]
    captions: List[str] = []
    skipped_missing = 0

    for sid in selected_ids:
        txt_path = os.path.join(text_dir, f"{sid}.txt")
        if not os.path.exists(txt_path):
            skipped_missing += 1
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # HumanML3D/KIT text line format: caption#tokens#f_tag#to_tag
                caption = line.split("#", 1)[0].strip()
                if caption:
                    captions.append(caption)

    stats = {
        "split_file": split_file,
        "source_motion_count": len(all_ids),
        "requested_motion_count": len(all_ids) if subset_size <= 0 else subset_size,
        "selected_motion_count": len(selected_ids),
        "missing_text_file_count": skipped_missing,
        "caption_count": len(captions),
    }
    return captions, selected_ids, stats


def split_texts(texts: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = max(1, int(len(texts) * val_ratio))
    val_idx = set(idx[:val_n])

    train_texts = [texts[i] for i in range(len(texts)) if i not in val_idx]
    val_texts = [texts[i] for i in range(len(texts)) if i in val_idx]
    return train_texts, val_texts


def masked_mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def encode_clip(clip_model: torch.nn.Module, texts: List[str], device: torch.device) -> torch.Tensor:
    text_tokens = clip.tokenize(texts, truncate=True).to(device)
    clip_feat = clip_model.encode_text(text_tokens).float()
    return clip_feat


@torch.no_grad()
def encode_mbert(
    tokenizer: AutoTokenizer,
    mbert_model: AutoModel,
    texts: List[str],
    device: torch.device,
    max_text_len: int,
) -> torch.Tensor:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = mbert_model(**batch)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = masked_mean_pool(outputs.last_hidden_state, batch["attention_mask"])
    return pooled.float()


def projection_losses(
    projected: torch.Tensor,
    clip_feat: torch.Tensor,
    lambda_mse: float,
    lambda_cos: float,
) -> Tuple[torch.Tensor, float, float]:
    proj_n = F.normalize(projected, dim=-1)
    clip_n = F.normalize(clip_feat, dim=-1)
    loss_mse = F.mse_loss(proj_n, clip_n)
    loss_cos = (1.0 - F.cosine_similarity(proj_n, clip_n, dim=-1)).mean()
    loss = lambda_mse * loss_mse + lambda_cos * loss_cos
    cos_sim = float(1.0 - loss_cos.detach().cpu())
    mse_val = float(loss_mse.detach().cpu())
    return loss, mse_val, cos_sim


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    clip_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    mbert_model: AutoModel,
    projection: nn.Module,
    device: torch.device,
    max_text_len: int,
    lambda_mse: float,
    lambda_cos: float,
) -> Dict[str, float]:
    projection.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    total_n = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for texts in pbar:
        clip_feat = encode_clip(clip_model, texts, device)
        mbert_feat = encode_mbert(tokenizer, mbert_model, texts, device, max_text_len)
        projected = projection(mbert_feat)
        loss, mse_val, cos_val = projection_losses(projected, clip_feat, lambda_mse, lambda_cos)

        bs = len(texts)
        total_loss += float(loss.detach().cpu()) * bs
        total_mse += mse_val * bs
        total_cos += cos_val * bs
        total_n += bs

    if total_n == 0:
        return {"loss": 0.0, "mse": 0.0, "cos": 0.0}

    return {
        "loss": total_loss / total_n,
        "mse": total_mse / total_n,
        "cos": total_cos / total_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train only mBERT->CLIP projection on motion captions with frozen encoders "
            "(default: full HumanML3D all.txt)."
        )
    )
    parser.add_argument("--data-root", type=str, default="dataset/HumanML3D", help="Path to dataset root (e.g. dataset/HumanML3D).")
    parser.add_argument("--split-file", type=str, default="all.txt", help="Split file under data root (e.g. all.txt/train.txt).")
    parser.add_argument("--subset-size", type=int, default=-1, help="Use first N motion IDs from split file. <=0 means use all IDs.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio over caption samples.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-cos", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mbert-model-name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--out-dir", type=str, default="output_projection")
    parser.add_argument("--exp-name", type=str, default="HML3D_mBERT_proj")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU node/session or use --device cpu.")

    set_seed(args.seed)
    device = torch.device(args.device)

    run_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    captions, selected_ids, ds_stats = parse_motion_texts(args.data_root, args.subset_size, args.split_file)
    if len(captions) < 10:
        raise RuntimeError(f"Too few captions parsed from {args.data_root}; got {len(captions)}")

    train_texts, val_texts = split_texts(captions, args.val_ratio, args.seed)
    train_loader = DataLoader(
        TextDataset(train_texts),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_text,
    )
    val_loader = DataLoader(
        TextDataset(val_texts),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_text,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.mbert_model_name)
    mbert_model = AutoModel.from_pretrained(args.mbert_model_name).to(device)
    mbert_model.eval()
    for p in mbert_model.parameters():
        p.requires_grad = False

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    projection = nn.Linear(int(mbert_model.config.hidden_size), 512).to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    info = {
        "args": vars(args),
        "dataset_stats": ds_stats,
        "train_caption_count": len(train_texts),
        "val_caption_count": len(val_texts),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "selected_motion_ids.txt"), "w", encoding="utf-8") as f:
        for sid in selected_ids:
            f.write(sid + "\n")

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best_cos = -1.0

    print("[proj-train] run_dir:", run_dir)
    print("[proj-train] dataset stats:", ds_stats)
    print("[proj-train] train/val captions:", len(train_texts), len(val_texts))

    for epoch in range(1, args.epochs + 1):
        projection.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        sum_loss = 0.0
        sum_mse = 0.0
        sum_cos = 0.0
        seen = 0

        for texts in pbar:
            with torch.no_grad():
                clip_feat = encode_clip(clip_model, texts, device)
                mbert_feat = encode_mbert(tokenizer, mbert_model, texts, device, args.max_text_len)

            projected = projection(mbert_feat)
            loss, mse_val, cos_val = projection_losses(projected, clip_feat, args.lambda_mse, args.lambda_cos)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = len(texts)
            seen += bs
            sum_loss += float(loss.detach().cpu()) * bs
            sum_mse += mse_val * bs
            sum_cos += cos_val * bs

            pbar.set_postfix({
                "loss": f"{sum_loss / seen:.4f}",
                "mse": f"{sum_mse / seen:.4f}",
                "cos": f"{sum_cos / seen:.4f}",
            })

        train_metrics = {
            "loss": sum_loss / max(seen, 1),
            "mse": sum_mse / max(seen, 1),
            "cos": sum_cos / max(seen, 1),
        }
        val_metrics = evaluate(
            val_loader,
            clip_model,
            tokenizer,
            mbert_model,
            projection,
            device,
            args.max_text_len,
            args.lambda_mse,
            args.lambda_cos,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        ckpt_last = {
            "proj": projection.state_dict(),
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "config": vars(args),
        }
        torch.save(ckpt_last, os.path.join(run_dir, "proj_last.pth"))

        if val_metrics["cos"] > best_cos:
            best_cos = val_metrics["cos"]
            torch.save(ckpt_last, os.path.join(run_dir, "proj_best.pth"))

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} train_cos={train_metrics['cos']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_cos={val_metrics['cos']:.4f}"
        )

    print("[proj-train] done")
    print("[proj-train] best checkpoint:", os.path.join(run_dir, "proj_best.pth"))
    print("[proj-train] last checkpoint:", os.path.join(run_dir, "proj_last.pth"))
    print("[proj-train] metrics:", metrics_path)


if __name__ == "__main__":
    main()
