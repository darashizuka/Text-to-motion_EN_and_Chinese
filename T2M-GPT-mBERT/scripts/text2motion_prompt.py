#!/usr/bin/env python3
import argparse
import os
import sys
import time
from types import SimpleNamespace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import clip
import numpy as np
import torch

import models.t2m_trans as trans
import models.vqvae as vqvae
from utils.motion_process import recover_from_ric
from visualization.plot_3d_global import draw_to_batch


def build_vq_model(device: torch.device, ckpt_path: str) -> torch.nn.Module:
    cfg = SimpleNamespace(dataname="t2m", quantizer="ema_reset", mu=0.99)
    model = vqvae.HumanVQVAE(
        cfg,
        nb_code=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["net"], strict=True)
    model.eval()
    return model


def build_trans_model(device: torch.device, ckpt_path: str) -> torch.nn.Module:
    model = trans.Text2Motion_Transformer(
        num_vq=512,
        embed_dim=1024,
        clip_dim=512,
        block_size=51,
        num_layers=9,
        n_head=16,
        drop_out_rate=0.1,
        fc_rate=4,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["trans"], strict=True)
    model.eval()
    return model


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a motion from a single text prompt using official t2m pretrained weights.")
    parser.add_argument("--text", type=str, required=True, help="Input text prompt, e.g. 'a person walks forward and waves'.")
    parser.add_argument("--out-dir", type=str, default="demo_outputs", help="Output directory for generated files.")
    parser.add_argument("--name", type=str, default="", help="Optional output prefix. Defaults to timestamp.")
    parser.add_argument("--vq-ckpt", type=str, default="pretrained/VQVAE/net_last.pth", help="Path to VQVAE checkpoint.")
    parser.add_argument("--trans-ckpt", type=str, default="pretrained/VQTransformer_corruption05/net_best_fid.pth", help="Path to transformer checkpoint.")
    parser.add_argument("--meta-dir", type=str, default="checkpoints/t2m/Comp_v6_KLD005/meta", help="Path containing mean.npy and std.npy for t2m denormalization.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in current shell. Run this on a GPU node/session.")

    device = torch.device(args.device)

    ensure_exists(args.vq_ckpt)
    ensure_exists(args.trans_ckpt)
    ensure_exists(os.path.join(args.meta_dir, "mean.npy"))
    ensure_exists(os.path.join(args.meta_dir, "std.npy"))

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name if args.name else time.strftime("%Y%m%d_%H%M%S")

    vq_model = build_vq_model(device, args.vq_ckpt)
    trans_model = build_trans_model(device, args.trans_ckpt)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    text_tokens = clip.tokenize([args.text], truncate=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens).float()
        motion_tokens = trans_model.sample(text_feat, if_categorial=False)
        pred_motion = vq_model.forward_decoder(motion_tokens)

    motion_norm = pred_motion[0].detach().cpu().numpy()
    mean = np.load(os.path.join(args.meta_dir, "mean.npy"))
    std = np.load(os.path.join(args.meta_dir, "std.npy"))
    motion_denorm = motion_norm * std + mean

    motion_t = torch.from_numpy(motion_denorm).float().unsqueeze(0).to(device)
    xyz = recover_from_ric(motion_t, 22)[0].detach().cpu().numpy()

    motion_path = os.path.join(args.out_dir, f"{name}_motion_263.npy")
    xyz_path = os.path.join(args.out_dir, f"{name}_xyz_22j.npy")
    gif_path = os.path.join(args.out_dir, f"{name}.gif")

    np.save(motion_path, motion_denorm)
    np.save(xyz_path, xyz)
    draw_to_batch(np.expand_dims(xyz, axis=0), title_batch=[args.text], outname=[gif_path])

    print("TEXT2MOTION_OK")
    print("device:", device)
    print("prompt:", args.text)
    print("frames:", motion_denorm.shape[0])
    print("tokens:", tuple(motion_tokens.shape))
    print("motion_npy:", motion_path)
    print("xyz_npy:", xyz_path)
    print("gif:", gif_path)


if __name__ == "__main__":
    main()
