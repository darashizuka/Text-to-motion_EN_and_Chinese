#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time
from types import SimpleNamespace

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

import models.t2m_trans as trans
import models.vqvae as vqvae
from utils.motion_process import recover_from_ric
from visualization.plot_3d_global import draw_to_batch


class MBertTextEncoder(torch.nn.Module):
    """mBERT encoder with a 768->512 projection for T2M-GPT cond input."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        out_dim: int = 512,
        max_length: int = 64,
        freeze_mbert: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for mBERT text encoding. "
                "Install it with: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mbert = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.mbert.config.hidden_size)
        self.proj = torch.nn.Linear(hidden_size, out_dim)
        self.max_length = max_length

        if freeze_mbert:
            self.mbert.eval()
            for p in self.mbert.parameters():
                p.requires_grad = False

    def _masked_mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_text(self, texts, device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = self.mbert(**batch)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = self._masked_mean_pool(outputs.last_hidden_state, batch["attention_mask"])

        return self.proj(pooled.float())


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


AUTO_PROJ_SENTINEL = "__AUTO_PROJ_CKPT__"


def resolve_proj_ckpt_arg(proj_ckpt_arg: str) -> str:
    if not proj_ckpt_arg:
        return ""
    if proj_ckpt_arg != AUTO_PROJ_SENTINEL:
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
        "Pass an explicit path via --proj-ckpt /path/to/proj_best.pth."
    )


def load_projection_weights(encoder: MBertTextEncoder, proj_ckpt_path: str, device: torch.device) -> dict:
    state = torch.load(proj_ckpt_path, map_location="cpu")
    proj_state = None
    mbert_state = None

    if isinstance(state, dict):
        if "proj" in state and isinstance(state["proj"], dict):
            proj_state = state["proj"]
        elif "weight" in state and "bias" in state:
            # Legacy format: direct nn.Linear state_dict.
            proj_state = state
        elif any(k.startswith("proj.") for k in state.keys()):
            proj_state = {k[len("proj."):]: v for k, v in state.items() if k.startswith("proj.")}

        if "mbert_trainable" in state and isinstance(state["mbert_trainable"], dict):
            mbert_state = state["mbert_trainable"]
        elif "mbert_last2" in state and isinstance(state["mbert_last2"], dict):
            mbert_state = state["mbert_last2"]
        elif "mbert" in state and isinstance(state["mbert"], dict):
            mbert_state = state["mbert"]
        elif any(k.startswith("mbert.") for k in state.keys()):
            mbert_state = {k[len("mbert."):]: v for k, v in state.items() if k.startswith("mbert.")}
    else:
        proj_state = state

    if proj_state is None:
        raise RuntimeError(
            f"Unsupported projection checkpoint format: {proj_ckpt_path}. "
            "Expected key 'proj' or direct linear state_dict with {weight,bias}."
        )

    encoder.proj.load_state_dict(proj_state, strict=True)
    encoder.proj.to(device)

    mbert_loaded = False
    if mbert_state is not None:
        encoder.mbert.load_state_dict(mbert_state, strict=False)
        mbert_loaded = True

    return {
        "proj_loaded": True,
        "mbert_loaded": mbert_loaded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a motion from a single text prompt using mBERT + 768->512 projection + "
            "official t2m pretrained weights. Supports projection-only or joint mBERT+projection checkpoints."
        )
    )
    parser.add_argument("--text", type=str, required=True, help="Input text prompt, e.g. 'a person walks forward and waves'.")
    parser.add_argument("--out-dir", type=str, default="demo_outputs", help="Output directory for generated files.")
    parser.add_argument("--name", type=str, default="", help="Optional output prefix. Defaults to timestamp.")
    parser.add_argument("--vq-ckpt", type=str, default="pretrained/VQVAE/net_last.pth", help="Path to VQVAE checkpoint.")
    parser.add_argument(
        "--trans-ckpt",
        type=str,
        default="pretrained/VQTransformer_corruption05/net_best_fid.pth",
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--meta-dir",
        type=str,
        default="checkpoints/t2m/Comp_v6_KLD005/meta",
        help="Path containing mean.npy and std.npy for t2m denormalization.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference.")
    parser.add_argument(
        "--mbert-model-name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Hugging Face model name/path for mBERT.",
    )
    parser.add_argument("--max-text-len", type=int, default=64, help="Max token length for mBERT tokenizer.")
    parser.add_argument(
        "--proj-ckpt",
        type=str,
        nargs="?",
        const=AUTO_PROJ_SENTINEL,
        default="",
        help=(
            "Optional path to projection-only or joint mBERT+projection checkpoint. "
            "If provided without a value, auto-resolve from output_projection."
        ),
    )
    parser.add_argument(
        "--trainable-mbert",
        action="store_true",
        help="Enable gradients for mBERT parameters. Default is frozen mBERT.",
    )
    args = parser.parse_args()
    proj_ckpt_path = resolve_proj_ckpt_arg(args.proj_ckpt)

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

    text_encoder = MBertTextEncoder(
        model_name=args.mbert_model_name,
        out_dim=512,
        max_length=args.max_text_len,
        freeze_mbert=not args.trainable_mbert,
    ).to(device)

    load_info = {"proj_loaded": False, "mbert_loaded": False}
    if proj_ckpt_path:
        ensure_exists(proj_ckpt_path)
        load_info = load_projection_weights(text_encoder, proj_ckpt_path, device)

    text_encoder.eval()

    with torch.no_grad():
        text_feat = text_encoder.encode_text([args.text], device=device)
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

    print("TEXT2MOTION_M_BERT_OK")
    print("device:", device)
    print("prompt:", args.text)
    print("mbert:", args.mbert_model_name)
    print("proj:", "loaded" if load_info["proj_loaded"] else "random_init")
    print("mbert_adapt:", "loaded" if load_info["mbert_loaded"] else "base")
    print("proj_ckpt:", proj_ckpt_path if proj_ckpt_path else "none")
    print("frames:", motion_denorm.shape[0])
    print("tokens:", tuple(motion_tokens.shape))
    print("motion_npy:", motion_path)
    print("xyz_npy:", xyz_path)
    print("gif:", gif_path)


if __name__ == "__main__":
    main()
