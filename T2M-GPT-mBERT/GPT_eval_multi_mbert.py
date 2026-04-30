import argparse
import importlib.util
import json
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import models.t2m_trans as trans
import models.vqvae as vqvae
import options.option_transformer as option_trans
import utils.eval_trans as eval_trans
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.motion_process import recover_from_ric

warnings.filterwarnings("ignore")


def parse_args():
    # Parse extra mBERT args first, then reuse the original option_transformer args
    # so CLI flags remain aligned with GPT_eval_multi.py.
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    extra_parser.add_argument("--mbert-model-name", type=str, default="bert-base-multilingual-cased")
    extra_parser.add_argument("--mbert-proj-ckpt", type=str, default="")
    extra_parser.add_argument("--max-text-len", type=int, default=64)
    extra_parser.add_argument("--repeat-time", type=int, default=20)

    extra_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining
        base_args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    for key, value in vars(extra_args).items():
        setattr(base_args, key, value)

    return base_args


def load_mbert_helpers(project_root):
    helper_path = os.path.join(project_root, "scripts", "text2motion_prompt_mbert.py")
    spec = importlib.util.spec_from_file_location("text2motion_prompt_mbert", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load mBERT helper module: {helper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_vqvae_down_t(state_dict):
    """Infer VQVAE temporal downsampling depth from encoder final conv key."""
    prefix = "vqvae.encoder.model."
    final_conv_indices = []
    for key in state_dict:
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        suffix = key[len(prefix) :]
        parts = suffix.split(".")
        if len(parts) == 2 and parts[1] == "weight" and parts[0].isdigit():
            final_conv_indices.append(int(parts[0]))

    if not final_conv_indices:
        return None

    # Encoder layout is [conv, relu, down blocks..., final conv], so final index
    # equals down_t + 2.
    return max(final_conv_indices) - 2


@torch.no_grad()
def evaluation_transformer_test_mbert(
    out_dir,
    val_loader,
    net,
    trans_encoder,
    logger,
    writer,
    eval_wrapper,
    text_encoder,
    device,
    savenpy=False,
):
    trans_encoder.eval()
    total_batches = len(val_loader)

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []

    r_precision_real = 0
    r_precision_pred = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0

    for batch_idx, batch in enumerate(val_loader, start=1):
        word_embeddings, pos_one_hots, captions, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        print(
            f"[eval] batch {batch_idx}/{total_batches}: batch_size={bs}, seq_len={seq}",
            flush=True,
        )

        feat_text = text_encoder.encode_text(list(captions), device=device).float()

        motion_multimodality_batch = []
        for i in range(30):
            if i == 0 or (i + 1) % 5 == 0 or i == 29:
                print(
                    f"[eval] batch {batch_idx}/{total_batches}: sampling {i + 1}/30",
                    flush=True,
                )

            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1]), device=device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans_encoder.sample(feat_text[k : k + 1], True)
                except Exception:
                    index_motion = torch.ones(1, 1, device=device).long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k : k + 1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and savenpy:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(device), num_joints)
                    np.save(os.path.join(out_dir, name[k] + "_pred.npy"), pred_xyz.detach().cpu().numpy())

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings,
                pos_one_hots,
                sent_len,
                pred_pose_eval,
                pred_len,
            )
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose_device = pose.to(device).float()
                et, em = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pose_device,
                    m_length,
                )

                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if savenpy:
                    pose_denorm = val_loader.dataset.inv_transform(pose_device.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose_denorm).float().to(device), num_joints)
                    for j in range(bs):
                        cur_m_len = int(m_length[j])
                        np.save(
                            os.path.join(out_dir, name[j] + "_gt.npy"),
                            pose_xyz[j][:cur_m_len].unsqueeze(0).cpu().numpy(),
                        )

                temp_r, temp_match = eval_trans.calculate_R_precision(
                    et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
                )
                r_precision_real += temp_r
                matching_score_real += temp_match

                temp_r, temp_match = eval_trans.calculate_R_precision(
                    et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
                )
                r_precision_pred += temp_r
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = eval_trans.calculate_activation_statistics(motion_annotation_np)
    mu, cov = eval_trans.calculate_activation_statistics(motion_pred_np)

    diversity_real = eval_trans.calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = eval_trans.calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    r_precision_real = r_precision_real / nb_sample
    r_precision_pred = r_precision_pred / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    motion_multimodality_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = eval_trans.calculate_multimodality(motion_multimodality_np, 10)

    fid = eval_trans.calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t mBERT Eval: FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, "
        f"Diversity. {diversity:.4f}, R_precision_real. {r_precision_real}, "
        f"R_precision. {r_precision_pred}, matching_score_real. {matching_score_real}, "
        f"matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    )
    logger.info(msg)

    trans_encoder.train()
    return (
        fid,
        diversity,
        r_precision_pred[0],
        r_precision_pred[1],
        r_precision_pred[2],
        matching_score_pred,
        multimodality,
        writer,
        logger,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in current shell. Use --device cpu or run on a GPU node.")

    args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    from utils.word_vectorizer import WordVectorizer

    w_vectorizer = WordVectorizer("./glove", "our_vab")
    val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

    dataset_opt_path = (
        "checkpoints/kit/Comp_v6_KLD005/opt.txt"
        if args.dataname == "kit"
        else "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    )

    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    print("loading checkpoint from {}".format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    inferred_down_t = infer_vqvae_down_t(ckpt["net"])
    if inferred_down_t is not None and inferred_down_t != args.down_t:
        logger.info(
            "Adjusting VQVAE down_t from %s to %s to match checkpoint %s",
            args.down_t,
            inferred_down_t,
            args.resume_pth,
        )
        args.down_t = inferred_down_t

    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
    )

    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=args.num_layers,
        n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate,
    )

    net.load_state_dict(ckpt["net"], strict=True)
    net.eval()
    net.to(device)

    if args.resume_trans is not None:
        print("loading transformer checkpoint from {}".format(args.resume_trans))
        ckpt = torch.load(args.resume_trans, map_location="cpu")
        trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.train()
    trans_encoder.to(device)

    project_root = os.path.dirname(os.path.abspath(__file__))
    mbert_helpers = load_mbert_helpers(project_root)

    if args.mbert_proj_ckpt:
        proj_ckpt_path = mbert_helpers.resolve_proj_ckpt_arg(args.mbert_proj_ckpt)
    else:
        proj_ckpt_path = mbert_helpers.resolve_proj_ckpt_arg(mbert_helpers.AUTO_PROJ_SENTINEL)

    mbert_helpers.ensure_exists(proj_ckpt_path)

    text_encoder = mbert_helpers.MBertTextEncoder(
        model_name=args.mbert_model_name,
        out_dim=args.clip_dim,
        max_length=args.max_text_len,
        freeze_mbert=True,
    ).to(device)

    load_info = mbert_helpers.load_projection_weights(text_encoder, proj_ckpt_path, device)
    text_encoder.eval()

    logger.info(
        "mBERT checkpoint loaded: proj_ckpt=%s, proj_loaded=%s, mbert_adapt_loaded=%s",
        proj_ckpt_path,
        load_info.get("proj_loaded", False),
        load_info.get("mbert_loaded", False),
    )

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    multi = []

    repeat_time = max(1, int(args.repeat_time))

    for i in range(repeat_time):
        print(f"[eval] repeat {i + 1}/{repeat_time} started", flush=True)
        (
            cur_fid,
            cur_div,
            cur_top1,
            cur_top2,
            cur_top3,
            cur_matching,
            cur_multi,
            writer,
            logger,
        ) = evaluation_transformer_test_mbert(
            args.out_dir,
            val_loader,
            net,
            trans_encoder,
            logger,
            writer,
            eval_wrapper,
            text_encoder,
            device,
            savenpy=(i == 0),
        )
        print(f"[eval] repeat {i + 1}/{repeat_time} finished: fid={cur_fid:.4f}", flush=True)

        fid.append(cur_fid)
        div.append(cur_div)
        top1.append(cur_top1)
        top2.append(cur_top2)
        top3.append(cur_top3)
        matching.append(cur_matching)
        multi.append(cur_multi)

    print("final result:")
    print("fid: ", sum(fid) / repeat_time)
    print("div: ", sum(div) / repeat_time)
    print("top1: ", sum(top1) / repeat_time)
    print("top2: ", sum(top2) / repeat_time)
    print("top3: ", sum(top3) / repeat_time)
    print("matching: ", sum(matching) / repeat_time)
    print("multi: ", sum(multi) / repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    multi = np.array(multi)

    msg_final = (
        f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, "
        f"Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, "
        f"TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, "
        f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, "
        f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, "
        f"Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, "
        f"Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
    )
    logger.info(msg_final)


if __name__ == "__main__":
    main()
