"""
Patched evaluation code for T2M-GPT with mT5 encoder.

This replaces the CLIP calls in the original eval_trans.py.
Instead of clip.tokenize() + clip_model.encode_text(),
we call mt5_encoder(text_list) directly.

Only the text encoding lines are changed. Everything else is identical.
"""

import os
import torch
import numpy as np
from scipy import linalg


# ================================================
# Utility functions (unchanged from original)
# ================================================

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
    plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
    writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter,
                           best_fid, best_iter, best_div, best_top1, best_top2,
                           best_top3, best_matching, clip_model=None, eval_wrapper=None,
                           draw=True, save=True, savegif=False, savenpy=False):
    """
    clip_model: in our case this is MT5CLIPWrapper, which has .encode_text()
                that accepts a list of strings directly.
    """
    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    multimodality = 0

    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        # === KEY CHANGE: use mT5 instead of CLIP ===
        # Original:
        #   text = clip.tokenize(clip_text, truncate=True).cuda()
        #   feat_clip_text = clip_model.encode_text(text).float()
        # Ours:
        feat_clip_text = clip_model.encode_text(list(clip_text)).float()

        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()

        for k in range(bs):
            try:
                index_motion = trans.sample(feat_clip_text[k:k+1], False)
            except:
                index_motion = torch.ones(1, 1).cuda().long()

            pred_pose = net.forward_decoder(index_motion)
            cur_len = pred_pose.shape[1]

            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            pred_len[k] = min(cur_len, seq)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )

        # R-precision (real)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        # R-precision (pred)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300)
    diversity = calculate_diversity(motion_pred_np, 300)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Iter {nb_iter} :, "
        f"FID. {fid:.4f}, "
        f"&Diversity Real. {diversity_real:.4f}, "
        f"Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, "
        f"R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, "
        f"matching_score_pred. {matching_score_pred:.4f}"
    )
    logger.info(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    if fid < best_fid:
        msg = f"--> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({
                'trans': trans.state_dict(),
            }, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


# ================================================
# Metric functions (unchanged from original)
# ================================================

def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmin = np.argsort(dist_mat, axis=1)
    top_k_mat = np.zeros_like(dist_mat, dtype=bool)
    for i in range(top_k):
        top_k_mat[np.arange(dist_mat.shape[0]), argmin[:, i]] = True

    R_count = np.zeros(top_k)
    for i in range(top_k):
        R_count[i] = (argmin[:, i] == np.arange(dist_mat.shape[0])).sum()

    if sum_all:
        return R_count
    return R_count / dist_mat.shape[0]


def euclidean_distance_matrix(matrix1, matrix2):
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(np.maximum(d1 + d2 + d3, 0.0))
    return dists


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()
