"""
Training script for T2M-GPT with mT5 text encoder.

This is a modified version of train_t2m_trans.py.
Changes from original:
  - CLIP replaced with MT5TextEncoder
  - clip.tokenize() replaced with mt5_encoder(captions)
  - Added mT5 parameters to optimizer (so projection layer trains)

Usage (run from inside T2M-GPT directory):
  python train_mt5.py --resume-pth ./pretrained/VQVAE/net_best_fid.pth
"""

import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json

# Add T2M-GPT to path so we can import its modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import eval_trans_mt5 as eval_trans
from dataset import dataset_TM_train, dataset_TM_eval, dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

# Import our mT5 encoder
from mt5_encoder import MT5TextEncoder

import traceback


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# Create output directory for this experiment
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir = os.path.join(
    "./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D",
    f'{args.vq_name}'
)
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.vq_dir, exist_ok=True)


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


##### ---- Dataloader ---- #####
# This loader is used ONCE to tokenize all motions with the VQ-VAE
train_loader_token = dataset_tokenize.DATALoader(
    args.dataname, 1, unit_length=2**args.down_t
)

# This loader is used for evaluation (needs word vectors for the evaluator)
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

# Load evaluation wrapper (pre-trained feature extractor for computing FID, etc.)
dataset_opt_path = (
    'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit'
    else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
)
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Network ---- #####

# 1. mT5 text encoder (REPLACES CLIP)
#    - freeze_mt5=False means we fine-tune mT5 along with the projection layer
#    - output_dim=512 to match what the GPT decoder expects (clip_dim)
mt5_encoder = MT5TextEncoder(
    model_name="google/mt5-small",
    output_dim=args.clip_dim,  # 512 by default
    freeze_mt5=False
)
mt5_encoder.train()
mt5_encoder.cuda()
logger.info("Loaded mT5 text encoder (google/mt5-small)")

# 2. VQ-VAE (frozen, pre-trained)
net = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate
)
print('loading VQ-VAE checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

# 3. GPT decoder (trainable)
trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=args.embed_dim_gpt,
    clip_dim=args.clip_dim,      # mT5 output matches this dim
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate
)
if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    if 'mt5_encoder' in ckpt:
        mt5_encoder.load_state_dict(ckpt['mt5_encoder'], strict=True)
        print('loaded mT5 encoder weights from checkpoint')
trans_encoder.train()
trans_encoder.cuda()


##### ---- Optimizer ---- #####
# IMPORTANT: we optimize BOTH the GPT decoder AND the mT5 encoder
# The original only optimized the GPT because CLIP was frozen
optimizer = torch.optim.AdamW(
    list(trans_encoder.parameters()) + list(mt5_encoder.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args.lr_scheduler, gamma=args.gamma
)


##### ---- Loss ---- #####
loss_ce = torch.nn.CrossEntropyLoss()


##### ---- Tokenize all motions with VQ-VAE (one-time step) ---- #####
nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

# Check if tokenization is already done (skip if resuming)
existing_tokens = os.listdir(args.vq_dir) if os.path.exists(args.vq_dir) else []
if len(existing_tokens) > 100:
    print(f'Skipping tokenization — {len(existing_tokens)} tokens already cached in {args.vq_dir}')
else:
    print('Tokenizing motions with VQ-VAE...')
    for batch in train_loader_token:
        pose, name = batch
        pose = pose.cuda().float()
        target = net.encode(pose)
        target = target.cpu().numpy()
        np.save(pjoin(args.vq_dir, name[0] + '.npy'), target)


##### ---- Training data loader ---- #####
train_loader = dataset_TM_train.DATALoader(
    args.dataname, args.batch_size, args.nb_code,
    args.vq_name, unit_length=2**args.down_t
)
train_loader_iter = dataset_TM_train.cycle(train_loader)


##### ---- Initial evaluation ---- #####
best_fid, best_iter, best_div = 1000, 0, 100
best_top1, best_top2, best_top3, best_matching = 0, 0, 0, 100

# For evaluation, we need a function that encodes text the same way CLIP did
# The eval code passes clip_model and calls clip_model.encode_text()
# We wrap mt5_encoder to have a compatible interface
class MT5CLIPWrapper:
    """
    Wraps MT5TextEncoder so eval_trans.py can use it like CLIP.
    eval_trans.py calls: clip_model.encode_text(clip.tokenize(text))
    We intercept this by making the wrapper accept raw text instead.
    """
    def __init__(self, encoder):
        self.encoder = encoder

    def encode_text(self, texts):
        """
        texts: either a list of strings OR pre-tokenized tensor
        If tensor (from clip.tokenize), we can't use it — but since we
        modified the eval code to pass raw text, this receives strings.
        """
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(texts)

mt5_wrapper = MT5CLIPWrapper(mt5_encoder)

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(
    args.out_dir, val_loader, net, trans_encoder, logger, writer, 0,
    best_fid=best_fid, best_iter=best_iter, best_div=best_div,
    best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
    best_matching=best_matching, clip_model=mt5_wrapper, eval_wrapper=eval_wrapper
)


##### ---- Training loop ---- #####
while nb_iter <= args.total_iter:
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    # clip_text is a list of caption strings like ["a person walks forward", ...]
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens.cuda()

    # === THIS IS THE KEY CHANGE ===
    # Original: text = clip.tokenize(clip_text).cuda()
    #           feat_clip_text = clip_model.encode_text(text).float()
    # Ours:     directly encode the captions with mT5
    feat_clip_text = mt5_encoder(list(clip_text))

    # Rest is identical to original T2M-GPT training
    input_index = target[:, :-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(
            input_index.shape, device=input_index.device
        ))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(
            input_index.shape, device=input_index.device
        ))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask * input_index + (1 - mask) * r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text)
    cls_pred = cls_pred.contiguous()

    loss_cls = 0.0
    for i in range(bs):
        loss_cls += loss_ce(
            cls_pred[i][:m_tokens_len[i] + 1],
            target[i][:m_tokens_len[i] + 1]
        ) / bs

        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)
        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)
        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (
            cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)
        ).sum().item()

    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()
    nb_iter += 1

    if nb_iter % args.print_iter == 0:
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter == 0:
        mt5_encoder.eval()
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(
            args.out_dir, val_loader, net, trans_encoder, logger, writer,
            nb_iter, best_fid, best_iter, best_div, best_top1, best_top2,
            best_top3, best_matching, clip_model=mt5_wrapper,
            eval_wrapper=eval_wrapper
        )
        mt5_encoder.train()

        # Save checkpoint
        torch.save({
            'trans': trans_encoder.state_dict(),
            'mt5_encoder': mt5_encoder.state_dict(),
        }, os.path.join(args.out_dir, 'net_last.pth'))

    if nb_iter == args.total_iter:
        msg_final = (
            f"Train. Iter {best_iter} : FID. {best_fid:.5f}, "
            f"Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, "
            f"TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        )
        logger.info(msg_final)
        break
