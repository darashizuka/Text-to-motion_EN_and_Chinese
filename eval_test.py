"""
Final evaluation on the TEST set.

Run this AFTER training is done to get your final reported numbers.
Uses the best checkpoint (net_best_fid.pth).

Usage (from T2M-GPT directory):
  python eval_test.py --dataname t2m --resume-pth ./pretrained/VQVAE/net_best_fid.pth \
      --down-t 2 --depth 3 --dilation-growth-rate 3 --vq-act relu --block-size 51
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options.option_transformer as option_trans
import models.vqvae as vqvae
import models.t2m_trans as trans
from dataset import dataset_TM_eval
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from eval_trans_mt5 import evaluation_transformer
from mt5_encoder import MT5TextEncoder

args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

##### ---- Load evaluation data (TEST set) ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

# is_test=True loads test.txt instead of val.txt
test_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = (
    'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit'
    else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
)
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Load models ---- #####

# mT5 encoder
mt5_encoder = MT5TextEncoder(
    model_name="google/mt5-small",
    output_dim=args.clip_dim,
    freeze_mt5=True
)

# VQ-VAE
net = vqvae.HumanVQVAE(
    args, args.nb_code, args.code_dim, args.output_emb_width,
    args.down_t, args.stride_t, args.width, args.depth,
    args.dilation_growth_rate
)
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

# GPT decoder
trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
    clip_dim=args.clip_dim, block_size=args.block_size,
    num_layers=args.num_layers, n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate
)

# Load best checkpoint
best_ckpt_path = './output_GPT_Final/mt5_t2m/net_last.pth'
print(f'Loading best checkpoint from {best_ckpt_path}')
ckpt = torch.load(best_ckpt_path, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
if 'mt5_encoder' in ckpt:
    mt5_encoder.load_state_dict(ckpt['mt5_encoder'], strict=True)
    print('Loaded mT5 encoder weights')

trans_encoder.eval()
trans_encoder.cuda()
mt5_encoder.eval()
mt5_encoder.cuda()

##### ---- Evaluate on TEST set ---- #####

class MT5CLIPWrapper:
    def __init__(self, encoder):
        self.encoder = encoder
    def encode_text(self, texts):
        with torch.no_grad():
            return self.encoder(texts)

mt5_wrapper = MT5CLIPWrapper(mt5_encoder)

print('\n' + '='*60)
print('EVALUATING ON TEST SET')
print('='*60 + '\n')

# Run evaluation multiple times and average (reduces variance)
all_fid, all_top1, all_top2, all_top3, all_div, all_match = [], [], [], [], [], []

num_runs = 5
for run in range(num_runs):
    print(f'\n--- Run {run+1}/{num_runs} ---')
    torch.manual_seed(args.seed + run)

    import utils.utils_model as utils_model
    os.makedirs('./test_eval_output', exist_ok=True)
    logger = utils_model.get_logger('./test_eval_output')

    class DummyWriter:
        def add_scalar(self, *args, **kwargs): pass

    fid, _, div, top1, top2, top3, match, _, _ = evaluation_transformer(
        './test_eval_output', test_loader, net, trans_encoder,
        logger, DummyWriter(), 0,
        best_fid=1000, best_iter=0, best_div=100,
        best_top1=0, best_top2=0, best_top3=0,
        best_matching=100, clip_model=mt5_wrapper,
        eval_wrapper=eval_wrapper, save=False
    )

    all_fid.append(fid)
    all_top1.append(top1)
    all_top2.append(top2)
    all_top3.append(top3)
    all_div.append(div)
    all_match.append(match)

print('\n' + '='*60)
print('FINAL TEST SET RESULTS (averaged over {} runs)'.format(num_runs))
print('='*60)
print(f'FID:             {np.mean(all_fid):.4f} ± {np.std(all_fid):.4f}')
print(f'R@1:             {np.mean(all_top1):.4f} ± {np.std(all_top1):.4f}')
print(f'R@2:             {np.mean(all_top2):.4f} ± {np.std(all_top2):.4f}')
print(f'R@3:             {np.mean(all_top3):.4f} ± {np.std(all_top3):.4f}')
print(f'Diversity:       {np.mean(all_div):.4f} ± {np.std(all_div):.4f}')
print(f'Matching Score:  {np.mean(all_match):.4f} ± {np.std(all_match):.4f}')
print('='*60)
