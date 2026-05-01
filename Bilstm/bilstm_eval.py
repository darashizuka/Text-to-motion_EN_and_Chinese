import os
for root, dirs, files in os.walk('/kaggle/input/datasets/angieesong/pretrained-t2mgpt'):
    for f in files:
        print(os.path.join(root, f))

import os
import subprocess

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.chdir('/kaggle/working/checkpoints')


subprocess.run(['gdown', '--fuzzy', 
    'https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view'])
subprocess.run(['gdown', '--fuzzy', 
    'https://drive.google.com/file/d/1KNU8CsMAnxFrwopKBBkC8jEULGLPBHQp/view'])


subprocess.run(['unzip', 't2m.zip'])
subprocess.run(['unzip', 'kit.zip'])

print('Done!')
os.chdir('/kaggle/working')

import subprocess
os.makedirs('/kaggle/working/glove', exist_ok=True)
os.chdir('/kaggle/working')

subprocess.run(['gdown', '--fuzzy',
    'https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view'])
subprocess.run(['unzip', 'glove.zip', '-d', 'glove/'])
print('Done!')

import os, sys

# ── paths ───────────────────────────────────────────────────────────────
REPO_DIR      = "/kaggle/working/T2M-GPT-mBERT"   
DATA_ROOT     = "/kaggle/input/datasets/mrriandmstique/humanml3d/HumanML3D/humanml"
CKPT_DIR      = "/kaggle/working/checkpoints"    
GLOVE_DIR     = "/kaggle/working/glove/glove"  
VQ_CKPT       = "/kaggle/input/datasets/angieesong/pretrained-t2mgpt/net_last.pth"
TRANS_CKPT    = "/kaggle/input/datasets/angieesong/pretrained-t2mgpt/net_best_fid.pth"
BILSTM_CKPT   = "/kaggle/input/datasets/angieesong/pretrained-t2mgpt/bilstm_proj_best.pth"
META_DIR      = os.path.join(CKPT_DIR, "t2m/Comp_v6_KLD005/meta")

sys.path.insert(0, REPO_DIR)
print("Paths set up.")

# clone Serena repo if not already there
import subprocess
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
        "https://github.com/Polarislaris/T2M-GPT-mBERT",
        REPO_DIR], check=True)
    print("Cloned repo.")
else:
    print("Repo already exists.")

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── BiLSTM encoder (same as training) ───────────────────────────────────
from transformers import AutoTokenizer, AutoModel

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, mbert_name="bert-base-multilingual-cased",
                 out_dim=512, max_length=64, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(mbert_name)
        _tmp = AutoModel.from_pretrained(mbert_name)
        self.token_emb = _tmp.embeddings.word_embeddings
        for p in self.token_emb.parameters():
            p.requires_grad = False
        del _tmp
        self.bilstm = nn.LSTM(self.token_emb.embedding_dim, hidden_dim,
                              num_layers, batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def encode_text(self, texts, device):
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        embeds = self.token_emb(input_ids)
        lstm_out, _ = self.bilstm(embeds)
        scores  = self.attn(lstm_out).squeeze(-1)
        scores  = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled  = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(pooled.float())


# load weights
text_encoder = BiLSTMTextEncoder().to(DEVICE)
text_encoder.load_state_dict(torch.load(BILSTM_CKPT, map_location="cpu"))
text_encoder.eval()
print("BiLSTM encoder loaded.")

# ── VQ-VAE + Transformer ─────────────────────────────────────────────────
import models.vqvae as vqvae
import models.t2m_trans as trans

cfg = SimpleNamespace(dataname="t2m", quantizer="ema_reset", mu=0.99)
vq_model = vqvae.HumanVQVAE(cfg, nb_code=512, code_dim=512, output_emb_width=512,
    down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
    activation="relu", norm=None).to(DEVICE)
ckpt = torch.load(VQ_CKPT, map_location="cpu")
vq_model.load_state_dict(ckpt["net"], strict=True)
vq_model.eval()

trans_model = trans.Text2Motion_Transformer(
    num_vq=512, embed_dim=1024, clip_dim=512, block_size=51,
    num_layers=9, n_head=16, drop_out_rate=0.1, fc_rate=4).to(DEVICE)
ckpt2 = torch.load(TRANS_CKPT, map_location="cpu")
trans_model.load_state_dict(ckpt2["trans"], strict=True)
trans_model.eval()
print("VQ-VAE and Transformer loaded.")

# ── Evaluator (for FID, R@1 etc.) ────────────────────────────────────────
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer

eval_opt = SimpleNamespace(
    dataset_name = "t2m",
    checkpoints_dir = CKPT_DIR,
    device = DEVICE,
    unit_length = 4,
    dim_movement_enc_hidden = 512,
    dim_movement_latent = 512,
)
wrapper = EvaluatorModelWrapper(eval_opt)

w_vectorizer = WordVectorizer(GLOVE_DIR, "our_vab")
print("Evaluator loaded.")

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HumanML3DDataset(Dataset):
    def __init__(self, data_root, split, vocab=None, max_seq_len=196, max_text_len=30, motion_dim=263):
        super().__init__()
        self.vocab = vocab
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.motion_dim = motion_dim
        
 
        self.text_path = os.path.join(data_root, "texts")
        self.motion_path = os.path.join(data_root, "new_joint_vecs") 
        
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file, 'r') as f:
            self.file_ids = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        

        text_file = os.path.join(self.text_path, f"{file_id}.txt")
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.readline().split('#')[0].strip()
        except:
            text = ""


        motion_file = os.path.join(self.motion_path, f"{file_id}.npy")
        try:
            motion = np.load(motion_file)
        except:
            motion = np.zeros((self.max_seq_len, self.motion_dim))
            

        seq_len = motion.shape[0]
        if seq_len >= self.max_seq_len:
            motion = motion[:self.max_seq_len, :]
        else:
            padding = np.zeros((self.max_seq_len - seq_len, self.motion_dim))
            motion = np.concatenate((motion, padding), axis=0)
        
        motion_tensor = torch.tensor(motion, dtype=torch.float32)


        if self.vocab is not None:
            token_ids = self.vocab.encode(text) 
            actual_text_len = len(token_ids)
            
            if actual_text_len >= self.max_text_len:
                token_ids = token_ids[:self.max_text_len]
                actual_text_len = self.max_text_len
            else:
                padding = [self.vocab.pad_idx] * (self.max_text_len - actual_text_len)
                token_ids = token_ids + padding
                
            tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
            return tokens_tensor, actual_text_len, motion_tensor, text
        

        return text, motion_tensor, seq_len
    

DATA_ROOT = "/kaggle/input/datasets/mrriandmstique/humanml3d/HumanML3D/humanml"


test_dataset = HumanML3DDataset(
    data_root=DATA_ROOT,
    split="test", 
    max_seq_len=196
)


val_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

print(f"total sample: {len(test_dataset)}")

# ── Generate motions ──────────────────────────────────────────────────────
all_gt_motions   = []
all_pred_motions = []
all_texts        = []
all_lengths      = []

mean_np = np.load(os.path.join(META_DIR, "mean.npy"))
std_np  = np.load(os.path.join(META_DIR, "std.npy"))


with torch.no_grad():
    for idx in tqdm(range(len(test_dataset)), desc="Generating"):
        text, gt_mot, length = test_dataset[idx]
        
        # encode text
        text_feat = text_encoder.encode_text([text], DEVICE)  # [1, 512]
        
        # generate tokens
        tokens = trans_model.sample(text_feat, if_categorial=False)  # [1, T_tok]
        

        pred = vq_model.forward_decoder(tokens)  # [1, T, 263]
        
        # denormalize
        gt_denorm   = gt_mot.numpy() * std_np + mean_np        # [T, 263]
        pred_denorm = pred[0].cpu().numpy() * std_np + mean_np # [T, 263]
        
        all_gt_motions.append(gt_denorm)
        all_pred_motions.append(pred_denorm)
        all_texts.append(text)
        all_lengths.append(length)

print(f"Generated {len(all_texts)} motions.")

import pickle
print(f"Generated {len(all_texts)} motions.")


save_dict = {
    "gt": all_gt_motions,
    "pred": all_pred_motions,
    "texts": all_texts,
    "lengths": all_lengths
}

with open("saved_motions.pkl", "wb") as f:
    pickle.dump(save_dict, f)

import pickle

with open("saved_motions.pkl", "rb") as f:
    loaded_data = pickle.load(f)
    
all_gt_motions   = loaded_data["gt"]
all_pred_motions = loaded_data["pred"]
all_texts        = loaded_data["texts"]
all_lengths      = loaded_data["lengths"]

# ── Compute FID, R@1, R@2, R@3, Diversity, Matching Score ───────────────
from scipy import linalg

def calc_fid(gt_embs, pred_embs):
    mu_gt,  cov_gt  = np.mean(gt_embs,   axis=0), np.cov(gt_embs,   rowvar=False)
    mu_pr,  cov_pr  = np.mean(pred_embs, axis=0), np.cov(pred_embs, rowvar=False)
    diff = mu_gt - mu_pr
    covmean, _ = linalg.sqrtm(cov_gt @ cov_pr, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(cov_gt + cov_pr - 2 * covmean))
    return fid

def calc_diversity(embs, num_samples=300):
    if len(embs) < 2: return 0.0
    idx = np.random.choice(len(embs), min(num_samples*2, len(embs)), replace=False)
    a, b = embs[idx[:len(idx)//2]], embs[idx[len(idx)//2:]]
    return float(np.linalg.norm(a - b, axis=1).mean())

def calc_recall_at_k(text_embs, motion_embs, ks=[1,2,3]):
    # cosine similarity matrix
    tn = text_embs  / (np.linalg.norm(text_embs,  axis=1, keepdims=True) + 1e-8)
    mn = motion_embs/ (np.linalg.norm(motion_embs, axis=1, keepdims=True) + 1e-8)
    sim = tn @ mn.T   # [N, N]
    results = {}
    for k in ks:
        topk = np.argsort(-sim, axis=1)[:, :k]
        hits = sum(i in topk[i] for i in range(len(sim)))
        results[f"R@{k}"] = hits / len(sim)
    return results

def calc_matching_score(text_embs, motion_embs):
    tn = text_embs  / (np.linalg.norm(text_embs,  axis=1, keepdims=True) + 1e-8)
    mn = motion_embs/ (np.linalg.norm(motion_embs, axis=1, keepdims=True) + 1e-8)
    return float((tn * mn).sum(axis=1).mean())

print("Metric functions defined.")

def get_embeddings(motions, lengths, wrapper):
    embs = []
    with torch.no_grad():
        for i in tqdm(range(len(motions)), desc="Extracting"):
            m = torch.from_numpy(motions[i]).float().to(DEVICE)
            if m.ndim == 2:
                m = m.unsqueeze(0)

            l = torch.tensor([lengths[i]]).to(DEVICE)
            
            em = wrapper.get_motion_embeddings(m, l)
            
            embs.append(em.cpu().numpy())
            
    return np.concatenate(embs, axis=0)

gt_embs = get_embeddings(all_gt_motions, all_lengths, wrapper)
pred_embs = get_embeddings(all_pred_motions, all_lengths, wrapper)

NUM_RUNS = 20  

fid_list, div_list, r1_list, r2_list, r3_list, match_list = [], [], [], [], [], []

for run in tqdm(range(NUM_RUNS), desc="Computing metrics"):

    fid   = calc_fid(gt_embs, pred_embs)
    div   = calc_diversity(pred_embs) 
    rec   = calc_recall_at_k(pred_embs, gt_embs)
    match = calc_matching_score(pred_embs, gt_embs)
    
    fid_list.append(fid)
    div_list.append(div)
    r1_list.append(rec['R@1'])
    r2_list.append(rec['R@2'])
    r3_list.append(rec['R@3'])
    match_list.append(match)

print("="*50)
print("BiLSTM + T2M-GPT Quantitative Results")
print("="*50)
print(f"FID:            {np.mean(fid_list):.4f} ± {np.std(fid_list):.4f}")
print(f"R@1:            {np.mean(r1_list):.4f} ± {np.std(r1_list):.4f}")
print(f"R@2:            {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"R@3:            {np.mean(r3_list):.4f} ± {np.std(r3_list):.4f}")
print(f"Diversity:      {np.mean(div_list):.4f} ± {np.std(div_list):.4f}")
print(f"Matching Score: {np.mean(match_list):.4f} ± {np.std(match_list):.4f}")
print("="*50)