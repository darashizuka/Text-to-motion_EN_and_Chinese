import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel
from tqdm import tqdm
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input/datasets/mrriandmstique/humanml3d/HumanML3D/humanml"
TEXT_PATH    = os.path.join(DATA_ROOT, "texts")
FEATURE_PATH = os.path.join(DATA_ROOT, "new_joint_vecs")

# ── Hyperparameters ─────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 1e-4
MAX_TEXT_LEN = 64        # tokenizer max length
OUT_DIM      = 512       # must match T2M-GPT clip_dim
HIDDEN_DIM   = 512       # BiLSTM hidden dim per direction (output = 1024)
NUM_LAYERS   = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── Dataset ─────────────────────────────────────────────────────────────
# We only need TEXT for projection training.
# Target = CLIP embedding of that text.
# No motion data needed here.

class TextOnlyDataset(Dataset):
    def __init__(self, data_root, split):
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file, "r") as f:
            file_ids = [line.strip() for line in f if line.strip()]

        self.texts = []
        for fid in tqdm(file_ids, desc=f"Loading {split} texts"):
            text_path = os.path.join(data_root, "texts", f"{fid}.txt")
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    # each file has multiple captions, take first line
                    line = f.readline()
                    text = line.split("#")[0].strip()
                    if text:
                        self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]   # just a string


train_dataset = TextOnlyDataset(DATA_ROOT, "train")
val_dataset   = TextOnlyDataset(DATA_ROOT, "val")
print(f"Train texts: {len(train_dataset)}, Val texts: {len(val_dataset)}")

# ── CLIP (frozen, target embeddings) ────────────────────────────────────
# We use the same CLIP model T2M-GPT was trained with.
# Its output IS the 512-dim "standard" that T2M-GPT expects.

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)

# Freeze completely — we never train CLIP
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False


@torch.no_grad()
def get_clip_embeddings(texts, device):
    """Returns CLIP text embeddings: [B, 512]"""
    batch = clip_tokenizer(
        texts, padding=True, truncation=True,
        max_length=77, return_tensors="pt"
    ).to(device)
    feats = clip_model.get_text_features(**batch)  # [B, 512]

    if not isinstance(feats, torch.Tensor):
        feats = feats.last_hidden_state[:, 0, :]
        
    # L2-normalize (CLIP standard)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


# Quick sanity check
sample = get_clip_embeddings(["a person walks forward"], DEVICE)
print("CLIP output shape:", sample.shape)  # should be [1, 512]

# ── BiLSTM Text Encoder ─────────────────────────────────────────────────
# Uses mBERT tokenizer + frozen token embeddings as input.
# BiLSTM processes the sequence, attention pooling -> 1024-dim.
# Projection layer -> 512-dim (matches T2M-GPT clip_dim).

class BiLSTMTextEncoder(nn.Module):
    def __init__(
        self,
        mbert_name  = "bert-base-multilingual-cased",
        out_dim     = 512,
        max_length  = 64,
        hidden_dim  = 512,
        num_layers  = 2,
        dropout     = 0.1,
    ):
        super().__init__()
        self.max_length = max_length

        # tokenizer — handles EN + ZH, no custom vocab needed
        self.tokenizer = AutoTokenizer.from_pretrained(mbert_name)

        # grab only the embedding layer from mBERT, freeze it
        _tmp = AutoModel.from_pretrained(mbert_name)
        self.token_emb = _tmp.embeddings.word_embeddings  # [vocab, 768]
        embed_dim = self.token_emb.embedding_dim
        for p in self.token_emb.parameters():
            p.requires_grad = False
        del _tmp

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size  = embed_dim,          # 768
            hidden_size = hidden_dim,          # 512
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        # attention pooling
        self.attn = nn.Linear(hidden_dim * 2, 1)

        # projection: 1024 -> 512
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def encode_text(self, texts, device):
        """texts: list of str -> [B, out_dim]"""
        batch = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)


        embeds = self.token_emb(input_ids)     # [B, T, 768]

        lstm_out, _ = self.bilstm(embeds)          # [B, T, 1024]

        # attention-weighted pooling
        scores = self.attn(lstm_out).squeeze(-1)   # [B, T]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)    # [B, T]
        pooled = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # [B, 1024]

        return self.proj(pooled.float())           # [B, 512]

    def forward(self, texts, device):
        return self.encode_text(texts, device)


# sanity check
encoder = BiLSTMTextEncoder(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
test_out = encoder.encode_text(["a person walks forward"], DEVICE)
print("BiLSTM output shape:", test_out.shape)  # should be [1, 512]

# ── Training ─────────────────────────────────────────────────────────────
# Loss: MSE between BiLSTM projection output and CLIP embedding.
# Only BiLSTM + projection layer are trained. CLIP and mBERT embeddings frozen.

def run_epoch(encoder, loader, optimizer, device, training=True):
    encoder.train(training)
    total_loss = 0.0
    criterion = nn.MSELoss()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for texts in tqdm(loader, desc="train" if training else "val", leave=False):
            # texts is a list of strings (default collate for strings)
            clip_targets = get_clip_embeddings(texts, device)          # [B, 512] frozen
            bilstm_preds = encoder.encode_text(texts, device)          # [B, 512] trainable

            loss = criterion(bilstm_preds, clip_targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, collate_fn=lambda x: x)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=lambda x: x)

# only train BiLSTM + proj, NOT token embeddings
trainable_params = [p for p in encoder.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(encoder, train_loader, optimizer, DEVICE, training=True)
    val_loss   = run_epoch(encoder, val_loader,   optimizer, DEVICE, training=False)

    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(encoder.state_dict(), "/kaggle/working/bilstm_proj_best.pth")
        print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

torch.save(encoder.state_dict(), "/kaggle/working/bilstm_proj_last.pth")
print("Training complete. Files saved to /kaggle/working/")

# ── Verify checkpoint ────────────────────────────────────────────────────
ckpt = torch.load("/kaggle/working/bilstm_proj_best.pth", map_location="cpu")
print("Checkpoint keys:", list(ckpt.keys())[:5], "...")

# reload and test
encoder_reload = BiLSTMTextEncoder(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
encoder_reload.load_state_dict(ckpt)
encoder_reload.eval().to(DEVICE)

with torch.no_grad():
    out = encoder_reload.encode_text(["a person kicks a ball"], DEVICE)
print("Reload test output shape:", out.shape)   # [1, 512]
print("Done! Download bilstm_proj_best.pth from /kaggle/working/")