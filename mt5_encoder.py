"""
mT5 Text Encoder for T2M-GPT

PURPOSE:
The original T2M-GPT uses CLIP to encode text into a 512-dim vector.
We replace CLIP with mT5's encoder to test how a multilingual
encoder affects motion generation quality.

FLOW:
    text string
        |
    mT5 tokenizer (text -> token IDs)
        |
    mT5 encoder (token IDs -> hidden states per token)
        |
    mean pooling (average all token hidden states into one vector)
        |
    projection layer (linear: mT5_dim -> 512)
        |
    512-dim text embedding (same shape CLIP would produce)
"""

import torch
import torch.nn as nn
from transformers import MT5EncoderModel, AutoTokenizer


class MT5TextEncoder(nn.Module):

    def __init__(self, model_name="google/mt5-small", output_dim=512, freeze_mt5=False):
        super().__init__()

        # Load the pre-trained mT5 encoder only (no decoder half)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = MT5EncoderModel.from_pretrained(model_name)

        # mT5-small: d_model=512, mT5-base: d_model=768
        mt5_hidden_size = self.encoder.config.d_model

        # Linear layer to project mT5 output to the size the GPT decoder expects
        self.projection = nn.Linear(mt5_hidden_size, output_dim)

        if freeze_mt5:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, texts):
        """
        Args:
            texts: list of strings, e.g. ["a person walks forward"]

        Returns:
            embeddings: (batch_size, output_dim) tensor
        """
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move to same device as model
        device = next(self.encoder.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Run through mT5 encoder -> (batch, seq_len, hidden_size)
        encoder_output = self.encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"]
        )
        hidden_states = encoder_output.last_hidden_state

        # Mean pool over real tokens only (ignore padding)
        mask = tokens["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
        summed = (hidden_states * mask).sum(dim=1)     # (batch, hidden_size)
        counts = mask.sum(dim=1)                       # (batch, 1)
        mean_pooled = summed / counts                  # (batch, hidden_size)

        # Project to output dimension
        embeddings = self.projection(mean_pooled)      # (batch, output_dim)

        return embeddings
