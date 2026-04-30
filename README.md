# Multilingual Text-to-Motion Generation with mT5 , mBERT and biLSTM

A text-to-motion generation system that replaces CLIP with Google's [mT5](https://huggingface.co/google/mt5-small) encoder, enabling motion generation from multilingual text prompts (English, Chinese, etc.). Built on top of [T2M-GPT](https://github.com/Mael-zys/T2M-GPT).

## Architecture

```
Text Prompt (any language) → mT5 Encoder → GPT-2 Decoder → VQ-VAE Decoder → Motion Sequence
```

- **mT5 Encoder** — Tokenizes multilingual text and produces 512-dim embeddings via mean pooling
- **GPT-2 Decoder** — Autoregressively generates discrete motion tokens from text embeddings
- **VQ-VAE Decoder** — Decodes token sequences into continuous 263-dim motion features (pre-trained, frozen)

## Results 

### MT5

Evaluated on the HumanML3D test set (averaged over 5 runs):

| Metric | Score |
|--------|-------|
| FID | 0.1857 ± 0.0070 |
| R@1 | 0.4704 ± 0.0075 |
| R@2 | 0.2014 ± 0.0075 |
| R@3 | 0.1036 ± 0.0046 |
| Diversity | 10.2409 ± 0.1629 |
| Matching Score | 2.9582 ± 0.0259 |

### mBERT
| Metric | Score |
|--------|-------|
| FID | 0.07336 |
| R@1 | 0.50744|
| R@2 | 0.71763 |
| R@3 | 0.80543 |
| Diversity | 10.220685 |
| Matching Score | 2.855296 |

## Setup

**Prerequisites:** Python 3.11+, PyTorch with CUDA, HuggingFace Transformers

**Pre-trained checkpoints required:**
```
./pretrained/VQVAE/net_best_fid.pth        # VQ-VAE weights
./checkpoints/t2m/Comp_v6_KLD005/opt.txt   # Evaluator config
./glove/                                    # GloVe embeddings (for evaluation)
./dataset/HumanML3D/                        # HumanML3D dataset
```

