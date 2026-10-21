# Courtroom-Inspired Reference-Free Financial Misinformation Detection

Code for **"Coherence at the Financial Misinformation Detection Challenge Task: A Courtroom-Inspired Framework for Reference-Free Verification"** (Pratyush Jha), the Team Coherence submission to the FMD 2026 shared task at the ICWSM 2026 MisD workshop. The system placed **3rd** on the final leaderboard.

## Method

For every financial claim, an LLM (DeepSeek-chat) writes two arguments:

- a **prosecution** argument explaining why the claim is likely false, and
- a **defense** argument that responds to the prosecution and explains why the claim could be true.

The claim and both arguments are concatenated into one input and classified by a multi-channel network:

1. **Encoder.** Frozen ModernBERT-base produces contextual token embeddings.
2. **Three parallel channels.** Each channel applies a Conv1D layer (kernel size 1, 2 or 3; 128 filters; ReLU), max pooling (kernel 2), dropout (0.2) and a bidirectional GRU (64 units per direction).
3. **Attention pooling.** Each channel's BiGRU outputs are pooled into a single vector with attention that uses a learned global context vector (Yang et al., 2016).
4. **Classification head.** The three pooled vectors are concatenated and passed through a 64-unit ReLU layer and a linear layer that outputs one logit. Label 1 means misinformation.

Training has two steps:

- **Step 1:** binary cross-entropy (BCE) at learning rate 1e-4.
- **Step 2:** fine-tuning from the step-1 weights with label-smoothed BCE (smoothing 0.1, so targets become 0.05 and 0.95) at learning rate 5e-6.

In both steps, the checkpoint with the best validation ROC-AUC is kept.

## Results

Official FMD 2026 leaderboard scores of the selected submission:

| Test split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Private (final ranking) | 0.955 | 0.9566 | 0.9547 | 0.9549 |
| Public | 0.9525 | 0.9542 | 0.9531 | 0.9524 |

The ablations (backbone, generated arguments, attention type, two-step training) are reported on the validation split in Tables 1 and 2 of the paper. Each row has a config here (see [Configs](#configs)).

## Setup

```bash
git clone https://github.com/PratyushJha254/courtroom-fmd.git
cd courtroom-fmd
pip install -r requirements.txt
```

It requires Python 3.10+, PyTorch 2.x and transformers ≥ 4.48 (the first version with ModernBERT). The paper's models were trained and evaluated on a single NVIDIA T4 GPU.

### Data

The task data comes from RFC-BENCH (Jiang et al., 2026; [github.com/1zw108/FMD](https://github.com/1zw108/FMD)). Every file is a JSON list of items with an `"Open-ended Verifiable Question"` field (the instruction followed by the claim) and, for training data, a `"Ground-True Answer"` field (`true` / `false`).

The training and test claims have to be augmented with generated arguments (step 1 below) before training. The argument files and trained checkpoints used in the paper are not public, but are available from the author on request. Whether generated or obtained, the files go here:

| File | Content |
|---|---|
| `data/misinfo_SFT_train_for_cot_aug.json`, `data/misinfo_RL_train_for_cot_aug.json` | training claims with one round of arguments (2 × 1000 items) |
| `data/test_aug_0.5.json`, `data/test_aug_0.50001.json`, `data/test_aug_0.500001.json` | the 1652 test claims, each file with an independently sampled set of arguments |
| `data/misinfo_*_aug2.json` | training claims with two rounds of arguments (only for the two-round experiment) |
| `checkpoints/debate_gca.pt` | step-1 model |
| `checkpoints/debate_gca_softbce.pt` | step-2 model (the submitted model) |

## Usage

All commands run from the repository root. Every training run is described by a YAML file in `configs/`.

### 1. Generate arguments

Set an API key for DeepSeek or any OpenAI-compatible endpoint (`--base-url`, `--model`):

```bash
export DEEPSEEK_API_KEY=...
python generate_arguments.py --input data/misinfo_SFT_train_for_cot.json --output data/misinfo_SFT_train_for_cot_aug.json
python generate_arguments.py --input data/misinfo_RL_train_for_cot.json  --output data/misinfo_RL_train_for_cot_aug.json
python generate_arguments.py --input data/test.json --output data/test_aug_0.5.json
# two more independent samples, only needed for the soft-vote ensemble
python generate_arguments.py --input data/test.json --output data/test_aug_0.50001.json --temperature 0.50001
python generate_arguments.py --input data/test.json --output data/test_aug_0.500001.json --temperature 0.500001
```

- **Settings:** temperature 0.5 and at most 350 generated tokens per argument, as in the paper.
- **Output format:** the arguments are appended to the question field as `[PROSECUTION] ...` and `[DEFENSE] ...` sections.
- **Reproducibility:** sampled arguments differ between runs, so exact reproduction of the reported numbers needs the original argument files.

### 2. Train

```bash
python train.py --config configs/modernbert_debate_gca.yaml          # step 1 -> checkpoints/debate_gca.pt
python train.py --config configs/modernbert_debate_gca_softbce.yaml  # step 2 -> checkpoints/debate_gca_softbce.pt
```

- **Step 2 starts from step 1.** It loads the step-1 checkpoint (`train.init_from`), and that checkpoint's validation AUC is the score to beat.
- **Split.** The validation set is the last 20% of the training files in file order, giving 1600 training and 400 validation samples.

### 3. Evaluate on the validation split

```bash
python evaluate.py --config configs/modernbert_debate_gca_softbce.yaml --checkpoint checkpoints/debate_gca_softbce.pt
```

The output reports accuracy, macro-F1, ROC-AUC and the confusion matrix (rows: true label, columns: predicted label). Add `--sweep` to also report the decision threshold that maximises macro-F1.

### 4. Predict on the test set

Single prediction run (the method described in the paper):

```bash
python predict.py --config configs/modernbert_debate_gca_softbce.yaml \
    --checkpoint checkpoints/debate_gca_softbce.pt \
    --test data/test_aug_0.5.json --output submissions/submission.csv
```

The selected leaderboard submission ran the same model on the three independently sampled argument sets and averaged the predicted probabilities (soft voting). On the public split it scored the same as the single run.

```bash
python predict.py --config configs/modernbert_debate_gca_softbce.yaml \
    --checkpoint checkpoints/debate_gca_softbce.pt \
    --test data/test_aug_0.5.json data/test_aug_0.50001.json data/test_aug_0.500001.json \
    --output submissions/submission_soft_vote.csv
```

Training uses inputs of up to 1000 tokens. At test time, the submitted runs truncated inputs to 512 tokens, which is set by `predict.max_length` in the config. Use `--max-length` to change it.

To check whether two submission files agree:

```bash
python compare_submissions.py submissions/submission.csv submissions/submission_soft_vote.csv
```

## Configs

| Config | Paper |
|---|---|
| `configs/roberta_claim_only.yaml` | Table 1: RoBERTa-base, claim only |
| `configs/modernbert_claim_only.yaml` | Table 1: ModernBERT-base, claim only |
| `configs/modernbert_debate_baseline_attention.yaml` | Table 2: claim + arguments, attention without a context vector |
| `configs/modernbert_debate_gca.yaml` | Table 2: claim + arguments, global context attention (step 1) |
| `configs/modernbert_debate_gca_softbce.yaml` | Table 2: + label-smoothed BCE (step 2), the submitted model |

The claim-only configs read the same argument files and drop everything after the claim (`data.use_arguments: false`). The ModernBERT claim-only row was trained on the SFT file alone, as in the original run.

## Other experiments

The following were also tried during the shared task. None of them improved on the system in the paper, which reports the best results. They are included so they can be rerun.

```bash
# three context vectors per channel (multi-head attention pooling), then label-smoothed fine-tuning
python train.py --config configs/other/multihead_gca.yaml
python train.py --config configs/other/multihead_gca_softbce.yaml

# a second debate round: the prosecution answers the defense and the defense replies once more
python generate_arguments.py --round 2 --temperature 0.7 \
    --input data/misinfo_SFT_train_for_cot_aug.json --output data/misinfo_SFT_train_for_cot_aug2.json
python generate_arguments.py --round 2 --temperature 0.7 \
    --input data/misinfo_RL_train_for_cot_aug.json --output data/misinfo_RL_train_for_cot_aug2.json
python train.py --config configs/other/two_round_debate.yaml

# unfreezing the last two ModernBERT layers after step 2
python train.py --config configs/other/gca_softbce_unfreeze.yaml

# other encoders on the claim alone
python train.py --config configs/other/finbert_claim_only.yaml
python train.py --config configs/other/deberta_claim_only.yaml

# hard (majority) voting over the three argument sets
python predict.py --config configs/modernbert_debate_gca_softbce.yaml --checkpoint checkpoints/debate_gca_softbce.pt \
    --test data/test_aug_0.5.json data/test_aug_0.50001.json data/test_aug_0.500001.json \
    --vote hard --output submissions/submission_hard_vote.csv

# regenerating arguments with deepseek-reasoner for the test items on which two runs disagree
python compare_submissions.py submissions/run_a.csv submissions/run_b.csv --write-ids disputed.txt
python generate_arguments.py --input data/test.json --output data/test_aug_reasoner.json \
    --ids disputed.txt --model deepseek-reasoner --max-tokens 0
python predict.py --config configs/modernbert_debate_gca_softbce.yaml --checkpoint checkpoints/debate_gca_softbce.pt \
    --test data/test_aug_0.5.json --replace data/test_aug_reasoner.json --output submissions/submission_reasoner.csv
```

Some ideas were tried only briefly and are not part of this code:
- Qwen2.5-7B-Instruct as the argument generator; it often hallucinated facts about the claims.
- A supervised contrastive loss with stochastic weight averaging.
- Collecting extra false financial claims from PolitiFact.

## Implementation notes

The code reproduces the submitted system as it was run. Two details matter if you build on it:

- **Padding is not masked after the encoder.** Inputs are padded to the maximum length. The encoder receives the attention mask, but the Conv1D, BiGRU and attention-pooling layers also process the padded positions.
- **The validation split is not shuffled.** It is the last 20% of the training files in file order.

## Repository layout

```
fmd/
  data.py         loading, label parsing, train/validation split, dataset
  models.py       multi-channel classifier and attention pooling variants
  losses.py       label-smoothed BCE
  engine.py       training loop with early stopping, evaluation, metrics
  generation.py   asynchronous prosecution/defense generation
  prompts.py      system prompts (also listed in the paper's appendix)
generate_arguments.py, train.py, evaluate.py, predict.py, compare_submissions.py
configs/          paper configs; configs/other/ for the experiments above
```

## Citation

```bibtex
@inproceedings{jha2026coherence,
  title     = {Coherence at the Financial Misinformation Detection Challenge Task: A Courtroom-Inspired Framework for Reference-Free Verification},
  author    = {Jha, Pratyush},
  booktitle = {Workshop on Misinformation Detection in the Era of LLMs (MisD), ICWSM},
  year      = {2026}
}
```

## License

MIT
