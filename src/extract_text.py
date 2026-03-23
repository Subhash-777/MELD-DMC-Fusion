# src/extract_text.py
"""
V11: 5-turn dialogue context (4 prior turns + target).
Tokenizer pair encoding: tokenizer(context, target) inserts real </s></s> token.
"""
import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from config import cfg


def build_ctx_pair(dial_sorted, target_idx, tokenizer):
    """
    context = up to 4 prior turns joined with ' [SEP] '
    target  = utterance to classify
    tokenizer(context, target) → <s> context </s></s> target </s>
    """
    target_row  = dial_sorted.iloc[target_idx]
    target_spk  = str(target_row.get("Speaker", "S")).strip()
    target_text = str(target_row["Utterance"]).strip()
    target_str  = f"{target_spk}: {target_text}"

    # ← CHANGE: was target_idx-2 (3-turn), now target_idx-4 (5-turn)
    start    = max(0, target_idx - 4)
    ctx_rows = dial_sorted.iloc[start:target_idx]   # up to 4 prior turns

    if len(ctx_rows) == 0:
        enc = tokenizer(
            target_str,
            max_length=cfg.MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    else:
        ctx_parts = []
        for _, row in ctx_rows.iterrows():
            spk  = str(row.get("Speaker", "S")).strip()
            text = str(row["Utterance"]).strip()
            ctx_parts.append(f"{spk}: {text}")
        ctx_str = " [SEP] ".join(ctx_parts)

        # Native pair encoding — inserts real </s></s> separator (token id=2)
        # truncation=True trims context from the LEFT, always preserving target
        enc = tokenizer(
            ctx_str,
            target_str,
            max_length=cfg.MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def extract_text_features(split="train"):
    print(f"\n[{split}] 5-turn context | max_len={cfg.MAX_TEXT_LEN} | {cfg.BERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)

    csv_map = {
        "train": "train_sent_emo.csv",
        "dev":   "dev_sent_emo.csv",
        "test":  "test_sent_emo.csv"
    }
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, csv_map[split]))

    features = {}
    for dial_id, dial_df in tqdm(df.groupby("Dialogue_ID"),
                                  desc=f"  [{split}]",
                                  total=df["Dialogue_ID"].nunique()):
        dial_sorted = dial_df.sort_values("Utterance_ID").reset_index(drop=True)
        for target_idx in range(len(dial_sorted)):
            row = dial_sorted.iloc[target_idx]
            key = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
            input_ids, attn_mask = build_ctx_pair(dial_sorted, target_idx, tokenizer)
            features[key] = {"input_ids": input_ids, "attention_mask": attn_mask}

    save_path = os.path.join(cfg.TEXT_FEAT_DIR, f"{split}_text.pt")
    torch.save(features, save_path)
    print(f"  Saved {len(features)} utterances → {save_path}")


if __name__ == "__main__":
    for s in ["train", "dev", "test"]:
        extract_text_features(s)
