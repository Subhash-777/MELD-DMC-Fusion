# src/dataset.py
import torch
import pandas as pd
from torch.utils.data import Dataset
from config import cfg


class MELDDialogueDataset(Dataset):
    def __init__(self, split="train"):
        csv_map = {"train": "train_sent_emo.csv",
                   "dev":   "dev_sent_emo.csv",
                   "test":  "test_sent_emo.csv"}
        self.df           = pd.read_csv(f"{cfg.DATA_DIR}/{csv_map[split]}")
        self.text_feats   = torch.load(f"{cfg.TEXT_FEAT_DIR}/{split}_text.pt")
        self.audio_feats  = torch.load(f"{cfg.AUDIO_FEAT_DIR}/{split}_audio.pt")
        self.visual_feats = torch.load(f"{cfg.VIS_FEAT_DIR}/{split}_visual.pt")

        speakers          = self.df["Speaker"].unique().tolist()
        self.speaker2id   = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)

        self.dialogues = []
        for _, grp in self.df.groupby("Dialogue_ID"):
            self.dialogues.append(grp.sort_values("Utterance_ID"))


    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        grp = self.dialogues[idx]
        ids_list, mask_list = [], []
        A_list, V_list      = [], []
        labels, spk_ids     = [], []

        for _, row in grp.iterrows():
            key  = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            feat = self.text_feats.get(key, None)

            if feat is None:
                ids_list.append(torch.zeros(cfg.MAX_TEXT_LEN, dtype=torch.long))
                mask_list.append(torch.zeros(cfg.MAX_TEXT_LEN, dtype=torch.long))
            else:
                ids_list.append(feat["input_ids"])
                mask_list.append(feat["attention_mask"])

            A_list.append(self.audio_feats.get( key, torch.zeros(cfg.HIDDEN_DIM)))
            V_list.append(self.visual_feats.get(key, torch.zeros(cfg.VIS_PROJ_DIM)))
            labels.append(cfg.LABEL_MAP[row["Emotion"].lower()])
            spk_ids.append(self.speaker2id.get(row["Speaker"], 0))

        return {
            "input_ids":   torch.stack(ids_list),    # (L, 64)
            "attn_mask":   torch.stack(mask_list),   # (L, 64)
            "audio":       torch.stack(A_list),      # (L, 768)
            "visual":      torch.stack(V_list),      # (L, 768)
            "labels":      torch.tensor(labels),     # (L,)
            "speaker_ids": torch.tensor(spk_ids),    # (L,)
            "length":      len(labels)
        }


def collate_dialogues(batch):
    max_len = max(b["length"] for b in batch)
    B, S    = len(batch), cfg.MAX_TEXT_LEN

    input_ids = torch.zeros(B, max_len, S, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, S, dtype=torch.long)
    audio     = torch.zeros(B, max_len, cfg.HIDDEN_DIM)
    visual    = torch.zeros(B, max_len, cfg.VIS_PROJ_DIM)
    labels    = torch.full((B, max_len), -1, dtype=torch.long)
    spk_ids   = torch.zeros(B, max_len, dtype=torch.long)
    mask      = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["length"]
        input_ids[i, :L] = b["input_ids"]
        attn_mask[i, :L] = b["attn_mask"]
        audio[i, :L]     = b["audio"]
        visual[i, :L]    = b["visual"]
        labels[i, :L]    = b["labels"]
        spk_ids[i, :L]   = b["speaker_ids"]
        mask[i, :L]      = True

    return {"input_ids": input_ids, "attn_mask": attn_mask,
            "audio": audio, "visual": visual,
            "labels": labels, "speaker_ids": spk_ids, "mask": mask}

