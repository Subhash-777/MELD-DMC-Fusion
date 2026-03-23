# src/extract_visual.py
import os, torch, cv2, pandas as pd
import timm
import torchvision.transforms as T
from torch import nn
from facenet_pytorch import MTCNN
from tqdm import tqdm
from config import cfg


class TemporalAttentionPool(nn.Module):
    """Learnable attention pooling over video frames — better than mean."""
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, x):
        # x: (N_frames, feat_dim)
        weights = torch.softmax(self.attn(x), dim=0)  # (N, 1)
        return (weights * x).sum(0)                    # (feat_dim,)


def extract_visual_features(split="train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting EfficientNet-B4 visual features [{split}] on {device}")

    # EfficientNet-B4: 1792D output
    backbone = timm.create_model(cfg.VIS_BACKBONE, pretrained=True, num_classes=0)
    backbone = backbone.to(device).eval()

    projector    = nn.Linear(cfg.VIS_FEAT_DIM, cfg.VIS_PROJ_DIM).to(device)
    attn_pool    = TemporalAttentionPool(cfg.VIS_PROJ_DIM).to(device)
    mtcnn        = MTCNN(device=device, keep_all=False, min_face_size=20)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((380, 380)),        # EfficientNet-B4 native resolution
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    face_transform = T.Compose([
        T.Resize((380, 380)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    csv_map = {"train": "train_sent_emo.csv",
               "dev":   "dev_sent_emo.csv",
               "test":  "test_sent_emo.csv"}
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, csv_map[split]))

    features = {}
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"EfficientNet [{split}]"):
            key = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            video_path = os.path.join(cfg.VIDEO_DIR, split,
                         f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_interval = max(int(fps / 2), 1)   # 2 fps (was 1 fps)
            frame_feats, frame_idx = [], 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % frame_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Try face crop first
                    try:
                        face = mtcnn(rgb)
                        if face is not None:
                            img_t = face_transform(face).unsqueeze(0).to(device)
                        else:
                            raise ValueError("no face")
                    except Exception:
                        img_t = transform(rgb).unsqueeze(0).to(device)

                    feat = backbone(img_t)                          # (1, 1792)
                    feat = projector(feat).squeeze(0)               # (768,)
                    frame_feats.append(feat.cpu())
                frame_idx += 1
            cap.release()

            if frame_feats:
                stacked = torch.stack(frame_feats).to(device)      # (N, 768)
                features[key] = attn_pool(stacked).cpu()           # (768,)
            else:
                features[key] = torch.zeros(cfg.VIS_PROJ_DIM)

    save_path = os.path.join(cfg.VIS_FEAT_DIR, f"{split}_visual.pt")
    torch.save(features, save_path)
    print(f"Saved {len(features)} visual features → {save_path}")


if __name__ == "__main__":
    for s in ["train", "dev", "test"]:
        extract_visual_features(s)
