# src/models.py
import torch
import torch.nn as nn
from transformers import AutoModel
from config import cfg


class ContextEncoder(nn.Module):
    def __init__(self, input_dim, num_speakers):
        super().__init__()
        self.spk_emb = nn.Embedding(num_speakers + 1, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=cfg.N_HEADS,
            dim_feedforward=input_dim * 4,
            dropout=cfg.DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.N_LAYERS)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, speaker_ids, mask):
        x = x + self.spk_emb(speaker_ids)
        pad_mask = ~mask
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.norm(out)


class ConfidenceNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DMCFusion(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        D = cfg.HIDDEN_DIM   # 768 throughout

        # ── Text: roberta-base (768-dim, no projection needed) ───────────────
        self.bert = AutoModel.from_pretrained(
            cfg.BERT_MODEL, add_pooling_layer=False)
        # NO gradient_checkpointing — it conflicts with AMP on this PyTorch version
        # NO text_proj — roberta-base already outputs 768-dim

        # ── Visual: already 768-dim on disk ──────────────────────────────────
        self.vis_proj = nn.Sequential(
            nn.Linear(cfg.VIS_FEAT_DIM, D),   # 768 → 768 (normalisation layer)
            nn.LayerNorm(D),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )

        # ── Context encoders ─────────────────────────────────────────────────
        self.ctx_text   = ContextEncoder(D, num_speakers)
        self.ctx_audio  = ContextEncoder(D, num_speakers)
        self.ctx_visual = ContextEncoder(D, num_speakers)

        # ── Confidence gating ────────────────────────────────────────────────
        self.conf_text   = ConfidenceNet(D)
        self.conf_audio  = ConfidenceNet(D)
        self.conf_visual = ConfidenceNet(D)

        # ── Cross-modal attention (1 layer, V3 proven) ───────────────────────
        cross_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.N_HEADS,
            dim_feedforward=D * 4,
            dropout=cfg.DROPOUT, batch_first=True
        )
        self.cross_attn = nn.TransformerEncoder(cross_layer, num_layers=1)

        # ── Heads ─────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(D // 2, cfg.NUM_CLASSES)
        )
        self.shift_head = nn.Linear(D, 2)

    def _encode_text(self, input_ids, attention_mask):
        """
        Chunk-based roberta-base forward.
        input_ids: [B, L, S=128] — includes 3-turn context pairs
        Returns:   [B, L, 768]
        """
        B, L, S = input_ids.shape
        ids_flat  = input_ids.view(B * L, S)
        mask_flat = attention_mask.view(B * L, S)

        outputs = []
        for i in range(0, B * L, cfg.BERT_CHUNK_SIZE):
            chunk_ids  = ids_flat[i: i + cfg.BERT_CHUNK_SIZE]
            chunk_mask = mask_flat[i: i + cfg.BERT_CHUNK_SIZE]
            out    = self.bert(input_ids=chunk_ids,
                               attention_mask=chunk_mask).last_hidden_state
            m      = chunk_mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-8)
            outputs.append(pooled)

        text = torch.cat(outputs, dim=0)        # [B×L, 768]
        return text.view(B, L, cfg.HIDDEN_DIM)  # [B, L, 768]

    def forward(self, input_ids, attention_mask, audio, visual,
                speaker_ids, mask):
        t = self._encode_text(input_ids, attention_mask)   # [B, L, 768]
        a = audio                                           # [B, L, 768]
        v = self.vis_proj(visual)                          # [B, L, 768]

        t = self.ctx_text(t,   speaker_ids, mask)
        a = self.ctx_audio(a,  speaker_ids, mask)
        v = self.ctx_visual(v, speaker_ids, mask)

        ct = self.conf_text(t)
        ca = self.conf_audio(a)
        cv = self.conf_visual(v)

        gates = torch.softmax(torch.stack([ct, ca, cv], dim=-1), dim=-1)
        fused = (gates[..., 0:1] * t
               + gates[..., 1:2] * a
               + gates[..., 2:3] * v)

        fused  = self.cross_attn(fused)
        logits = self.classifier(fused)

        shift_logits = None
        if self.training:
            shift_logits = self.shift_head(fused[:, 1:, :])

        return logits, shift_logits, gates
