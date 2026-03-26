# DMC-Fusion: Visual Flow Diagrams & Code References

## 1️⃣ Complete Data Flow (Simple)

```
Raw Dialogue (5 utterances from Friends)
    ↓
    ├─→ Text Processing (RoBERTa) → 5 × 768-dim vectors
    ├─→ Audio Processing (WavLM) → 5 × 768-dim vectors
    └─→ Visual Processing (EfficientNet) → 5 × 768-dim vectors
    ↓
[Context Encoders] - Learn temporal patterns per modality
    ↓
[Confidence Gates] - Which modality is reliable for this utterance?
    ↓
[Fusion] - Weighted sum of modalities
    ↓
[Cross-Modal Attention] - Modalities "talk" to each other
    ↓
[Classification Head] → 7-dim logits → argmax → EMOTION LABEL
```

---

## 2️⃣ Code Location Reference (Quick Nav)

| What You Want | File Location | Lines |
|---------------|---------------|-------|
| All settings | `src/config.py` | 1-66 |
| Model architecture | `src/models.py` | 1-138 |
| Training loop | `src/train.py` | 228-280+ |
| Data loading | `src/dataset.py` | 1-80 |
| Feature extraction | `src/extract_text.py` | - |
| Evaluation & metrics | `src/evaluate.py` | - |
| Loss computation | `src/train.py` | 184-190 (CE), 31-35 (shift), 38-40 (conf_entropy) |
| Optimizer setup | `src/train.py` | 43-59 (different LR groups) |
| Early stopping | `src/train.py` | ~250-280 (no_improve counter) |
| Ensemble logic | `src/train.py` | 92-126 (ensemble_evaluate) |

---

## 3️⃣ Training Timeline (What Happens Each Epoch)

```
EPOCH 1 → RoBERTa FROZEN ❄️
  Train Loss: 2.1  | Val WF1: 0.24  ← Model barely detects anything
  └─ Fusion layers learning basic feature alignment

EPOCH 2-5 → Still Frozen
  Train Loss: 1.2  | Val WF1: 0.26  ← Minimal improvement
  └─ Gradient accumulation: 8 steps per optimizer update

EPOCH 6 → RoBERTa UNFROZEN 🔓 (lr = 2e-5)
  Train Loss: 1.5  | Val WF1: 0.30  ← Begins learning
  └─ Optimizer rebuilt with different LRs for different param groups

EPOCH 7-8 → 🚀 BIG JUMP
  Train Loss: 0.8  | Val WF1: 0.53  ← Suddenly works!
  └─ Text encoder fine-tuning kicks in harder

EPOCH 9-15 → Steady Improvement
  Train Loss: 0.5  | Val WF1: 0.58-0.60
  └─ Model refining per-class discrimination

EPOCH 16+ → Plateau & Overfitting
  Train Loss: 0.4  | Val WF1: ~0.625 (plateauing)
  └─ Train loss keeps dropping but val loss increases (normal)

EPOCH 25+ → Early Stopping Triggers
  (PATIENCE = 25 epochs with no improvement)
  └─ Training stops, best checkpoint loaded
```

---

## 4️⃣ Layer-by-Layer Code Walkthrough

### Layer 1: Feature Extraction (models.py:47-59)

```python
# TEXT: RoBERTa-base (pre-trained, frozen/unfrozen)
self.bert = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
# No projection needed: roberta outputs 768-dim already ✓

# VISUAL: EfficientNet-B4 (pre-extracted, needs projection)
self.vis_proj = nn.Sequential(
    nn.Linear(cfg.VIS_FEAT_DIM, D),      # 768→768 (in practice, no-op)
    nn.LayerNorm(D),                      # Normalize features
    nn.ReLU(),
    nn.Dropout(cfg.DROPOUT)               # 30% dropout for regularization
)

# AUDIO: Already 768-dim, used directly (no projection)
```

**Code location:** `src/models.py:47-59`

---

### Layer 2: Context Encoding (models.py:61-64, 8-25)

```python
class ContextEncoder(nn.Module):
    def __init__(self, input_dim, num_speakers):
        super().__init__()
        # Speaker embeddings: which speaker is this utterance from?
        self.spk_emb = nn.Embedding(num_speakers + 1, input_dim)

        # Transformer: learns temporal patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=cfg.N_HEADS,              # 8 attention heads
            dim_feedforward=input_dim * 4,
            dropout=cfg.DROPOUT,            # 30%
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.N_LAYERS  # 2 layers
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, speaker_ids, mask):
        # Add speaker identity to features
        x = x + self.spk_emb(speaker_ids)

        # Create padding mask (True = padding, False = real token)
        pad_mask = ~mask

        # Transformer processes all 5 utterances together
        out = self.transformer(x, src_key_padding_mask=pad_mask)

        # Final layer norm
        return self.norm(out)

# Used 3 times: for text, audio, visual
self.ctx_text   = ContextEncoder(D, num_speakers)
self.ctx_audio  = ContextEncoder(D, num_speakers)
self.ctx_visual = ContextEncoder(D, num_speakers)
```

**Code location:** `src/models.py:8-25, 61-64`

---

### Layer 3: Confidence Gating (models.py:28-39, 121-125)

```python
class ConfidenceNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Simple 2-layer MLP: input → hidden/2 → 1 scalar (confidence)
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),   # 768 → 384
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),                # 30%
            nn.Linear(input_dim // 2, 1)            # 384 → 1
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B, L, 1] → [B, L]

# Input: context-encoded features [B, L, 768]
# Output: confidence score per utterance [B, L]

# Example: for utterance, compute 3 confidence scores
ct = self.conf_text(t)      # [B, L]
ca = self.conf_audio(a)     # [B, L]
cv = self.conf_visual(v)    # [B, L]

# Softmax: convert to probability distribution
gates = torch.softmax(torch.stack([ct, ca, cv], dim=-1), dim=-1)
# Shape: [B, L, 3] where gates[:, :, i] is probability of modality i
```

**Code location:** `src/models.py:28-39, 121-125`

---

### Layer 4: Fusion (models.py:126-128)

```python
# Input: context-encoded modality features (each [B, L, 768])
# Input: gates [B, L, 3] (soft probabilities per modality)
# Output: fused representation [B, L, 768]

fused = (gates[..., 0:1] * t           # gate_text × text_features
       + gates[..., 1:2] * a           # gate_audio × audio_features
       + gates[..., 2:3] * v)          # gate_visual × visual_features

# gates[..., 0:1] is [B, L, 1] to broadcast properly
# Result: weighted average of 3 modalities per utterance
```

**Code location:** `src/models.py:126-128`

---

### Layer 5: Cross-Modal Attention (models.py:71-77, 130)

```python
# Create transformer layer for cross-modal interaction
cross_layer = nn.TransformerEncoderLayer(
    d_model=D,
    nhead=cfg.N_HEADS,                   # 8 heads
    dim_feedforward=D * 4,
    dropout=cfg.DROPOUT,                 # 30%
    batch_first=True
)
self.cross_attn = nn.TransformerEncoder(
    cross_layer, num_layers=1           # 1 layer (not 2!)
)

# Usage: fused [B, L, 768] → transformer → [B, L, 768]
fused = self.cross_attn(fused)

# What happens inside:
# - Query/Key/Value all from fused (self-attention)
# - Learn which parts of each utterance attend to which other parts
# - Example: "flat audio" head attends to "sad words" token
```

**Code location:** `src/models.py:71-77, 130`

---

### Layer 6: Classification Heads (models.py:80-86, 131-136)

```python
# Main classifier: emotion prediction
self.classifier = nn.Sequential(
    nn.Linear(D, D // 2),               # 768 → 384
    nn.ReLU(),
    nn.Dropout(cfg.DROPOUT),
    nn.Linear(D // 2, cfg.NUM_CLASSES)  # 384 → 7 emotions
)

# Auxiliary head: emotion change detection
self.shift_head = nn.Linear(D, 2)      # 768 → binary (change or not)

# Usage:
logits = self.classifier(fused)         # [B, L, 7] - main prediction
shift_logits = self.shift_head(fused[:, 1:, :])  # [B, L-1, 2] - auxiliary

# Final prediction:
predictions = logits.argmax(-1)         # Pick highest class per utterance
```

**Code location:** `src/models.py:80-86, 131-136`

---

## 5️⃣ Training Loss Breakdown (train.py)

```python
# ============ Loss Component 1: Classification Loss ============
class_weights = torch.tensor([3.0, 10.0, 18.0, 8.0, 3.0, 8.0, 5.0]).to(device)
ce_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights,               # Weighted by class frequency
    ignore_index=-1,                    # Ignore padded utterances
    label_smoothing=0.05                # Soft targets
)

# Usage:
logits = model(...)                     # [B, L, 7]
labels = batch["labels"]                # [B, L]
ce_loss = ce_loss_fn(
    logits.view(B * L, 7),              # Flatten: [B*L, 7]
    labels.view(B * L)                  # Flatten: [B*L]
)

# ============ Loss Component 2: Shift Loss (Auxiliary) ============
def get_shift_labels(labels, mask):
    L, R = labels[:, :-1], labels[:, 1:]  # Compare adjacent utterances
    shift = (L != R).long()                # Binary: did emotion change?
    valid = mask[:, :-1] & mask[:, 1:] & (L != -1) & (R != -1)
    return shift, valid

shift_labels, shift_valid = get_shift_labels(labels, mask)
shift_loss = nn.CrossEntropyLoss()(
    shift_logits[shift_valid],
    shift_labels[shift_valid]
)

# ============ Loss Component 3: Entropy Regularization ============
def confidence_entropy_loss(conf):
    eps = 1e-8
    # H = -sum(p * log(p)) for each probability distribution
    return -(conf * (conf + eps).log()).sum(-1).mean()

conf_ent = confidence_entropy_loss(gates)  # gates shape [B, L, 3]

# ============ Total Loss ============
total_loss = ce_loss + 0.3 * shift_loss + 0.1 * conf_ent

# Loss weights chosen by empirical tuning:
# - 0.3: shift loss important but secondary to main classification
# - 0.1: entropy regularization just prevents gate collapse
```

**Code location:** `src/train.py:184-190, 31-35, 38-40`

---

## 6️⃣ Optimizer & Learning Rate Setup (train.py:43-59)

```python
def build_optimizer(model, bert_frozen):
    params = [
        # ===== New layers: aggressive learning =====
        {"params": model.ctx_text.parameters(),    "lr": cfg.LR_FUSION},      # 1e-4
        {"params": model.ctx_audio.parameters(),   "lr": cfg.LR_FUSION},
        {"params": model.ctx_visual.parameters(),  "lr": cfg.LR_FUSION},
        {"params": model.conf_text.parameters(),   "lr": cfg.LR_FUSION},
        {"params": model.conf_audio.parameters(),  "lr": cfg.LR_FUSION},
        {"params": model.conf_visual.parameters(), "lr": cfg.LR_FUSION},
        {"params": model.cross_attn.parameters(),  "lr": cfg.LR_FUSION},
        {"params": model.classifier.parameters(),  "lr": cfg.LR_FUSION},
        {"params": model.shift_head.parameters(),  "lr": cfg.LR_FUSION},
        {"params": model.vis_proj.parameters(),    "lr": cfg.LR_FUSION},
    ]

    if not bert_frozen:
        # ===== Pretrained layer: conservative learning =====
        params.append({"params": model.bert.parameters(),
                       "lr": cfg.LR_PRETRAINED})  # 2e-5

    return torch.optim.AdamW(params, weight_decay=1e-2)

# Learning rate ratio: 1e-4 / 2e-5 = 5× (fusion learns 5× faster)
# Why? Fusion layers random init, RoBERTa has 125M pre-trained weights
```

**Code location:** `src/train.py:43-59`

---

## 7️⃣ Training Loop (Epoch-by-epoch, train.py:228+)

```python
for epoch in range(start_epoch, cfg.EPOCHS + 1):

    # ===== PHASE TRANSITION: Unfreeze RoBERTa =====
    if bert_frozen and epoch > cfg.BERT_FREEZE_EPOCHS:
        print(f"🔓 Epoch {epoch}: Unfreezing RoBERTa...")
        bert_frozen = False
        for p in model.bert.parameters():
            p.requires_grad = True  # Allow gradients to flow to RoBERTa
        # Rebuild optimizer with RoBERTa params at lower LR
        optimizer = build_optimizer(model, bert_frozen=False)
        scheduler = get_cosine_schedule_with_warmup(...)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    # ===== TRAIN STEPS =====
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Get batch
        ids, attn_mask, audio, visual, labels, spk_ids, mask = batch_to_device(batch, device)

        # Forward pass with mixed precision (AMP)
        with torch.amp.autocast("cuda", enabled=cfg.USE_AMP, dtype=torch.float16):
            logits, shift_logits, gates = model(ids, attn_mask, audio, visual, spk_ids, mask)

            # Loss computation
            B, L, C = logits.shape
            ce_loss = ce_loss_fn(logits.view(B * L, C), labels.view(B * L))
            shift_labels, shift_valid = get_shift_labels(labels, mask)
            shift_loss = shift_loss_fn(shift_logits[shift_valid], shift_labels[shift_valid])
            conf_ent = confidence_entropy_loss(gates)
            loss = ce_loss + 0.3 * shift_loss + 0.1 * conf_ent

        # Gradient accumulation: accumulate 8 steps before update
        scaler.scale(loss / cfg.GRAD_ACCUM).backward()

        if (step + 1) % cfg.GRAD_ACCUM == 0:
            scaler.step(optimizer)      # Apply accumulated gradients
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    # ===== VALIDATION =====
    val_wf1, val_acc, val_loss, per_class_f1 = evaluate(model, dev_loader, device, ce_loss_fn)

    # ===== CHECKPOINTING & EARLY STOPPING =====
    if val_wf1 > best_f1:
        best_f1 = val_wf1
        no_improve = 0
        torch.save({"model_state_dict": model.state_dict(),
                    "val_wf1": val_wf1,
                    "epoch": epoch}, "best_model.pt")
    else:
        no_improve += 1

    # Save top-5 checkpoints for ensemble
    heapq.heappushpop(top_k_heap, (val_wf1, epoch, epoch_checkpoint_path))

    # Early stopping
    if no_improve >= cfg.PATIENCE:
        print(f"🛑 Early stopping at epoch {epoch} (no improve for {cfg.PATIENCE})")
        break

    # ===== LOGGING =====
    mlflow.log_metrics({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_wf1": val_wf1,
        "val_f1_neutral": per_class_f1[0],
        "val_f1_surprise": per_class_f1[1],
        ...
    })
```

**Code location:** `src/train.py:228-280+`

---

## 8️⃣ Ensemble Inference (train.py:92-126)

```python
def ensemble_evaluate(ckpt_paths, num_speakers, loader, device):
    """Average predictions from top-5 checkpoints"""

    # Load all models
    models_list = []
    for path in ckpt_paths:  # e.g., [ep7_wf10.53.pt, ep15_wf10.58.pt, ...]
        m = DMCFusion(num_speakers=num_speakers).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models_list.append(m)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids, attn_mask, audio, visual, labels, spk_ids, mask = batch_to_device(batch, device)

            # Get predictions from each model
            avg_probs = None
            for m in models_list:
                logits, _, _ = m(ids, attn_mask, audio, visual, spk_ids, mask)
                probs = torch.softmax(logits, dim=-1)  # [B, L, 7]
                avg_probs = probs if avg_probs is None else avg_probs + probs

            # Average probability across models
            avg_probs /= len(models_list)  # [B, L, 7]

            # Get final predictions
            preds = avg_probs.argmax(-1)
            valid = (labels != -1) & mask
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

    # Compute metrics
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    return wf1, acc

# Example output:
# Ensemble Val WF1 = 0.6250
# Ensemble accuracy = 0.5915
```

**Code location:** `src/train.py:92-126`

---

## 9️⃣ Inference Usage (Simple)

```python
# Load best single model
model = DMCFusion(num_speakers=...)
model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
model.eval()

# Or load ensemble
all_ckpts = torch.load("ensemble_config.pt")
predictions = ensemble_evaluate(all_ckpts, num_speakers, test_loader, device)

# Predict on one dialogue (manually)
with torch.no_grad():
    logits, _, gates = model(input_ids, attn_mask, audio, visual, speaker_ids, mask)

    # [B, L, 7] → get max class per utterance
    emotions = logits.argmax(-1)  # [B, L]

    # Get confidence (probability of predicted class)
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(-1).values  # [B, L]

    # Which modality was most important?
    important_modality = gates.argmax(-1)  # 0=text, 1=audio, 2=visual
```

---

## 🔟 Summary: You Need to Know

| Concept | Why Important | Where in Code |
|---------|--------------|---------------|
| **Feature Extraction** | Pre-computed, never delete features/ | extract_*.py |
| **5-Turn Context** | Emotions build over conversation | ContextEncoder in models.py |
| **Confidence Gates** | Handle missing modalities (dark, noisy) | ConfidenceNet in models.py |
| **Cross-Modal Attention** | Learn correlations between modalities | cross_attn layer in models.py |
| **Class Weights** | Handle imbalance (fear gets weight 18.0) | config.py:32, train.py:184 |
| **Label Smoothing** | Prevent overconfidence on ambiguous | train.py:188 |
| **Shift Loss** | Track emotion changes | train.py:31-35 |
| **Fusion Warm-up** | Let fusion learn before RoBERTa training | train.py:193-196, 232-243 |
| **Different LRs** | Fusion 1e-4, RoBERTa 2e-5 (5× slower) | train.py:43-59 |
| **Early Stopping** | Stop if no improvement for 25 epochs | train.py:~250 |
| **Top-K Ensemble** | Average 5 best checkpoints for robustness | train.py:92-126 |

---

**Total Parameters:**
- RoBERTa: 125M
- Context Encoders (3×): ~2M
- Confidence Nets (3×): ~500K
- Cross-Modal Attention: ~3M
- Classification Head: ~300K
- **Total: ~131M** (125M from pretrained RoBERTa)

**Training Time:** ~15-20 hours per full run (best model was V12: 15.5h)
**VRAM:** 4GB GPU (batch_size=2, grad_accum=8 required for fitting)
**Features Size:** 4-8GB (never delete, extraction takes 2-3 hours)
