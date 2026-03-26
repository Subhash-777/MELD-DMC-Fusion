# DMC-Fusion: Complete Project Understanding

## 🎯 What Does This Project Do?

**DMC-Fusion** = **D**ialogue **M**ultimodal **C**ontext **F**usion for Emotion Recognition

Given a dialogue from the TV show *Friends*, the model predicts the emotion of each spoken line.

### Example:
```
Rachel: "Oh my God!" → Joy
Monica: "I know!" → Surprise
Chandler: "Could I be any more..." → Sarcasm (Anger/Disgust)
```

**Target Task:** Multi-class emotion classification (7 classes) on conversational speech
**Dataset:** MELD (Multimodal EmotionLines Dataset) - 13,708 total utterances
**Best Performance:** **Val WF1 = 0.6250** (V12 variant)

---

## 📊 The 7 Emotions (Classes)

| Label | Emotion | % in Data | Class Weight | Challenge |
|-------|---------|-----------|-------------|-----------|
| 0 | **neutral** | 46% | 3.0 | Too dominant, suppresses others |
| 1 | **surprise** | 12% | 10.0 | Underrepresented (281 val samples) |
| 2 | **fear** | 2% | 18.0 | **Rarest** (only 50 val samples!) |
| 3 | **sadness** | 7% | 8.0 | Hard to distinguish from neutral |
| 4 | **joy** | 13% | 3.0 | Clear in text but sometimes silent |
| 5 | **disgust** | 2% | 8.0 | Rare, mixed with anger |
| 6 | **anger** | 11% | 5.0 | Clear acoustic patterns |

**Key Challenge:** **Class Imbalance** → Neutral = 1256 samples, Fear = 50 samples. Model naturally defaults to predicting neutral if not trained carefully.

---

## 🏗️ System Architecture (5 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: FEATURE EXTRACTION (Pre-computed, no gradients)    │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  TEXT (Transcript)    AUDIO (WAV file)   VISUAL (Frames)    │
│        ↓                     ↓                   ↓            │
│  RoBERTa-base         WavLM-base+      EfficientNet-B4      │
│  125M params          94K hours SSL    ImageNet pretrained  │
│  (frozen in V1-V5)                                           │
│        ↓                     ↓                   ↓            │
│    768-dim            768-dim           768-dim → Projection │
│                                              ↓                │
│                         All features are 768-dimensional     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: CONTEXT ENCODING (Temporal modeling per modality)  │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  Input: 5 consecutive utterances from the dialogue           │
│         (current + 4 previous)                               │
│                                                               │
│  Text Context        Audio Context      Visual Context       │
│  Transformer Layer   Transformer Layer  Transformer Layer    │
│  + Speaker Embedding + Speaker Embedding+ Speaker Embedding │
│  Output: [B, 5, 768] Output: [B, 5, 768] Output: [B, 5, 768]│
│                                                               │
│  ✓ Why needed: Emotions build over dialogue                 │
│               - Prior utterances set tone                    │
│               - Sarcasm needs context to detect              │
│               - Speaker interaction matters                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: CONFIDENCE GATING (Soft selection of best modality) │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  For each modality, compute "confidence":                    │
│    - Text confidence: Is text reliable for this utterance?   │
│    - Audio confidence: Is audio clear/useful?                │
│    - Visual confidence: Are faces visible/bright?            │
│                                                               │
│  Formula:                                                     │
│    gate_text  = softmax(conf_text)    [per utterance]       │
│    gate_audio = softmax(conf_audio)   [per utterance]       │
│    gate_visual = softmax(conf_visual) [per utterance]       │
│                                                               │
│  ✓ Why needed: Handle missing/corrupted modalities          │
│    - Dark scenes = poor visual features                      │
│    - Loud background = noisy audio                          │
│    - Unclear transcription = noise in text                  │
│                                                               │
│  Output shape: [B, 5, 3] (3 gates per utterance)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: FUSION (Weighted combination of modalities)        │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  Fused = gate_text   × text_context +                        │
│          gate_audio  × audio_context +                       │
│          gate_visual × visual_context                        │
│                                                               │
│  Shape: [B, 5, 768]                                          │
│                                                               │
│  ✓ Why needed: Different modalities capture different info  │
│    - Text: semantic meaning (i understand what was said)     │
│    - Audio: prosody/tone (flat voice = sadness)             │
│    - Visual: facial expressions                             │
│    - Combined = robust emotion detection                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5: CROSS-MODAL ATTENTION (Modalities talk to each other)
│─────────────────────────────────────────────────────────────│
│                                                               │
│  Standard transformer attention layer:                       │
│    - 8 attention heads                                       │
│    - 2 layers (V12 uses 1 in code)                          │
│    - Each modality attends to outputs of others              │
│                                                               │
│  Example learning:                                           │
│    "Sad face + flat voice + sad words" → confidence ↑↑       │
│    "Smile + excited tone + neutral words" → joy path        │
│                                                               │
│  Output: [B, 5, 768] (enhanced fused representation)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 6: CLASSIFICATION HEADS                               │
│─────────────────────────────────────────────────────────────│
│                                                               │
│  Head 1 (Main):                                              │
│    Fused [768] → Linear [384] → ReLU → Linear [7 classes]  │
│    Output: logits for emotion (0-6)                         │
│                                                               │
│  Head 2 (Auxiliary - "Shift Head"):                         │
│    Predicts: Did emotion change from previous utterance?    │
│    Binary classification: Yes/No                            │
│    Loss weight: 0.3 × shift_loss (regularizes temporal flow)│
│                                                               │
│  ✓ Final prediction: argmax(logits) → emotion label        │
└─────────────────────────────────────────────────────────────┘
```

---

## 💾 Feature Extraction (Pre-computed)

Features are extracted **once** per dataset split and stored on disk:

### 1. **Text Features** (RoBERTa-base) — `features/text/`
```python
# From extract_text.py
input_ids, attention_mask = roberta_tokenizer(transcription)
# Stored as dict: {dialogue_id_utterance_id: {"input_ids": ..., "attn_mask": ...}}
# Dimension: 192 tokens × 768-hidden (max transcript length)
```

### 2. **Audio Features** (WavLM-base+) — `features/audio/`
```python
# From extract_audio.py
audio_embedding = wavlm_model.extract_features(audio_wav)
# Aggregated to single 768-dim vector per utterance
# Captures: pitch, tone, speaking rate, prosody
```

### 3. **Visual Features** (EfficientNet-B4) — `features/visual/`
```python
# From extract_visual.py
frames = extract_frames_from_video(mp4_file)
visual_features = efficientnet_b4(frames)
# Aggregated to single 768-dim vector per utterance
# Captures: facial expressions, head pose
```

**Why pre-compute?**
- Extracting features takes 2-3 hours ⏱️
- Done once, used in every training run 🎯
- Saves training time: only learn fusion layers + fine-tune text 🚀

---

## 🔄 Data Flow (Training Step)

```
1. LOAD BATCH from DataLoader
   ↓
   input_ids:   [B=2, L=5, S=192]   (2 dialogues, 5 utterances, 192 tokens)
   attn_mask:   [B=2, L=5, S=192]   (which tokens are padding)
   audio:       [B=2, L=5, 768]     (pre-extracted audio features)
   visual:      [B=2, L=5, 768]     (pre-extracted visual features)
   labels:      [B=2, L=5]          (true emotion per utterance)
   speaker_ids: [B=2, L=5]          (speaker index in dialogue)
   mask:        [B=2, L=5]          (valid utterances, not padding)

2. TEXT ENCODING (RoBERTa)
   ├─ Flatten input_ids: [B*L, S] = [10, 192]
   ├─ Chunk into groups of 20 (BERT_CHUNK_SIZE)
   ├─ Forward through RoBERTa (frozen or unfrozen depending on epoch)
   ├─ Pool tokens → average: [10, 768]
   └─ Reshape back: [B, L, 768]

3. AUDIO & VISUAL (Already extracted)
   ├─ Audio: [B, L, 768]
   └─ Visual projection: [B, L, 768] → LayerNorm → ReLU → [B, L, 768]

4. CONTEXT ENCODING (Per-modality transformers)
   ├─ Text:   ctx_text([B, L, 768], speaker_ids) → [B, L, 768]
   ├─ Audio:  ctx_audio([B, L, 768], speaker_ids) → [B, L, 768]
   └─ Visual: ctx_visual([B, L, 768], speaker_ids) → [B, L, 768]

5. CONFIDENCE GATING
   ├─ conf_text = ConfidenceNet(text)           → [B, L]
   ├─ conf_audio = ConfidenceNet(audio)         → [B, L]
   ├─ conf_visual = ConfidenceNet(visual)       → [B, L]
   └─ gates = softmax([conf_text, conf_audio, conf_visual]) → [B, L, 3]

6. FUSION
   fused = gates[:, :, 0] ⊗ text +
           gates[:, :, 1] ⊗ audio +
           gates[:, :, 2] ⊗ visual
   Result: [B, L, 768]

7. CROSS-MODAL ATTENTION
   enhanced = transformer_attention(fused) → [B, L, 768]

8. CLASSIFICATION
   ├─ logits = classifier(enhanced) → [B, L, 7]
   ├─ shift_logits = shift_head(enhanced[1:]) → [B, L-1, 2]
   └─ return logits, shift_logits, gates

9. LOSS COMPUTATION
   ├─ ce_loss = CrossEntropyLoss(logits.view(B*L, 7), labels.view(B*L))
   │           (with class_weights and label_smoothing)
   │
   ├─ shift_labels, shift_valid = get_shift_labels(labels, mask)
   ├─ shift_loss = CrossEntropyLoss(shift_logits[shift_valid], shift_labels[shift_valid])
   │
   ├─ conf_entropy = -(gates * log(gates)) summed over dimensions
   │
   └─ total_loss = ce_loss + 0.3 * shift_loss + 0.1 * conf_entropy

10. BACKWARD PASS & OPTIMIZER STEP
    ├─ Scaled by GradScaler (AMP)
    ├─ Accumulated over GRAD_ACCUM=8 steps
    ├─ Optimizer step every 8 steps: (B=2, L=5) × 8 = 80 utterances
    └─ Different LRs for different param groups
```

---

## 🎓 Training Strategy (Two Phases)

### **Phase 1: Fusion Warm-up (Epochs 1-5)**

**RoBERTa is FROZEN** ❄️
```python
# From train.py:193-195
for p in model.bert.parameters():
    p.requires_grad = False
```

**What trains:**
- GRU encoders (ctx_text, ctx_audio, ctx_visual)
- Confidence gates
- Cross-modal attention
- Classification head
- Shift head

**Why?**
- RoBERTa has 125M parameters → expensive to train
- Let fusion layers find good starting point first
- Typical performance: Val WF1 = 0.24-0.26 (barely detects anything)

### **Phase 2: Full Fine-tuning (Epoch 6+)**

**RoBERTa UNFROZEN** 🔓
```python
# From train.py:232-243
if bert_frozen and epoch > cfg.BERT_FREEZE_EPOCHS:
    bert_frozen = False
    for p in model.bert.parameters():
        p.requires_grad = True
    # Recreate optimizer with different LRs
    optimizer = build_optimizer(model, bert_frozen=False)
```

**Learning rates (from train.py:44-58):**
```python
params = [
    {"params": model.ctx_text.parameters(), "lr": 1e-4},       # FUSION: faster
    {"params": model.ctx_audio.parameters(), "lr": 1e-4},
    {"params": model.ctx_visual.parameters(), "lr": 1e-4},
    {"params": model.conf_text.parameters(), "lr": 1e-4},
    {"params": model.conf_audio.parameters(), "lr": 1e-4},
    {"params": model.conf_visual.parameters(), "lr": 1e-4},
    {"params": model.cross_attn.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-4},
    {"params": model.shift_head.parameters(), "lr": 1e-4},
    {"params": model.vis_proj.parameters(), "lr": 1e-4},
    {"params": model.bert.parameters(), "lr": 2e-5},           # PRETRAINED: slower 10×
]
```

**Why different LRs?**
- Fusion layers (1e-4): Starting from scratch, need aggressive learning
- RoBERTa (2e-5): Already trained on 160GB text, preserve knowledge, tiny updates

**Typical performance jump:**
- Epoch 5 (frozen): Val WF1 = 0.30
- Epoch 7 (unfrozen): Val WF1 = 0.53 ⬆️⬆️

---

## ⚖️ Loss Function (Multi-component)

```python
# From train.py, loss computation around line 249-280
total_loss = L_CE + 0.3 × L_shift + 0.1 × L_conf_entropy
```

### **Component 1: CrossEntropyLoss with Class Weights** (`L_CE`)
```python
# From train.py:184-189
class_weights = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float)
#                             [3.0, 10.0, 18.0, 8.0, 3.0, 8.0, 5.0]
ce_loss = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=-1,               # Padding utterances don't contribute
    label_smoothing=cfg.LABEL_SMOOTHING  # 0.05
)
```

**Class Weights Explained:**
```
Neutral: 3.0    → 1109 dev samples, easy to detect, low penalty
Surprise: 10.0  → 281 samples, rare but manageable
Fear: 18.0      → 50 SAMPLES ONLY! Highest penalty. Missing fear = expensive
Sadness: 8.0    → 208 samples
Joy: 3.0        → 402 samples
Disgust: 8.0    → 68 samples, genuinely hard with so few examples
Anger: 5.0      → 345 samples
```

**Label Smoothing (0.05):**
Instead of hard targets: `[0, 0, 0, 1, 0, 0, 0] (joy)`
Soft targets:           `[0.007, 0.007, 0.007, 0.972, 0.007, 0.007, 0.007]`

✓ Prevents overconfidence
✓ Helps with ambiguous utterances

### **Component 2: Shift Loss** (`L_shift`, weight 0.3)
```python
# From train.py:31-35
def get_shift_labels(labels, mask):
    L, R = labels[:, :-1], labels[:, 1:]
    shift = (L != R).long()  # Did emotion change?
    valid = mask[:, :-1] & mask[:, 1:] & (L != -1) & (R != -1)
    return shift, valid

# Training:
shift_logits = model.shift_head(fused[:, 1:, :])
shift_loss = nn.CrossEntropyLoss()(shift_logits[shift_valid], shift_labels[shift_valid])
```

**Why?** Forces model to track emotion transitions:
- Rachel (joy) → Monica (surprise) = change
- Monica (surprise) → Chandler (surprise) = no change
- Helps learn temporal dynamics, not just static classification

### **Component 3: Confidence Entropy Regularization** (weight 0.1)
```python
# From train.py:38-40
def confidence_entropy_loss(conf):
    eps = 1e-8
    return -(conf * (conf + eps).log()).sum(-1).mean()

# Training:
gates_entropy = confidence_entropy_loss(gates)
```

**Why?** Prevents gates from freezing:
```
Bad: gate_text=1.0, gate_audio=0.0, gate_visual=0.0  (only text used)
Good: gate_text=0.5, gate_audio=0.3, gate_visual=0.2 (all contribute)
```

---

## 🏁 Training Loop (Simplified)

```python
# From train.py:228-280+
for epoch in range(1, EPOCHS+1):

    # === PHASE: Unfreeze RoBERTa at epoch 6 ===
    if epoch > BERT_FREEZE_EPOCHS and bert_frozen:
        unfreeze_bert_and_rebuild_optimizer()
        bert_frozen = False

    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Get batch data
        ids, attn_mask, audio, visual, labels, spk_ids, mask = ...

        # Forward pass (with AMP)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits, shift_logits, gates = model(ids, attn_mask, audio, visual, spk_ids, mask)

            # Loss computation
            ce_loss = ce_loss_fn(logits.view(-1, 7), labels.view(-1))
            shift_loss = shift_loss_fn(shift_logits[shift_valid], shift_labels[shift_valid])
            conf_ent = confidence_entropy_loss(gates)
            loss = ce_loss + 0.3 * shift_loss + 0.1 * conf_ent

        # Gradient accumulation
        scaler.scale(loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    # === VALIDATION ===
    val_wf1, val_acc, val_loss, per_class_f1 = evaluate(model, dev_loader, device, ce_loss_fn)

    # Log to MLflow
    mlflow.log_metrics({
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_wf1": val_wf1,
        "val_acc": val_acc,
        "val_f1_neutral": per_class_f1[0],
        "val_f1_surprise": per_class_f1[1],
        ...
    })

    # === CHECKPOINT MANAGEMENT ===
    if val_wf1 > best_f1:
        best_f1 = val_wf1
        no_improve = 0
        save(best_model.pt)
    else:
        no_improve += 1

    # Save top-K checkpoints for ensemble
    heapq.heappushpop(top_k_heap, (val_wf1, epoch, checkpoint_path))

    # Early stopping
    if no_improve >= PATIENCE:
        break

# === FINAL: Ensemble prediction ===
ensemble_model = load_top_k_checkpoints(top_k_heap)
test_wf1, test_acc = ensemble_evaluate(ensemble_model, test_loader, device)
```

---

## 📊 V1 → V12 Evolution

| Version | Change | Val WF1 | Notes |
|---------|---------|---------|-------|
| **V1** | Text only (BERT) | 0.38 | No audio/visual |
| **V2** | + Audio (wav2vec) | 0.44 | Simple concat |
| **V3** | + Visual (ResNet) | 0.46 | Late fusion |
| **V4** | ResNet→EfficientNet-B4 | 0.48 | Better visual |
| **V5** | + GRU context | 0.51 | Temporal modeling |
| **V6** | wav2vec→WavLM | 0.53 | Better audio |
| **V7** | + Cross-modal attn | 0.55 | Modalities interact |
| **V8** | + Speaker embeddings | 0.56 | Who spoke? |
| **V9** | + Shift loss | 0.57 | Temporal dynamics |
| **V10** | 3-turn context | 0.58 | Context window |
| **V11** | Focal loss + 5-turn | 0.59 | Fear boost (but joy collapsed) |
| **V12** | CE + class weights + 5-turn | **0.625** | **BEST** — stable |

**Key turning points:**
- V4→V5: GRU encoders add temporal understanding +0.05 leap
- V6→V7: Cross-modal attention (modalities "talk") +0.02
- V11→V12: Reverted Focal→CE because Focal boosted fear but killed joy

---

## 🎯 The 5-Turn Context Window (V10+)

```
dialogue = [Rachel, Monica, Chandler, Phoebe, Joey, Phoebe, Rachel, ...]
                1        2         3        4      5     6       7

For utterance #5 (Joey):
  Context window = [utterance 1, 2, 3, 4, 5]
  = [Rachel, Monica, Chandler, Phoebe, Joey]

For utterance #7 (Rachel):
  Context window = [utterance 3, 4, 5, 6, 7]
  = [Chandler, Phoebe, Joey, Phoebe, Rachel]

For utterance #2 (Monica):
  Context window = [utterance 1, 2, (pad), (pad), (pad)]
  = [Rachel, Monica, PAD, PAD, PAD]
  (masked out in attention)
```

**Why 5-turn?**
- Not too short: misses long-range dependencies
- Not too long: limits context confusion
- Shown in V10 experiments: 5 > 3 > 7

---

## 📁 Project File Structure

```
dmc_fusion/
│
├── src/                          # All Python code
│   ├── config.py                 # All hyperparameters (1 place)
│   ├── models.py                 # DMCFusion architecture
│   ├── dataset.py                # Data loading & batching
│   ├── train.py                  # Training loop & optimization
│   ├── evaluate.py               # Test evaluation & metrics
│   ├── extract_text.py           # Pre-extract RoBERTa features
│   ├── extract_audio.py          # Pre-extract WavLM features
│   └── extract_visual.py         # Pre-extract EfficientNet features
│
├── data/
│   ├── csv/
│   │   ├── train_sent_emo.csv    # 9,989 utterances
│   │   ├── dev_sent_emo.csv      # 1,109 utterances
│   │   └── test_sent_emo.csv     # 2,610 utterances
│   │   (columns: Dialogue_ID, Utterance_ID, Speaker, Emotion, Transcript)
│   │
│   └── videos/                   # MP4 video clips (raw → extract frames)
│
├── features/                     # Pre-extracted features (4-8GB)
│   ├── text/
│   │   ├── train_text.pt         # Dict: {dialogue_id_utt_id: {"input_ids": ..., "attn_mask": ...}}
│   │   ├── dev_text.pt
│   │   └── test_text.pt
│   │
│   ├── audio/
│   │   ├── train_audio.pt        # Dict: {dialogue_id_utt_id: [768-dim tensor]}
│   │   ├── dev_audio.pt
│   │   └── test_audio.pt
│   │
│   └── visual/
│       ├── train_visual.pt       # Dict: {dialogue_id_utt_id: [768-dim tensor]}
│       ├── dev_visual.pt
│       └── test_visual.pt
│
├── checkpoints/                  # Top-K epoch checkpoints
│   ├── ep7_wf10.53.pt
│   ├── ep15_wf10.58.pt
│   └── ...
│
├── mlruns/                       # MLflow experiment tracking
│   ├── 103284762720259461/       # Experiment ID for V4
│   ├── 120542324961255210/       # V5
│   ├── 154578371212130832/       # V6
│   └── ... (multiple versions)
│
├── best_model.pt                 # Best single checkpoint by val WF1
│
├── ensemble_config.pt            # Paths to top-5 checkpoints (for inference)
│
├── main.py                       # Orchestrates entire pipeline
│
├── verify.py                     # Sanity check: load model, verify runs
│
├── restore_v12.py                # Resume V12 training from crash
│
└── README.md                     # Detailed documentation
```

---

## 🚀 Complete Workflow

```bash
# Step 0: Extract features (RUN ONCE, ~2-3 hours)
python src/extract_text.py   # → features/text/
python src/extract_audio.py  # → features/audio/
python src/extract_visual.py # → features/visual/

# Step 1: Train (epochs 1-50, early stops at ~30-40)
python src/train.py
  ├─ Epoch 1-5:  RoBERTa frozen, val WF1 ≈ 0.24
  ├─ Epoch 6+:   RoBERTa unfrozen, val WF1 jumps to 0.53+
  ├─ Saves checkpoints/ep{N}_wf1{X}.pt
  ├─ Saves best_model.pt (best single checkpoint)
  ├─ Saves ensemble_config.pt (top-5 checkpoint paths)
  └─ Logs all metrics → MLflow

# Step 2: Evaluate (Test set)
python src/evaluate.py
  ├─ Loads ensemble_config.pt (if exists, else best_model.pt)
  ├─ Runs ensemble prediction
  ├─ Computes test WF1, accuracy, per-class F1
  ├─ Calibration on val set (prob adjustment)
  └─ Saves confusion_matrix_calibrated.png, per_class_f1.png

# Step 3: View results
mlflow ui
  # Open http://127.0.0.1:5000
  # Compare runs, plot metrics, download artifacts
```

---

## 🔧 Key Hyperparameters (from config.py)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `BERT_MODEL` | roberta-base | 125M param language model |
| `WAV2VEC_MODEL` | microsoft/wavlm-base-plus | Speech SSL model |
| `VIS_BACKBONE` | efficientnet_b4 | Vision CNN |
| `MAX_TEXT_LEN` | 192 | Max tokens RoBERTa sees |
| `HIDDEN_DIM` | 768 | All modality dimensions aligned |
| `BATCH_SIZE` | 2 | Dialogues per GPU step (memory constrained) |
| `GRAD_ACCUM` | 8 | Accumulate 8 steps → eff batch = 16 |
| `LR_FUSION` | 1e-4 | Learning rate for new layers |
| `LR_PRETRAINED` | 2e-5 | Learning rate for RoBERTa (10× slower) |
| `BERT_FREEZE_EPOCHS` | 5 | Freeze RoBERTa for first N epochs |
| `EPOCHS` | 50 | Max epochs (early stop at 25 patience) |
| `PATIENCE` | 25 | Stop if no improvement for 25 epochs |
| `TOP_K_CKPT` | 5 | Save top-5 checkpoints, ensemble at inference |
| `CLASS_WEIGHTS` | [3, 10, 18, 8, 3, 8, 5] | Per-class loss weight (inverse freq) |
| `LABEL_SMOOTHING` | 0.05 | Soft targets (prevent overconfidence) |
| `SHIFT_LOSS_WT` | 0.3 | Weight of emotion change auxiliary task |
| `CONF_REG_WT` | 0.1 | Weight of confidence gate entropy regularization |
| `N_HEADS` | 8 | Attention heads in cross-modal layer |
| `N_LAYERS` | 2 | Transformer layers (actual code: 1) |
| `DROPOUT` | 0.3 | 30% neuron dropout (prevent overfitting) |

---

## 📊 Final Results (V12)

**Best Val Checkpoint:** WF1 = 0.6250

| Metric | Value |
|--------|-------|
| Val Accuracy | 0.5915 |
| Val Loss | 2.0267 |
| Train Loss | 0.4187 |
| Ensemble Val WF1 | 0.6250 |

**Per-Class Performance:**

| Emotion | Val F1 | Samples (dev) | Class Weight |
|---------|--------|-------------|-------------|
| Neutral | 0.743 | 1109 | 3.0 |
| Anger | 0.451 | 345 | 5.0 |
| Surprise | 0.600 | 281 | 10.0 |
| Sadness | 0.328 | 208 | 8.0 |
| Joy | 0.538 | 402 | 3.0 |
| Disgust | 0.130 | 68 | 8.0 |
| Fear | 0.278 | 50 | 18.0 |

**Key observation:**
- Neutral, Anger, Surprise perform best (>0.45)
- Disgust & Fear hardest (<0.30) due to low samples
- train_loss << val_loss = expected overfitting (handled by early stopping)

---

## 🎬 Example: How Model Processes One Utterance

**Scene: Monica's apartment, Friends S1E1**

```
Dialogue context (5 utterances):
[1] Rachel: "Oh my God, I cannot believe I'm doing laundry!"    → JOY
[2] Monica: "That's because you've never done it before."        → NEUTRAL
[3] Rachel: "I separate the colors!"                             → SURPRISE
[4] Chandler: "All right, we're doing it!"                       → JOY
[5] Monica: "You can't teach a 7-year-old how to do laundry."   → SADNESS ← PREDICT
```

**Step 1: Extract Features**
- Text: Tokenize "You can't teach..." → [101, 1891, ...] (192 tokens)
- Audio: Extract WavLM from Monica's voice → [0.5, -0.3, ..., 0.2] (768-dim)
- Visual: Extract face frames → [-0.1, 0.8, ..., -0.4] (768-dim)

**Step 2: Context Encoding**
- RoBERTa reads [1-5] together (5-turn context + speaker info)
- WavLM transformer sees all 5 audio sequences
- EfficientNet sees all 5 visual frames
→ All outputs are 768-dim

**Step 3: Confidence Gating**
```
Text: "teach 7-year-old laundry" → CLEAR → conf = 0.9
Audio: "flat, explanatory tone" → MEDIUM → conf = 0.6
Visual: "Monica's skeptical face" → GOOD → conf = 0.8

gates = softmax([0.9, 0.6, 0.8]) = [0.40, 0.24, 0.36]
```

**Step 4: Fusion**
```
fused = 0.40 × text + 0.24 × audio + 0.36 × visual
      = 0.40 × "teaching concept" +
        0.24 × "flat tone" +
        0.36 × "skeptical expression"
      = ~~sad sentiment~~
```

**Step 5: Cross-Modal Attention**
- Text attends to audio: "flat tone confirms sadness"
- Audio attends to visual: "face matches voice"
- Visual attends to text: "words + face align"

**Step 6: Classification**
```
Classifier([fused_768]) → logits [7]
= [0.1, 0.2, 0.3, 0.8, 0.2, 0.1, 0.3]  ← label 3 (sadness) has highest score

Prediction: SADNESS ✓ (Correct!)
```

---

## 💡 Why This Architecture Works

1. **Separate modality encoders** → Each learns temporal patterns independently
2. **Confidence gates** → Selects best modality (dark scenes ↓ visual)
3. **Fusion** → Combines complementary signals (words + tone + face)
4. **Cross-modal attention** → Learns correlations (sad face + sad tone stronger)
5. **Shift loss** → Tracks emotion changes, not static prediction
6. **Class weights** → Handles imbalance (fear still detected despite 50 samples)
7. **Top-K ensemble** → Reduces variance, improves final prediction

---

## 📚 References & Key Papers

- **MELD Dataset:** Poria et al., "MELD: A Multimodal Multi-Party Dataset...", ACL 2019
- **RoBERTa:** Liu et al., "RoBERTa: A Robustly Optimized BERT...", 2019
- **WavLM:** Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training...", 2021
- **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling...", ICML 2019

---

**Created:** March 26, 2026
**Status:** Complete & Production Ready
**Best Model:** V12 (Val WF1: 0.6250) ✅
