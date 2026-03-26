# DMC-Fusion: Quick Reference Cheat Sheet

## ⚡ One-Sentence Summary
**Multimodal emotion recognition**: Given TV dialogue, predict emotion (7 classes) using RoBERTa (text) + WavLM (audio) + EfficientNet (visual) combined via transformer fusion with confidence gating.

---

## 🎯 The Problem
- **Input:** 5-turn dialogue from Friends with transcriptions, audio, and video
- **Output:** Emotion label (0-6) for each utterance
- **Challenge:** Severe class imbalance (neutral=46%, fear=2%), only 50 test fear samples, emotions depend on context

---

## 🏆 Best Solution (V12)
```
Val WF1 = 0.6250
├─ Neutral:  0.743 (1109 samples, easy)
├─ Anger:    0.451 (345 samples, clear)
├─ Surprise: 0.600 (281 samples)
├─ Joy:      0.538 (402 samples)
├─ Sadness:  0.328 (208 samples, hard)
├─ Disgust:  0.130 (68 samples, very hard)
└─ Fear:     0.278 (50 samples, hardest)
```

---

## 🏗️ Architecture (6 Layers)

```
┌─ MODALITY ENCODERS ──────────────────────┐
│ Text: RoBERTa-base (125M params)         │
│ Audio: WavLM-base+ (SSL pre-trained)     │
│ Visual: EfficientNet-B4 (ImageNet)       │
│ All → 768-dim embeddings                 │
└──────────────────────────────────────────┘
                    ↓
┌─ CONTEXT ENCODERS (Per-Modality) ───────┐
│ Transformer encoder × 3 (text/audio/vis) │
│ Learns temporal patterns in 5-turn window│
│ Adds speaker embeddings                  │
└──────────────────────────────────────────┘
                    ↓
┌─ CONFIDENCE GATES ───────────────────────┐
│ Computes softmax([text_conf, audio_conf, │
│                   visual_conf])          │
│ → Soft probability for each modality     │
│ Idea: suppress weak modalities (dark     │
│       scenes, noisy audio)               │
└──────────────────────────────────────────┘
                    ↓
┌─ FUSION ────────────────────────────────┐
│ fused = gate_text×text + gate_audio×    │
│         audio + gate_visual×visual      │
│ Weighted sum of modalities              │
└─────────────────────────────────────────┘
                    ↓
┌─ CROSS-MODAL ATTENTION ─────────────────┐
│ Transformer layer (8 heads)              │
│ Modalities interact/learn correlations   │
│ (sad face + flat tone + sad words)       │
└─────────────────────────────────────────┘
                    ↓
┌─ CLASSIFICATION HEADS ──────────────────┐
│ Main: Linear(768→384→7) = emotion       │
│ Aux: Linear(768→2) = shift (change?)    │
│ Returns: logits[B,L,7] + shift_logits   │
└─────────────────────────────────────────┘
```

---

## 📊 Data & Metrics

| Aspect | Value |
|--------|-------|
| Dataset | MELD (Friends TV show) |
| Train/Dev/Test | 9,989 / 1,109 / 2,610 utterances |
| Features | Text + Audio + Visual |
| Feature Pre-comp | 2-3 hours (done once) |
| Feature Size | 4-8GB (store on disk) |
| Best Val WF1 | 0.6250 |
| Best Val Acc | 0.5915 |
| Total Model Params | ~131M (125M is RoBERTa) |

---

## ⚙️ Key Hyperparameters

```python
# Model
BERT_MODEL = "roberta-base"           # 125M param language model
WAV2VEC_MODEL = "microsoft/wavlm-base-plus"
VIS_BACKBONE = "efficientnet_b4"
MAX_TEXT_LEN = 192                    # Token limit per dialogue window
HIDDEN_DIM = 768                      # All modalities aligned to this

# Training
BATCH_SIZE = 2                        # Dialogues per GPU step (VRAM limited)
GRAD_ACCUM = 8                        # Effective batch = 2×8 = 16
EPOCHS = 50
PATIENCE = 25                         # Early stop if no improve for 25 epochs
LR_FUSION = 1e-4                      # New layers (fast)
LR_PRETRAINED = 2e-5                 # RoBERTa (10× slower)
BERT_FREEZE_EPOCHS = 5                # Freeze RoBERTa first 5 epochs

# Loss
CLASS_WEIGHTS = [3.0, 10.0, 18.0, 8.0, 3.0, 8.0, 5.0]  # Per-class weight
LABEL_SMOOTHING = 0.05                # Soft targets (prevent overconfidence)
SHIFT_LOSS_WT = 0.3                   # Emotion change auxiliary task
CONF_REG_WT = 0.1                     # Gate entropy regularization

# Ensemble
TOP_K_CKPT = 5                        # Save 5 best checkpoints
USE_AMP = True                        # Mixed precision training (FP16)
DROPOUT = 0.3
N_HEADS = 8                           # Attention heads
```

---

## 🔄 Training Pipeline (2 Phases)

### Phase 1: Warm-up (Epochs 1-5)
```
RoBERTa: ❌ FROZEN (no gradient updates)
Fusion layers:  ✅ TRAINING
Val WF1: ~0.24-0.26 (barely works)
Idea: Let fusion learn good geometry before expensive RoBERTa training
```

### Phase 2: Fine-tuning (Epoch 6+)
```
RoBERTa: ✅ UNFROZEN (lr = 2e-5)
Fusion layers: ✅ TRAINING (lr = 1e-4, 5× faster)
Val WF1: 0.30 → 0.53 ⬆️⬆️ (big jump at epoch 7-8)
Train until no improvement for 25 epochs (early stopping)
```

---

## 💾 Loss Function

```
Total = L_CE + 0.3 × L_shift + 0.1 × L_entropy

L_CE:     ClassEntropyLoss with class weights + label smoothing (0.05)
          → ClassWeight[fear]=18.0 (highest, only 50 samples)
          → ClassWeight[neutral]=3.0 (lowest, 1256 samples)

L_shift:  Binary "did emotion change?" prediction
          → Forces temporal dynamics tracking

L_entropy: -sum(p*log(p)) of confidence gates
          → Prevents gates from freezing (e.g., always text)
```

---

## 🎯 File Structure

```
src/config.py         ← All hyperparameters (single source of truth)
src/models.py         ← DMCFusion architecture
src/dataset.py        ← Data loading (loads pre-extracted features)
src/train.py          ← Training loop + optimizer + checkpointing
src/evaluate.py       ← Test evaluation + metrics
src/extract_*.py      ← Feature extraction (run once)

data/csv/             ← MELD CSV files (metadata, transcriptions)
features/             ← ⚠️ Pre-extracted features (1GB each, NEVER DELETE)
checkpoints/          ← Top-K epoch checkpoints
mlruns/               ← MLflow experiment logs

best_model.pt         ← Best single checkpoint (can delete + retrain)
ensemble_config.pt    ← Paths to top-5 checkpoints (can delete + retrain)
```

**🚨 Critical:** Never delete `features/` folder (extraction takes 2-3 hours)

---

## ▶️ Quick Commands

```bash
# One-time feature extraction (2-3 hours)
python src/extract_text.py
python src/extract_audio.py
python src/extract_visual.py

# Train (15-20 hours, auto early stops)
python src/train.py

# Resume after crash
python src/train.py --resume

# Evaluate on test set
python src/evaluate.py

# View MLflow dashboard
mlflow ui
# → http://127.0.0.1:5000
```

---

## 📈 Version History (V1→V12)

| V | Change | WF1 | Insight |
|---|--------|-----|---------|
| 1 | Text only | 0.38 | Need multiple modalities |
| 2 | +Audio | 0.44 | Audio captures tone |
| 3 | +Visual | 0.46 | Faces matter |
| 4 | ResNet→EfficientNet | 0.48 | Better visual backbone |
| 5 | +Context encoders | 0.51 | Temporal modeling critical |
| 6 | wav2vec→WavLM | 0.53 | Better audio pre-training |
| 7 | +Cross-attn | 0.55 | Modalities must interact |
| 8 | +Speaker emb | 0.56 | Who spoke? matters |
| 9 | +Shift loss | 0.57 | Track emotion flow |
| 10 | 3→5 turn context | 0.58 | Longer context helps |
| 11 | Focal+5turn | 0.59 | Focal boosts fear (but kills joy) |
| **12** | **CE+weights+5turn** | **0.625** | **Best balance** ✅ |

**Key insight:** V11→V12 reverted Focal→CE because Focal optimized for rare classes (fear) but broke common ones (joy). V12's class weights give explicit control per class.

---

## 🔍 What Each Modality Captures

| Modality | Captures | Example |
|----------|----------|---------|
| **Text (RoBERTa)** | Semantic meaning, negation, sarcasm | "That's NOT okay!" → anger |
| **Audio (WavLM)** | Pitch, tone, speaking rate, prosody | Flat voice → sadness |
| **Visual (EfficientNet)** | Facial expressions, head movement | Smile but sad words → context |
| **Combined** | Robust emotion via consensus | All 3 align → high confidence |

---

## ⚠️ Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Val WF1 stuck at 0.25 | Normal - RoBERTa frozen | Wait for epoch 6+ unfreeze |
| CUDA OOM | Batch too large | Already at limit (batch=2, grad_accum=8) |
| Disgust F1 = 0.13 | Only 68 samples | Inherent limitation, not fixable |
| Fear F1 = 0.28 | Only 50 samples | Class weight 18.0 forces some detection |
| Training crashed | GPU thermal throttle | Restart PC, run `python train.py --resume` |
| Features not found | Pre-extraction failed | Run extract_*.py scripts |

---

## 🎓 Key Concepts Explained

### **5-Turn Context Window**
```
Dialogue: [R: "hi", M: "hey", C: "what?", P: "....", J: "ok", P: "yes", R: "no"]
                1       2        3          4       5     6       7

For utterance 5 (Joey "ok"):     context = [1, 2, 3, 4, 5]
For utterance 7 (Rachel "no"):   context = [3, 4, 5, 6, 7]
For utterance 2 (Monica "hey"):  context = [1, 2, PAD, PAD, PAD]

Why? Emotions depend on recent history. 5 turns = sweet spot.
```

### **Confidence Gates**
```
Dark scene? → Visual confidence ↓ → Visual gate ↓
Loud background? → Audio confidence ↓ → Audio gate ↓
Unclear transcript? → Text confidence ↓ → Text gate ↓

gates = softmax([0.8, 0.3, 0.9]) = [0.39, 0.12, 0.49]
       = 39% text + 12% audio + 49% visual
```

### **Cross-Modal Attention**
```
Attention mechanism allows modalities to "understand" each other:
- Text head learns: "sad words might have sad face"
- Audio head learns: "flat tone confirms sadness"
- Visual head learns: "frown reinforces sadness"

Result: Self-reinforcing signal = high confidence
```

### **Class Weights**
```
Fear: only 50 samples → weight 18.0 (highest penalty for getting it wrong)
Neutral: 1256 samples → weight 3.0 (lowest, model naturally detects it)

Loss = CE(pred, label) × weight[label]
Predicting 'neutral' when label='fear' → loss × 18.0 (very expensive!)
```

### **Label Smoothing**
```
Hard target:    [0, 0, 1, 0, 0, 0, 0] for joy
Smooth:         [0.007, ..., 0.972 (joy), ..., 0.007]

Benefit: Prevents overconfidence. Ambiguous utterances like "...that sarcastic..."
shouldn't force model to 100% pick one class.
```

---

## 📊 Sample Predictions

```
[Scene: Monica's apartment]
Rachel: "Oh my God!"        → logits=[0.1, 0.9, 0.2, 0.3, 0.7, 0.1, 0.2] → SURPRISE ✓
Monica: "Welcome home"      → logits=[0.8, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2] → NEUTRAL ✓
Chandler: "...whatever..."  → logits=[0.2, 0.1, 0.3, 0.2, 0.2, 0.3, 0.7] → ANGER ✓

Per-modality contribution (confidence gates):
Utterance 1: [0.4 text, 0.3 audio, 0.3 visual]  (all contribute)
Utterance 2: [0.6 text, 0.1 audio, 0.3 visual]  (dark scene, suppress visual)
Utterance 3: [0.2 text, 0.7 audio, 0.1 visual]  (clear tone, boost audio)
```

---

## 🎯 Success Metrics

✅ **Achieved:**
- Multimodal fusion working (all 3 modalities contribute)
- Handles class imbalance reasonably (rare classes still detected)
- Ensemble robustness (top-5 checkpoints average better than single)
- Context matters (5-turn > 3-turn)
- Confidence gates learn sensible patterns

⚠️ **Limitations:**
- Disgust & Fear still difficult (<0.30 F1) due to data scarcity
- Train loss << val loss (overfitting) but handled by early stopping
- 4GB GPU only allows batch_size=2 (small batch = noisier gradients)

---

## 🚀 Next Steps (If Improving)

1. **More data for rare classes** → Get disgust/fear examples
2. **Data augmentation** → Augment rare class samples
3. **Focal Loss tuning** → Revisit focal loss with different γ
4. **Larger model** → Use cuda with larger model if VRAM available
5. **Context length tuning** → Try 7 or 9 turn windows
6. **Confidence gate refinement** → Learn per-modality normalization
7. **Speaker-specific modeling** → Different models per speaker personality
8. **Post-hoc calibration** → Platt scaling or temperature scaling on val set

---

## 📚 Papers Referenced

- MELD: Poria et al., ACL 2019
- RoBERTa: Liu et al., 2019
- WavLM: Chen et al., 2021
- EfficientNet: Tan & Le, ICML 2019
- Transformers: Vaswani et al., NeurIPS 2017

---

**TL;DR:**
- 3 pretrained encoders (RoBERTa + WavLM + EfficientNet) → 768-dim
- Context encoders learn temporal patterns per modality
- Confidence gates suppress weak modalities
- Fusion + cross-modal attention = multimodal understanding
- Train in 2 phases: freeze RoBERTa (5 epochs) then unfreeze
- Top-5 ensemble predictions for robustness
- **Result: 0.6250 val WF1** ✅
