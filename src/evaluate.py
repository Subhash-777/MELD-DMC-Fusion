# src/evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, f1_score,
                              accuracy_score, ConfusionMatrixDisplay)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import os

from config import cfg
from dataset import MELDDialogueDataset, collate_dialogues
from models import DMCFusion


def get_ensemble_probs(source, num_speakers, loader, device, mode="ensemble"):
    """Returns averaged softmax probs + flat label array for the full loader."""
    if mode == "single":
        models_list = [source]   # source is already a loaded model
    else:
        models_list = []
        for path in source:
            m    = DMCFusion(num_speakers=num_speakers).to(device)
            ckpt = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            models_list.append(m)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids     = batch["input_ids"].to(device)
            amask   = batch["attn_mask"].to(device)
            audio   = batch["audio"].to(device)
            visual  = batch["visual"].to(device)
            labels  = batch["labels"].to(device)
            spk_ids = batch["speaker_ids"].to(device)
            mask    = batch["mask"].to(device)

            avg_probs = None
            for m in models_list:
                logits, _, _ = m(ids, amask, audio, visual, spk_ids, mask)
                probs        = torch.softmax(logits, dim=-1)
                avg_probs    = probs if avg_probs is None else avg_probs + probs
            avg_probs /= len(models_list)

            valid = (labels != -1) & mask
            all_probs.append(avg_probs[valid].cpu().numpy())
            all_labels.extend(labels[valid].cpu().tolist())

    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def calibrate_biases(val_probs, val_labels, n_classes=7):
    """
    Coordinate-descent post-hoc calibration.
    Finds per-class log-probability biases that maximise val WF1:
        calibrated_score_c = log(p_c) + b_c  →  argmax gives prediction
    Uses a coarse grid (-2 to +3, step 0.1) so discrete WF1 jumps
    are visible to the optimizer — Nelder-Mead fails here because it
    uses tiny perturbations that don't change any predictions.
    """
    log_p    = np.log(val_probs + 1e-9)
    biases   = np.zeros(n_classes)
    best_wf1 = f1_score(val_labels, log_p.argmax(axis=1),
                        average="weighted", zero_division=0)

    for _ in range(5):          # up to 5 full passes over all classes
        improved = False
        for c in range(n_classes):
            best_b = biases[c]
            for b in np.linspace(-2.0, 3.0, 51):
                trial    = biases.copy()
                trial[c] = b
                preds    = (log_p + trial[np.newaxis, :]).argmax(axis=1)
                wf1      = f1_score(val_labels, preds,
                                    average="weighted", zero_division=0)
                if wf1 > best_wf1 + 1e-6:
                    best_wf1 = wf1k
                    best_b   = b
                    improved = True
            biases[c] = best_b
        if not improved:
            break   # converged

    return biases, best_wf1


def evaluate():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["neutral","surprise","fear","sadness","joy","disgust","anger"]
    print(f"Evaluating on: {device}\n")

    val_loader  = DataLoader(MELDDialogueDataset("dev"),  batch_size=2,
                             shuffle=False, collate_fn=collate_dialogues,
                             num_workers=0)
    test_loader = DataLoader(MELDDialogueDataset("test"), batch_size=2,
                             shuffle=False, collate_fn=collate_dialogues,
                             num_workers=0)

    # ── Determine inference mode (ensemble > single) ──────────────────────
    if os.path.exists("ensemble_config.pt"):
        ens        = torch.load("ensemble_config.pt", map_location="cpu",
                                weights_only=True)
        source     = ens["ensemble_paths"]
        n_spk      = ens["num_speakers"]
        mode       = "ensemble"
        mode_label = f"Ensemble ({len(source)} models)"
        print(f"📦 {mode_label}")
        for p in source:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            print(f"   ✓ {os.path.basename(p)}  "
                  f"(epoch={ck['epoch']}, val_wf1={ck['val_wf1']:.4f})")

    elif os.path.exists("best_model.pt"):
        ckpt       = torch.load("best_model.pt", map_location=device,
                                weights_only=True)
        n_spk      = ckpt["num_speakers"]
        model      = DMCFusion(num_speakers=n_spk).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        source     = model
        mode       = "single"
        mode_label = "Single best model"
        ep  = ckpt.get("epoch",   "?")
        wf1 = ckpt.get("val_wf1", 0.0)
        print(f"📦 Single best model  (epoch={ep}, val_wf1={wf1:.4f})")

    else:
        print("❌ No model found. Run train.py first.")
        return

    # ── Step 1: Calibrate on val set ──────────────────────────────────────
    print("\n🔧 Calibrating on val set...")
    val_probs, val_labels = get_ensemble_probs(
        source, n_spk, val_loader, device, mode)

    val_raw_preds = val_probs.argmax(axis=1)
    val_raw_wf1   = f1_score(val_labels, val_raw_preds,
                             average="weighted", zero_division=0)

    biases, val_cal_wf1 = calibrate_biases(val_probs, val_labels)
    bstr = "  ".join(
        f"{n[:3]}={b:+.2f}" for n, b in zip(class_names, biases))
    print(f"   Val WF1 (raw):        {val_raw_wf1:.4f}")
    print(f"   Val WF1 (calibrated): {val_cal_wf1:.4f}  "
          f"(Δ{val_cal_wf1-val_raw_wf1:+.4f})")
    print(f"   Biases: [{bstr}]")

    # ── Step 2: Raw test evaluation ───────────────────────────────────────
    print(f"\n─── Raw Test ({mode_label}) ───")
    test_probs, test_labels = get_ensemble_probs(
        source, n_spk, test_loader, device, mode)

    raw_preds = test_probs.argmax(axis=1)
    raw_wf1   = f1_score(test_labels, raw_preds,
                         average="weighted", zero_division=0)
    raw_mf1   = f1_score(test_labels, raw_preds,
                         average="macro",    zero_division=0)
    raw_acc   = accuracy_score(test_labels, raw_preds)

    print(classification_report(test_labels, raw_preds,
                                target_names=class_names, digits=2))
    print(f"Weighted F1 : {raw_wf1:.4f}")
    print(f"Macro F1    : {raw_mf1:.4f}")
    print(f"Accuracy    : {raw_acc:.4f}")

    # ── Step 3: Calibrated test evaluation ───────────────────────────────
    print(f"\n─── Calibrated Test ({mode_label}) ───")
    log_test  = np.log(test_probs + 1e-9)
    cal_preds = (log_test + biases[np.newaxis, :]).argmax(axis=1)
    cal_wf1   = f1_score(test_labels, cal_preds,
                         average="weighted", zero_division=0)
    cal_mf1   = f1_score(test_labels, cal_preds,
                         average="macro",    zero_division=0)
    cal_acc   = accuracy_score(test_labels, cal_preds)

    print(classification_report(test_labels, cal_preds,
                                target_names=class_names, digits=2))
    print(f"Weighted F1 : {cal_wf1:.4f}  "
          f"(Δ{cal_wf1-raw_wf1:+.4f} vs raw)")
    print(f"Macro F1    : {cal_mf1:.4f}")
    print(f"Accuracy    : {cal_acc:.4f}")
    print(f"Mode        : {mode_label} + calibration")

    # ── Confusion matrix ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, cal_preds,
        display_labels=class_names,
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(
        f"V12 {mode_label} | "
        f"WF1={cal_wf1:.4f}  MacroF1={cal_mf1:.4f}")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved → confusion_matrix.png")

    # ── MLflow logging ────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.MLFLOW_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXP)
    with mlflow.start_run(run_name="V12_test_eval_calibrated"):
        mlflow.log_metrics({
            "test_wf1_raw":  raw_wf1,
            "test_wf1_cal":  cal_wf1,
            "test_mf1_cal":  cal_mf1,
            "test_acc_cal":  cal_acc,
            "val_wf1_raw":   val_raw_wf1,
            "val_wf1_cal":   val_cal_wf1,
            "cal_gain":      cal_wf1 - raw_wf1,
        })
        mlflow.log_artifact("confusion_matrix.png")

    print(f"\n✅ Evaluation complete.")
    print(f"   Test WF1 (raw):        {raw_wf1:.4f}")
    print(f"   Test WF1 (calibrated): {cal_wf1:.4f}  ← FINAL SCORE")
    print(f"   Test Macro F1:         {cal_mf1:.4f}")
    print(f"   Test Accuracy:         {cal_acc:.4f}")
    print(f"\nMLflow UI: mlflow ui → http://127.0.0.1:5000")


if __name__ == "__main__":
    evaluate()
