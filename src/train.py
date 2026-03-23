# src/train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import math
import heapq
import shutil
import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse

from config import cfg
from dataset import MELDDialogueDataset, collate_dialogues
from models import DMCFusion

torch.manual_seed(cfg.SEED)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_shift_labels(labels, mask):
    L, R  = labels[:, :-1], labels[:, 1:]
    shift = (L != R).long()
    valid = mask[:, :-1] & mask[:, 1:] & (L != -1) & (R != -1)
    return shift, valid


def confidence_entropy_loss(conf):
    eps = 1e-8
    return -(conf * (conf + eps).log()).sum(-1).mean()


def build_optimizer(model, bert_frozen):
    params = [
        {"params": model.ctx_text.parameters(),    "lr": cfg.LR_FUSION},
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
        params.append({"params": model.bert.parameters(),
                        "lr": cfg.LR_PRETRAINED})
    return torch.optim.AdamW(params, weight_decay=1e-2)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids     = batch["input_ids"].to(device)
            amask   = batch["attn_mask"].to(device)
            audio   = batch["audio"].to(device)
            visual  = batch["visual"].to(device)
            labels  = batch["labels"].to(device)
            spk_ids = batch["speaker_ids"].to(device)
            mask    = batch["mask"].to(device)

            logits, _, _ = model(ids, amask, audio, visual, spk_ids, mask)
            B, L, C = logits.shape
            loss    = loss_fn(logits.view(B * L, C), labels.view(B * L))
            total_loss += loss.item()

            preds = logits.argmax(-1)
            valid = (labels != -1) & mask
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

    wf1     = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    per_cls = f1_score(all_labels, all_preds, average=None,
                       labels=list(range(cfg.NUM_CLASSES)), zero_division=0)
    return wf1, acc, total_loss / len(loader), per_cls


def ensemble_evaluate(ckpt_paths, num_speakers, loader, device):
    models_list = []
    for path in ckpt_paths:
        m    = DMCFusion(num_speakers=num_speakers).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models_list.append(m)

    all_preds, all_labels = [], []
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

            preds = avg_probs.argmax(-1)
            valid = (labels != -1) & mask
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

    wf1 = f1_score(all_preds, all_labels, average="weighted", zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return wf1, acc


def train(resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | AMP: {cfg.USE_AMP} | GRAD_ACCUM: {cfg.GRAD_ACCUM}")
    print(f"V12: CE + 5-turn + recall weights | "
          f"PATIENCE={cfg.PATIENCE} | top-{cfg.TOP_K_CKPT} ensemble")
    print(f"Weights: {cfg.CLASS_WEIGHTS}\n")

    train_ds = MELDDialogueDataset("train")
    dev_ds   = MELDDialogueDataset("dev")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_dialogues, num_workers=0, pin_memory=True)
    dev_loader   = DataLoader(
        dev_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_dialogues, num_workers=0, pin_memory=True)

    model = DMCFusion(num_speakers=train_ds.num_speakers).to(device)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_f1      = 0.0
    no_improve   = 0
    top_k_heap   = []

    if resume:
        latest = find_latest_checkpoint()
        if latest:
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            top_k_heap  = rebuild_top_k_heap(cfg.TOP_K_CKPT)

            # Restore best_f1 from best_model.pt if it exists
            if os.path.exists("best_model.pt"):
                bm       = torch.load("best_model.pt", map_location="cpu",
                                      weights_only=True)
                best_f1  = bm["val_wf1"]

            # Estimate no_improve by scanning all checkpoint wf1 values
            all_wf1s = []
            import glob
            for p in sorted(glob.glob("checkpoints/ep*.pt"),
                            key=lambda x: int(os.path.basename(x).split("_")[0][2:])):
                c = torch.load(p, map_location="cpu", weights_only=True)
                all_wf1s.append((c["epoch"], c["val_wf1"]))
            # Count epochs since last improvement
            last_best_ep = max(all_wf1s, key=lambda x: x[1])[0]
            no_improve   = ckpt["epoch"] - last_best_ep

            print(f"▶ Resuming from epoch {start_epoch}  "
                  f"(best_f1={best_f1:.4f}, no_improve={no_improve}/{cfg.PATIENCE})")
        else:
            print("No checkpoint found — starting fresh.")

    class_weights = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float).to(device)
    ce_loss_fn    = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-1,
        label_smoothing=cfg.LABEL_SMOOTHING
    )
    shift_loss_fn = nn.CrossEntropyLoss()

    # Freeze RoBERTa for first N epochs — fusion layers learn first
    bert_frozen = True
    for p in model.bert.parameters():
        p.requires_grad = False
    print(f"RoBERTa frozen for first {cfg.BERT_FREEZE_EPOCHS} epochs.")

    total_steps  = (cfg.EPOCHS * len(train_loader)) // cfg.GRAD_ACCUM
    warmup_steps = (cfg.WARMUP_EPOCHS * len(train_loader)) // cfg.GRAD_ACCUM

    optimizer = build_optimizer(model, bert_frozen=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    mlflow.set_tracking_uri(cfg.MLFLOW_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXP)

    
    class_names = ["neutral","surprise","fear","sadness","joy","disgust","anger"]
    os.makedirs("checkpoints", exist_ok=True)

    with mlflow.start_run(run_name="DMCFusion_V12_CE_5turn_RecallWeights"):
        mlflow.log_params({
            "bert_model":      cfg.BERT_MODEL,
            "text_ctx":        "5-turn-pair-encoding",
            "max_text_len":    cfg.MAX_TEXT_LEN,
            "loss":            "CrossEntropyLoss",
            "class_weights":   str(cfg.CLASS_WEIGHTS),
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "patience":        cfg.PATIENCE,
            "top_k":           cfg.TOP_K_CKPT,
            "lr_pretrained":   cfg.LR_PRETRAINED,
            "lr_fusion":       cfg.LR_FUSION,
            "batch_size":      cfg.BATCH_SIZE,
            "grad_accum":      cfg.GRAD_ACCUM,
        })

        for epoch in range(start_epoch, cfg.EPOCHS + 1):
            torch.cuda.empty_cache()

            # Unfreeze RoBERTa after freeze period
            if bert_frozen and epoch > cfg.BERT_FREEZE_EPOCHS:
                bert_frozen = False
                for p in model.bert.parameters():
                    p.requires_grad = True
                remaining = ((cfg.EPOCHS - cfg.BERT_FREEZE_EPOCHS)
                             * len(train_loader)) // cfg.GRAD_ACCUM
                optimizer = build_optimizer(model, bert_frozen=False)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, warmup_steps=200, total_steps=remaining)
                scaler    = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)
                print(f"\n🔓 Epoch {epoch}: RoBERTa unfrozen | "
                      f"LR={cfg.LR_PRETRAINED}\n")

            model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")):
                ids     = batch["input_ids"].to(device)
                amask   = batch["attn_mask"].to(device)
                audio   = batch["audio"].to(device)
                visual  = batch["visual"].to(device)
                labels  = batch["labels"].to(device)
                spk_ids = batch["speaker_ids"].to(device)
                mask    = batch["mask"].to(device)

                with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                    logits, shift_logits, conf = model(
                        ids, amask, audio, visual, spk_ids, mask)
                    B, L, C = logits.shape

                    # Primary CE loss with class weights + label smoothing
                    loss = ce_loss_fn(
                        logits.view(B * L, C), labels.view(B * L))

                    # Auxiliary: emotion shift detection
                    if shift_logits is not None:
                        shift_gt, shift_valid = get_shift_labels(labels, mask)
                        if shift_valid.any():
                            loss = loss + cfg.SHIFT_LOSS_WT * shift_loss_fn(
                                shift_logits[shift_valid],
                                shift_gt[shift_valid])

                    # Auxiliary: confidence entropy regularisation
                    loss = loss + cfg.CONF_REG_WT * confidence_entropy_loss(conf)
                    loss = loss / cfg.GRAD_ACCUM

                scaler.scale(loss).backward()
                total_loss += loss.item() * cfg.GRAD_ACCUM

                if (step + 1) % cfg.GRAD_ACCUM == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            avg_loss = total_loss / len(train_loader)
            val_f1, val_acc, val_loss, per_cls = evaluate(
                model, dev_loader, device, ce_loss_fn)

            cls_line = " | ".join(
                f"{n[:3]}={v:.2f}"
                for n, v in zip(class_names, per_cls))

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_loss":   val_loss,
                "val_wf1":    val_f1,
                "val_acc":    val_acc,
            }, step=epoch)
            for i, n in enumerate(class_names):
                mlflow.log_metric(f"val_f1_{n}", per_cls[i], step=epoch)

            lm_status = "frozen" if bert_frozen else "on"
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | "
                  f"Val WF1: {val_f1:.4f} | Acc: {val_acc:.4f} | "
                  f"LM: {lm_status}")
            print(f"         [{cls_line}]")

            # Save checkpoint and maintain top-K heap
            ckpt_path = (f"checkpoints/ep{epoch:02d}"
                         f"_wf1{val_f1:.4f}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_speakers":     train_ds.num_speakers,
                "epoch":            epoch,
                "val_wf1":          float(val_f1),
            }, ckpt_path)

            heapq.heappush(top_k_heap, (val_f1, epoch, ckpt_path))
            if len(top_k_heap) > cfg.TOP_K_CKPT:
                _, _, worst_path = heapq.heappop(top_k_heap)
                if os.path.exists(worst_path):
                    os.remove(worst_path)

            if val_f1 > best_f1:
                best_f1, no_improve = val_f1, 0
                shutil.copy(ckpt_path, "best_model.pt")
                mlflow.log_artifact("best_model.pt")
                print(f"  ✅ New best WF1: {best_f1:.4f} — saved")
            else:
                no_improve += 1
                print(f"  ⏳ No improvement ({no_improve}/{cfg.PATIENCE})")
                if no_improve >= cfg.PATIENCE:
                    print(f"\n⏹ Early stopping at epoch {epoch}")
                    break

        # ── Final ensemble ─────────────────────────────────────────────────
        top_k_paths = [p for _, _, p in sorted(top_k_heap, reverse=True)]
        print(f"\n📊 Ensemble of top-{len(top_k_paths)} checkpoints:")
        for p in top_k_paths:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            print(f"   ✓ {os.path.basename(p)}  "
                  f"(epoch={ck['epoch']}, val_wf1={ck['val_wf1']:.4f})")

        ens_wf1, ens_acc = ensemble_evaluate(
            top_k_paths, train_ds.num_speakers, dev_loader, device)
        print(f"\n🏆 Ensemble Val WF1 : {ens_wf1:.4f} | Acc: {ens_acc:.4f}")
        print(f"   Best single Val WF1: {best_f1:.4f}")

        mlflow.log_metric("ensemble_val_wf1", ens_wf1)
        mlflow.log_metric("best_val_wf1",     best_f1)

        if ens_wf1 > best_f1:
            torch.save({"ensemble_paths": top_k_paths,
                        "num_speakers":   train_ds.num_speakers},
                       "ensemble_config.pt")
            print("✅ Ensemble is better — saved ensemble_config.pt")
        else:
            print("✅ Single best model wins — use best_model.pt")

        print(f"\nTraining complete. Best Val WF1: {best_f1:.4f}")
        print(f"MLflow UI: mlflow ui → http://127.0.0.1:5000")


# ── Resume helper ─────────────────────────────────────────────────────────────
def find_latest_checkpoint():
    """Returns path of the highest-epoch checkpoint on disk."""
    import glob
    ckpts = sorted(glob.glob("checkpoints/ep*.pt"))
    if not ckpts:
        return None
    # sort by epoch number embedded in filename
    ckpts.sort(key=lambda p: int(os.path.basename(p).split("_")[0][2:]))
    return ckpts[-1]


def rebuild_top_k_heap(n_top=5):
    """Scans checkpoints/ and rebuilds the heap from whatever is on disk."""
    import glob
    heap = []
    for path in glob.glob("checkpoints/ep*.pt"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        heapq.heappush(heap, (ck["val_wf1"], ck["epoch"], path))
        if len(heap) > n_top:
            heapq.heappop(heap)
    return heap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in checkpoints/")
    args = parser.parse_args()
    train(resume=args.resume)
