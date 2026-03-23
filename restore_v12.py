# restore_v12.py
import os, glob, shutil

MLRUNS = "mlruns"

# ── Find V12 experiment folder ────────────────────────────────────────
exp_dir = None
for folder in os.listdir(MLRUNS):
    meta = os.path.join(MLRUNS, folder, "meta.yaml")
    if os.path.exists(meta):
        with open(meta) as f:
            if "DMC-Fusion-MELD-V12" in f.read():
                exp_dir = os.path.join(MLRUNS, folder)
                break

if exp_dir is None:
    print("❌ V12 experiment not found in mlruns/")
    exit(1)

print(f"✓ Found V12 experiment: {exp_dir}")

# ── Find all best_model.pt artifacts saved during V12 ────────────────
candidates = glob.glob(
    os.path.join(exp_dir, "**", "best_model.pt"), recursive=True
)

if not candidates:
    print("❌ No best_model.pt in V12 mlruns. Artifacts may not have been saved.")
    exit(1)

# Pick the most recently modified one (highest epoch = last saved best)
candidates.sort(key=os.path.getmtime)
src = candidates[-1]
print(f"✓ Found {len(candidates)} checkpoint(s) — using most recent:")
print(f"  {src}")

# ── Clear conflicting files ───────────────────────────────────────────
for f in ["ensemble_config.pt", "swa_model.pt"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed {f}")

shutil.copy(src, "best_model.pt")
print(f"✓ Restored → best_model.pt")
print("\nReady. Run:  python src/evaluate.py")
