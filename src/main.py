# main.py
import argparse, subprocess, sys
import mlflow
from config import cfg

def run_step(script, step_name):
    print(f"\n{'='*60}\n🔷 {step_name}\n{'='*60}")
    result = subprocess.run([sys.executable, f"src/{script}"])
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name}")

parser = argparse.ArgumentParser()
parser.add_argument("--skip-extract", action="store_true",
                    help="Skip feature extraction (already done)")
args = parser.parse_args()

mlflow.set_tracking_uri(cfg.MLFLOW_URI)
mlflow.set_experiment(cfg.MLFLOW_EXP)

if not args.skip_extract:
    run_step("extract_text.py",   "Text Feature Extraction (BERT)")
    run_step("extract_audio.py",  "Audio Feature Extraction (wav2vec2)")
    run_step("extract_visual.py", "Visual Feature Extraction (ResNet50)")

run_step("train.py",    "Training DMC-Fusion")
run_step("evaluate.py", "Evaluation on Test Set")

print("\n✅ Pipeline complete. Launch MLflow UI:")
print("   conda activate dmc_fusion && mlflow ui")
print("   Open → http://127.0.0.1:5000")
