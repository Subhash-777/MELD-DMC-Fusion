# src/extract_audio.py
import os, torch, torchaudio, pandas as pd
from transformers import AutoFeatureExtractor, WavLMModel
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from config import cfg

MAX_AUDIO_SAMPLES = 160_000   # 10 seconds at 16kHz — prevents OOM on long clips
CLEAR_CACHE_EVERY = 100       # free GPU memory every N items


def extract_audio_from_video(video_path, sr=16000):
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            return None
        audio_path = video_path.replace(".mp4", "_tmp_audio.wav")
        clip.audio.write_audiofile(audio_path, fps=sr, logger=None)
        clip.close()
        waveform, orig_sr = torchaudio.load(audio_path)
        os.remove(audio_path)
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze(0)

        # ── Truncate long clips to prevent OOM ───────────────────────────
        if waveform.shape[0] > MAX_AUDIO_SAMPLES:
            waveform = waveform[:MAX_AUDIO_SAMPLES]
        return waveform
    except Exception:
        return None


def run_model_safe(model, feature_extractor, waveform, device):
    """Try GPU first; fall back to CPU if OOM on this sample."""
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs["input_values"]

    # ── GPU attempt ───────────────────────────────────────────────────────
    try:
        out  = model(input_values.to(device))
        feat = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        return feat, False   # (feat, used_fallback)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # ── CPU fallback for this one sample ─────────────────────────────────
    try:
        model_cpu = model.cpu()
        out  = model_cpu(input_values)
        feat = out.last_hidden_state.mean(dim=1).squeeze(0)
        model.to(device)    # move back to GPU for next samples
        return feat, True   # (feat, used_fallback)
    except Exception:
        model.to(device)
        return None, True


def extract_audio_features(split="train"):
    # ── Skip if already extracted ─────────────────────────────────────────
    save_path = os.path.join(cfg.AUDIO_FEAT_DIR, f"{split}_audio.pt")
    if os.path.exists(save_path):
        existing = torch.load(save_path, weights_only=False)
        csv_map  = {"train": "train_sent_emo.csv",
                    "dev":   "dev_sent_emo.csv",
                    "test":  "test_sent_emo.csv"}
        df = pd.read_csv(os.path.join(cfg.DATA_DIR, csv_map[split]))
        if len(existing) == len(df):
            print(f"[{split}] Already extracted ({len(existing)} items) — skipping.")
            return
        print(f"[{split}] Partial file found ({len(existing)}/{len(df)}) — re-extracting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting WavLM-base-plus audio features [{split}] on {device}")


    # ----------------------------------------------------------------------------------------
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.WAV)
    model             = WavLMModel.from_pretrained(cfg.WAV).to(device).eval()
    # ----------------------------------------------------------------------------------------

    
    csv_map = {"train": "train_sent_emo.csv",
               "dev":   "dev_sent_emo.csv",
               "test":  "test_sent_emo.csv"}
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, csv_map[split]))

    features      = {}
    zero_count    = 0
    cpu_fallbacks = 0

    with torch.no_grad():
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                            desc=f"WavLM [{split}]")):
            key        = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            video_path = os.path.join(cfg.VIDEO_DIR, split,
                         f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")

            waveform = extract_audio_from_video(video_path)
            if waveform is None:
                features[key] = torch.zeros(cfg.HIDDEN_DIM)
                zero_count += 1
                continue

            feat, used_fallback = run_model_safe(model, feature_extractor,
                                                 waveform, device)
            if feat is None:
                features[key] = torch.zeros(cfg.HIDDEN_DIM)
                zero_count += 1
            else:
                features[key] = feat
                if used_fallback:
                    cpu_fallbacks += 1

            # Periodically free fragmented GPU memory
            if (idx + 1) % CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()

    torch.save(features, save_path)
    print(f"Saved {len(features)} audio features → {save_path}")
    print(f"  Zeros fallback : {zero_count}")
    print(f"  CPU fallbacks  : {cpu_fallbacks}")


if __name__ == "__main__":
    for s in ["train", "dev", "test"]:
        extract_audio_features(s)
