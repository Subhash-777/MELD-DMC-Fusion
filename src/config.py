# src/config.py
import os

class Config:
    DATA_DIR       = "data/csv"
    VIDEO_DIR      = "data/videos"
    FEATURE_DIR    = "features"
    TEXT_FEAT_DIR  = "features/text"
    AUDIO_FEAT_DIR = "features/audio"
    VIS_FEAT_DIR   = "features/visual"

    BERT_MODEL     = "roberta-base"
    BERT_HIDDEN    = 768
    HIDDEN_DIM     = 768
    VIS_FEAT_DIM   = 768

    WAV2VEC_MODEL  = "microsoft/wavlm-base-plus"
    VIS_BACKBONE   = "efficientnet_b4"
    VIS_PROJ_DIM   = 768

    MAX_TEXT_LEN   = 192
    MAX_DIAL_LEN   = 15

    BERT_FREEZE_EPOCHS = 5
    BERT_CHUNK_SIZE    = 20
    USE_AMP            = True

    LABEL_SMOOTHING    = 0.05

    # V12 proven recall-tuned weights
    # neutral kept low (dominant class), fear highest (50 test samples only)
    CLASS_WEIGHTS      = [3.0, 10.0, 18.0, 8.0, 3.0, 8.0, 5.0]
    #                     neu   sur   fea   sad  joy  dis  ang

    NUM_CLASSES    = 7
    LABEL_MAP      = {
        'neutral': 0, 'surprise': 1, 'fear': 2,
        'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6
    }

    BATCH_SIZE     = 2
    EPOCHS         = 50
    PATIENCE       = 25
    GRAD_ACCUM     = 8

    LR_PRETRAINED  = 2e-5
    LR_FUSION      = 1e-4
    WARMUP_EPOCHS  = 2
    DROPOUT        = 0.3
    SHIFT_LOSS_WT  = 0.3
    CONF_REG_WT    = 0.1

    TOP_K_CKPT     = 5

    MLFLOW_EXP     = "DMC-Fusion-MELD-V12"
    MLFLOW_URI     = "mlruns"

    N_HEADS        = 8
    N_LAYERS       = 2
    SEED           = 42

cfg = Config()
os.makedirs(cfg.TEXT_FEAT_DIR,  exist_ok=True)
os.makedirs(cfg.AUDIO_FEAT_DIR, exist_ok=True)
os.makedirs(cfg.VIS_FEAT_DIR,   exist_ok=True)
