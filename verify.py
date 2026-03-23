# verify_env.py  (replace previous version)
import sys
print(f"Python:          {sys.version.split()[0]}")

import torch
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"CUDA version:    {torch.version.cuda}")
print(f"GPU:             {torch.cuda.get_device_name(0)}")

import numpy as np
assert np.__version__.startswith("1."), f"FAIL: numpy must be 1.x, got {np.__version__}"
print(f"numpy:           {np.__version__}  ✅ (<2.0 — facenet-pytorch safe)")

from PIL import Image
from PIL import __version__ as pil_ver
assert pil_ver == "10.2.0", f"FAIL: Pillow must be 10.2.0, got {pil_ver}"
print(f"Pillow:          {pil_ver}  ✅ (>=10.2.0,<10.3.0 — facenet-pytorch safe)")

import torchvision, torchaudio
print(f"torchvision:     {torchvision.__version__}")
print(f"torchaudio:      {torchaudio.__version__}")

import transformers
print(f"transformers:    {transformers.__version__}")

import mlflow
print(f"mlflow:          {mlflow.__version__}")

import cv2
print(f"opencv:          {cv2.__version__}")

import librosa
print(f"librosa:         {librosa.__version__}")

import moviepy
from moviepy.editor import VideoFileClip    # tests the v1 API
import decorator
assert decorator.__version__ == "4.4.2", f"FAIL: decorator {decorator.__version__}"
print(f"moviepy:         {moviepy.__version__}")
print(f"decorator:       {decorator.__version__}  ✅")

from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN()
print(f"facenet-pytorch: MTCNN + InceptionResnetV1 loaded ✅")

import sklearn, scipy, pandas, matplotlib, seaborn
print(f"scikit-learn:    {sklearn.__version__}")
print(f"scipy:           {scipy.__version__}")
print(f"pandas:          {pandas.__version__}")

print("\n✅ All packages verified — no conflicts!")
