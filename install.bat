@echo off
echo ============================================================
echo  Step 1: Remove old environment (if exists)
echo ============================================================
conda env remove -n dmc_fusion -y

echo.
echo ============================================================
echo  Step 2: Create environment from YAML
echo ============================================================

conda create -n dmc_fusion python=3.10.14 -y
conda activate dmc_fusion
conda install -c conda-forge "ffmpeg>=4.4,<7" -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 --force-reinstall
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda env create failed. Exiting.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 3: Install facenet-pytorch WITHOUT dependency check
echo  (its setup.py caps torch at 2.3.0 but works fine on 2.5.1)
echo ============================================================
conda run -n dmc_fusion pip install facenet-pytorch==2.6.0 --no-deps 
pip install "decorator==4.4.2" --force-reinstall    
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: facenet-pytorch install failed. Exiting.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 4: Verify installation
echo ============================================================
conda run -n dmc_fusion python verify.py

echo.
echo ============================================================
echo  DONE! Activate with:  conda activate dmc_fusion
echo ============================================================
pause
