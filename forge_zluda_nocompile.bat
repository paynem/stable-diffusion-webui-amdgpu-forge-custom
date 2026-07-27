@echo off
REM === Forge + ZLUDA stable launcher (no torch.compile) ===
REM Adjust these paths if your folders differ.

REM ---- Stable cache dirs so compiles (if any) persist ----
set "TORCHINDUCTOR_CACHE_DIR=E:\SDCACHE\inductor"
set "TRITON_CACHE_DIR=E:\SDCACHE\triton"

REM ---- Reduce compile/fuser churn on ZLUDA ----
set "PYTORCH_NVFUSER_DISABLE=1"
set "PYTORCH_JIT=0"

REM ---- Launch directory ----
cd /d "E:\SDTEST\stable-diffusion-webui-amdgpu-forge"

REM ---- Start Forge with your known-good flags ----
REM (No torch.compile flags are enabled here; turn off Compile/Inductor in the UI as well)
"venv\Scripts\python.exe" launch.py --theme dark --use-zluda --skip-ort --cuda-stream --opt-sdp-attention --no-download-sd-model
