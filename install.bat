@echo off
cd /d "%~dp0"
echo === Upgrading pip ===
venv\Scripts\python.exe -m pip install --upgrade pip
echo === Installing CPU-only PyTorch ===
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo === Installing requirements ===
venv\Scripts\python.exe -m pip install -r requirements.txt
echo === INSTALL COMPLETE ===
