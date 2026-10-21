@echo off
cd /d "%~dp0"
echo === Installing local opencv wheel ===
venv\Scripts\python.exe -m pip install C:\ocv_dl\opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl
echo === Installing requirements from cache ===
venv\Scripts\python.exe -m pip install -r requirements.txt
echo === PACKAGE LIST ===
venv\Scripts\python.exe -m pip list
echo === INSTALL2 COMPLETE ===
