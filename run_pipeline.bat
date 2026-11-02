:<<"::END_OF_WINDOWS_PART"
@echo off
REM =====================================================================================================
REM  SN segmentation pipeline launcher - ONE file for Windows and Linux.
REM    Windows: double-click run_pipeline.bat, or in Command Prompt:   run_pipeline.bat
REM    Linux:   in a terminal:                                         bash run_pipeline.bat
REM  Edit the paths in the part for YOUR system (Windows below, Linux further down).
REM  Extra options are passed on, for example:  run_pipeline.bat --stages audit subject_table
REM =====================================================================================================

REM ---------------------------- WINDOWS SETTINGS (edit these) ----------------------------
REM Folder that holds Healthy_controls, Schiz and testing (or a folder above it)
set "DATASET_DIR=C:\Users\Meghana G\Desktop\engineering\cheal\PES_university_data"
REM Folder with CIT168_T1w_head_700um.nii, prob_atlas_bilateral.nii, prob_atlas.nii, labels.txt
set "ATLAS_DIR=C:\Users\Meghana G\Desktop\engineering\cheal\atlas"
REM Output folder. Leave empty to create <dataset folder name>_sn_pipeline_output next to the dataset folder
set "OUTPUT_DIR="
REM nnU-Net results folder that contains Dataset721_chealdev49r1. Leave empty to train (GPU needed)
set "MODEL_DIR="
REM auto, cpu or cuda
set "DEVICE=auto"
REM -----------------------------------------------------------------------------------------

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment in .venv (first run only^) ...
    where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
    if not exist ".venv\Scripts\python.exe" (
        echo Could not create the virtual environment. Install Python 3.10-3.13 from python.org and tick "Add python to PATH".
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. See the messages above.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip freeze > requirements-lock.txt
)
".venv\Scripts\python.exe" -m sn_pipeline --dataset "%DATASET_DIR%" --atlas "%ATLAS_DIR%" --output "%OUTPUT_DIR%" --model-dir "%MODEL_DIR%" --device %DEVICE% %*
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (echo Pipeline finished. Masks are in the sn_masks folder of the output folder.) else (echo Pipeline stopped with an error. See the log in the logs folder of the output folder.)
pause
exit /b %RC%
::END_OF_WINDOWS_PART

# =====================================================================================================
#  LINUX PART (run with:  bash run_pipeline.bat  [extra options])
# =====================================================================================================

# ---------------------------- LINUX SETTINGS (edit these) ----------------------------
DATASET_DIR="/path/to/PES_university_data"       # folder that holds Healthy_controls, Schiz and testing
ATLAS_DIR="/path/to/atlas"                        # CIT168_T1w_head_700um.nii, prob_atlas_bilateral.nii, prob_atlas.nii, labels.txt
OUTPUT_DIR=""                                     # empty = <dataset folder name>_sn_pipeline_output next to the dataset folder
MODEL_DIR=""                                      # nnU-Net results folder with Dataset721_chealdev49r1; empty = train (GPU needed)
DEVICE="auto"                                     # auto, cpu or cuda
PYTHON_BIN="python3"
# --------------------------------------------------------------------------------------

set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
    echo "Creating the Python environment in .venv (first run only) ..."
    "$PYTHON_BIN" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip freeze > requirements-lock.txt
fi
set +e
.venv/bin/python -m sn_pipeline --dataset "$DATASET_DIR" --atlas "$ATLAS_DIR" --output "$OUTPUT_DIR" --model-dir "$MODEL_DIR" --device "$DEVICE" "$@"
RC=$?
echo
if [ "$RC" -eq 0 ]; then echo "Pipeline finished. Masks are in the sn_masks folder of the output folder."; else echo "Pipeline stopped with an error. See the log in the logs folder of the output folder."; fi
exit $RC
