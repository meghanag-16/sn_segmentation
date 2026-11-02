:<<"::END_OF_WINDOWS_PART"
@echo off
REM =====================================================================================================
REM  PREDICTION ONLY from outputs that already exist (registration + crop + trained model).
REM  No audit, no registration, no cropping again: only nnU-Net prediction, post-processing, visual QC, summary.
REM  ONE file for Windows and Linux.
REM    Windows: double-click run_prediction_only.bat, or in Command Prompt:   run_prediction_only.bat
REM    Linux:   in a terminal:                                                bash run_prediction_only.bat
REM =====================================================================================================

REM ---------------------------- WINDOWS SETTINGS (edit these) ----------------------------
REM Folder that holds the earlier outputs (searched by file name; sub-folders are fine):
REM   subjects.csv, qc_table_1_8.csv (or qc_table.csv) + crop_windows.csv + crops\, phase_1_3b_decisions.json
REM   (or atlas_registration_decisions.json), and optionally nnUNet_results\Dataset721_chealdev49r1\...
set "SAVED_DIR=C:\Users\Meghana G\Desktop\engineering\cheal\kaggle_outputs"
REM Output folder for the masks, QC pages and summary
set "OUTPUT_DIR=C:\Users\Meghana G\Desktop\engineering\cheal\prediction_output"
REM nnU-Net results folder that contains Dataset721_chealdev49r1. Empty = search below SAVED_DIR
set "MODEL_DIR="
REM test = only the test subjects; all = every subject included for inference
set "SUBJECTS=test"
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
".venv\Scripts\python.exe" -m sn_pipeline.predict_from_saved --saved-dir "%SAVED_DIR%" --output "%OUTPUT_DIR%" --model-dir "%MODEL_DIR%" --subjects %SUBJECTS% --device %DEVICE% %*
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (echo Prediction finished. Masks are in the sn_masks folder of the output folder.) else (echo Prediction stopped with an error. See the log in the logs folder of the output folder.)
pause
exit /b %RC%
::END_OF_WINDOWS_PART

# =====================================================================================================
#  LINUX PART (run with:  bash run_prediction_only.bat  [extra options])
# =====================================================================================================

# ---------------------------- LINUX SETTINGS (edit these) ----------------------------
SAVED_DIR="/path/to/earlier_outputs"      # subjects.csv, qc_table_1_8.csv + crop_windows.csv + crops/, phase_1_3b_decisions.json
OUTPUT_DIR="/path/to/prediction_output"
MODEL_DIR=""                              # folder that contains Dataset721_chealdev49r1; empty = search below SAVED_DIR
SUBJECTS="test"                           # test or all
DEVICE="auto"                             # auto, cpu or cuda
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
.venv/bin/python -m sn_pipeline.predict_from_saved --saved-dir "$SAVED_DIR" --output "$OUTPUT_DIR" --model-dir "$MODEL_DIR" --subjects "$SUBJECTS" --device "$DEVICE" "$@"
RC=$?
echo
if [ "$RC" -eq 0 ]; then echo "Prediction finished. Masks are in the sn_masks folder of the output folder."; else echo "Prediction stopped with an error. See the log in the logs folder of the output folder."; fi
exit $RC
