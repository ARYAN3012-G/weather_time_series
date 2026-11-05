@echo off
REM ==========================================================================
REM  v3 drought forecasting — full run on the lab desktop (Windows)
REM  Run from the project root:   run_all.bat
REM  Each step can be re-run safely; finished model runs are skipped.
REM ==========================================================================
setlocal
cd /d %~dp0

REM Use the project virtual environment (.venv). Create it on first run.
REM PyTorch comes from the CUDA index: the default PyPI wheel on Windows is CPU-only.
if not exist .venv\Scripts\python.exe (
    echo [0/7] Creating .venv and installing PyTorch with CUDA
    python -m venv .venv || goto :error
    .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
    .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :error
)
set PY=.venv\Scripts\python.exe

echo [0/7] Installing requirements
%PY% -m pip install -r requirements.txt || goto :error
%PY% -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo [1/7] Site audit + duplicate removal + SPEI recalibration (seconds)
%PY% src\dataprep\prepare_data.py || goto :error

echo [2/7] Integrity / leakage tests
%PY% -m pytest tests -q || goto :error

echo [3/7] Smoke test (about 1-2 minutes)
%PY% src\experiments\run_experiments.py --quick --horizons 7 --seeds 0 || goto :error

echo [4/7] MAIN STUDY: temporal split, 7/15/30 days, 3 feature sets, 4 models, 5 seeds
%PY% src\experiments\run_experiments.py --split temporal || goto :error

echo [5/7] Leave-one-cluster-out (spatial generalisation)
%PY% src\experiments\run_experiments.py --split loco --feature-sets weather weather_spei || goto :error

echo [6/7] Marusov et al. protocol comparison
%PY% src\experiments\marusov_protocol.py || goto :error

echo [7/7] Aggregate: tables, bootstrap CIs, thresholds, reports
%PY% src\experiments\aggregate.py || goto :error

echo.
echo DONE. Open results\v3\tables\REPORT_temporal.md, REPORT_loco.md and
echo results\v3\marusov_protocol\MARUSOV_PROTOCOL.md
goto :eof

:error
echo.
echo FAILED at the step above. Check the newest file in logs\ . Re-running resumes where it stopped.
exit /b 1
