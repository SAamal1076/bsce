@echo off
rem bsce workbench: double-click to open the whole workflow (environment -> data -> training -> figures) in your browser.
rem Keep this window open while runs are going. Closing it stops the app and its runs.
cd /d "%~dp0"
set "PY="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\Anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniforge3") do (
  if not defined PY if exist "%%~P\python.exe" set "PY=%%~P\python.exe"
)
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY (
  echo Anaconda's Python was not found.
  echo Open the Anaconda Prompt, go to this folder and run:   python app\bsce_app.py
  pause
  exit /b 1
)
echo Starting the bsce workbench with %PY%
"%PY%" app\bsce_app.py %*
pause
