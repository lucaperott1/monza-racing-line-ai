@echo off
cd /d "%~dp0"
rem Uses Python 3.12 from its default install folder, or the "py" launcher if it is elsewhere.
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" 2_train.py
) else (
    py -3.12 2_train.py
)
pause
