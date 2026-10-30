@echo off
title GRID GUARD AI Launcher
echo ========================================================
echo           GRID GUARD AI - Diagnostic System
echo ========================================================
echo.
echo [1/2] Checking Python dependencies...
python -m pip install -r requirements.txt --quiet
echo.
echo [2/2] Launching Grid Guard AI Dashboard...
echo.
echo The dashboard will automatically open in your default browser.
echo Press Ctrl+C in this window anytime to stop the server.
echo ========================================================
python -m streamlit run app.py
pause
