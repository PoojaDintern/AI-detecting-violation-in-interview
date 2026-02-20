@echo off
echo ========================================
echo       RecruitAI - Starting Server
echo ========================================

REM ── SET YOUR GEMINI API KEY HERE ──
REM ── Get it free from: https://aistudio.google.com/app/apikey ──
set GEMINI_API_KEY=AIzaSyBjgFXlqUfiPq-KSGZJVo2xoEbu7v2FfUU

REM ── Sets it permanently so you don't need to set it every time ──
setx GEMINI_API_KEY %GEMINI_API_KEY% >nul 2>&1

echo [OK] Gemini API Key set
echo [OK] Starting Flask server...
echo [OK] Open browser at: http://localhost:5000
echo.

cd /d "%~dp0"
python app.py

pause