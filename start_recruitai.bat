@echo off
echo ========================================
echo       RecruitAI - Starting Server
echo ========================================

REM ── GEMINI API KEY ─────────────────────────────────────────────────────────
set GEMINI_API_KEY=AIzaSyBjgFXlqUfiPq-KSGZJVo2xoEbu7v2FfUU
setx GEMINI_API_KEY %GEMINI_API_KEY% >nul 2>&1
echo [OK] Gemini API Key set

REM ── START NGROK IN BACKGROUND ───────────────────────────────────────────────
echo [..] Starting ngrok tunnel on port 5000...
start /B ngrok http 5000 --log=stdout > ngrok.log 2>&1

REM ── WAIT FOR NGROK TO INITIALIZE ────────────────────────────────────────────
timeout /t 3 /nobreak >nul

REM ── FETCH NGROK PUBLIC URL VIA ITS LOCAL API ────────────────────────────────
echo [..] Fetching ngrok public URL...
for /f "delims=" %%i in ('powershell -NoProfile -Command ^
  "try { $r = Invoke-RestMethod http://127.0.0.1:4040/api/tunnels; $r.tunnels[0].public_url } catch { '' }"') do set NGROK_URL=%%i

REM ── CHECK IF WE GOT A URL ───────────────────────────────────────────────────
if "%NGROK_URL%"=="" (
    echo [!!] Could not get ngrok URL.
    echo [!!] Make sure ngrok is installed: https://ngrok.com/download
    echo [!!] Falling back to localhost only - candidates on other networks won't be able to join.
    set BASE_URL=http://localhost:5000
) else (
    set BASE_URL=%NGROK_URL%
    echo [OK] ngrok tunnel active!
    echo [OK] Public URL: %NGROK_URL%
    echo.
    echo =====================================================
    echo  Share this URL with candidates:
    echo  %NGROK_URL%
    echo =====================================================
)

echo.
setx BASE_URL %BASE_URL% >nul 2>&1
echo [OK] BASE_URL set to: %BASE_URL%
echo [OK] Email: Each recruiter sets up their Gmail in the dashboard (one time)
echo [OK] Starting Flask server...
echo [OK] Open browser at: %BASE_URL%
echo.

cd /d "%~dp0"
python app.py

pause