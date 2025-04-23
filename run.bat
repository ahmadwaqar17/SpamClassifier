@echo off
echo Starting SMS Spam Detection Application...

REM Start the Flask backend server
start cmd /k "echo Starting Flask backend server... && python app.py"

REM Wait for the backend to initialize
timeout /t 5

REM Start the React frontend
start cmd /k "echo Starting React frontend... && cd frontend && npm run dev"

echo Both servers are starting. Please wait...
echo.
echo Backend: http://localhost:5000
echo Frontend: Check the terminal for the URL (typically http://localhost:5173)
echo.
echo Press any key to stop all servers...
pause > nul

REM Kill all servers when the user presses a key
taskkill /f /im cmd.exe
