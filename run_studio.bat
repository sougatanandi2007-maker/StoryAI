@echo off
echo ===================================================================
echo     STORYWEAVE AI - INVESTIGATIVE STORYTELLING STUDIO
echo ===================================================================
echo.
cd /d "%~dp0writer-helper\story_ui_repo"

echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "StoryWeave Backend" cmd /k "python backend/run.py"

echo Starting Vite React Frontend on http://localhost:5173 ...
start "StoryWeave Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Studio services dispatched.
echo - Frontend: http://localhost:5173
echo - Backend API: http://127.0.0.1:8000
echo - Swagger Docs: http://127.0.0.1:8000/docs
echo ===================================================================
pause

