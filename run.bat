@echo off
cd /d "%~dp0backend"

start chrome http://localhost:8000

uvicorn main:app --port 8000
