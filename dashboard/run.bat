@echo off
REM Launch the Streamlit dashboard from the project root using the venv Python.
REM Pinned to port 8502 (matches scripts/screenshot_dashboard.py and the deck's
REM "localhost:8502" references).
cd /d %~dp0..
.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8502
