@echo off
REM Launch the separate NER dashboard.
REM Run from repo root: dashboard_ner\run.bat
.venv\Scripts\python.exe -m streamlit run dashboard_ner\app.py
