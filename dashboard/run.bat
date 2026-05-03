@echo off
REM Launch the Streamlit dashboard from the project root using the venv Python.
cd /d %~dp0..
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
