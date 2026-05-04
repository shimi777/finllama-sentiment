"""Capture screenshots of the live Streamlit dashboard for embedding in the deck.

Assumes the dashboard is running at http://localhost:8502 (start with dashboard/run.bat).
Saves PNGs into presentation/key_figures/dashboard_*.png.

Usage: .venv/Scripts/python.exe scripts/screenshot_dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "presentation" / "key_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = os.environ.get("DASHBOARD_URL", "http://localhost:8502")
SHOTS = [
    # (filename, scroll_y_pixels, viewport_height) — full-page caps
    ("dashboard_overview.png", 0, 1100),
    ("dashboard_table_and_per_class.png", 900, 1300),
    ("dashboard_highlights.png", 2100, 1300),
    ("dashboard_per_example.png", 3300, 1100),
    ("dashboard_confusions.png", 4400, 1100),
]


def main() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(viewport={"width": 1400, "height": 1100})
            page = context.new_page()
            print(f"Loading {URL}…")
            page.goto(URL, wait_until="networkidle", timeout=60000)
            # Streamlit re-runs scripts on first load; let it settle
            page.wait_for_timeout(3000)

            # Full-page screenshot first
            full = OUT_DIR / "dashboard_full.png"
            page.screenshot(path=str(full), full_page=True)
            print(f"wrote {full}")

            # Section captures by scrolling
            for name, scroll_y, vh in SHOTS:
                page.set_viewport_size({"width": 1400, "height": vh})
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                page.wait_for_timeout(800)
                out = OUT_DIR / name
                page.screenshot(path=str(out), full_page=False)
                print(f"wrote {out}")
        finally:
            browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"screenshot failed: {e}", file=sys.stderr)
        sys.exit(1)
