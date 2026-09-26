"""Run with: python -m streamlit run app.py."""

from pathlib import Path
import sys

# Also support running directly from a fresh source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nfl_sims.peter_app import main

main()
