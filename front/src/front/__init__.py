import sys
from pathlib import Path

from streamlit.web import cli


def main():
    sys.argv = ["streamlit", "run", str(Path(__file__).parent / "app.py"), *sys.argv[1:]]
    sys.exit(cli.main())
