"""Canonical application entry point: use the integrated, validated CLIMES app."""
from pathlib import Path
import os
import runpy
import sys


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    return runpy.run_path(str(root / 'climes_current' / 'integrated_final.py'), run_name='__main__')


if __name__ == '__main__':
    main()
