#!/usr/bin/env python3
"""Build checked-in LaTeX inputs. Requires Tectonic; no experiment or install."""
import argparse
from pathlib import Path
import shutil
import subprocess
p = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler-path', help='path to tectonic executable')
a = parser.parse_args()
compiler = a.compiler_path or shutil.which('tectonic')
if not compiler:
    bundled = Path('/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic')
    compiler = str(bundled) if bundled.is_file() else None
if not compiler:
    parser.error('Tectonic not found; provide --compiler-path')
subprocess.run([compiler, '-X', 'compile', '--outdir', str(p), '--outfmt', 'pdf', '--print', '--untrusted', str(p/'manuscript.tex')], cwd=p, check=True)
