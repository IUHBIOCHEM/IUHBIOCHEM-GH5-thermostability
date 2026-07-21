#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
14_run_cdhit_cluster_GH5.py
Run CD-HIT to cluster GH5 sequences for non-leaking (group-based) splits.

Outputs:
  - <out_prefix>.fasta
  - <out_prefix>.clstr
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def pick_word_size(c: float) -> int:
    # protein CD-HIT typical choices
    if c >= 0.88:
        return 5
    if c >= 0.80:
        return 4
    return 3


def main():
    ap = argparse.ArgumentParser(description="Run CD-HIT clustering for GH5 FASTA.")
    ap.add_argument("-i", "--input", required=True, help="Input FASTA (e.g., GH5_bacteria.fasta)")
    ap.add_argument("-o", "--out_prefix", required=True, help="Output prefix (e.g., GH5_cdhit80)")
    ap.add_argument("--c", type=float, default=0.8, help="Identity threshold (0.8 or 0.9). Default=0.8")
    ap.add_argument("--threads", type=int, default=8, help="Threads (-T). Default=8")
    ap.add_argument("--mem_mb", type=int, default=0, help="Memory MB (-M). 0=unlimited. Default=0")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input FASTA not found: {inp}")

    cdhit = shutil.which("cd-hit")
    if not cdhit:
        raise RuntimeError(
            "cd-hit not found in PATH.\n"
            "Install on macOS (conda):  conda install -c bioconda cd-hit\n"
            "or (homebrew): brew install cd-hit"
        )

    c = float(args.c)
    if not (0.5 <= c <= 1.0):
        raise ValueError("--c must be between 0.5 and 1.0 (typical 0.8 or 0.9).")

    n = pick_word_size(c)

    out_prefix = Path(args.out_prefix)
    out_fasta = out_prefix.with_suffix(".fasta")
    # CD-HIT ghi file cluster là "<-o value>.clstr" (nối thêm .clstr), KHÔNG thay đuôi
    out_clstr = Path(str(out_fasta) + ".clstr")

    cmd = [
        cdhit,
        "-i", str(inp),
        "-o", str(out_fasta),
        "-c", f"{c:.2f}",
        "-n", str(n),
        "-T", str(args.threads),
        "-M", str(args.mem_mb),
    ]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.stdout.strip():
        print(res.stdout)
    if res.stderr.strip():
        print(res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"CD-HIT failed with return code {res.returncode}")

    if not out_fasta.exists():
        raise RuntimeError(f"Expected output FASTA not found: {out_fasta}")
    if not out_clstr.exists():
        raise RuntimeError(f"Expected cluster file not found: {out_clstr}")

    print("\nDONE.")
    print("Output FASTA :", out_fasta)
    print("Output CLSTR :", out_clstr)


if __name__ == "__main__":
    main()

