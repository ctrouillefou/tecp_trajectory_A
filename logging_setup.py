#!/usr/bin/env python3
# =====================================================================
# logging_setup.py
#
# Publication-grade log helpers for TECP runs.
# Matches the multi-config log format used in V10.3-Audit, adds:
#   - provenance header (date, host, user, git, versions, parameters)
#   - completion footer  (elapsed, verdicts, violations, files)
#
# Usage:
#   from logging_setup import log_run_header, log_run_footer
#   log_run_header("Porte 1bis — Matter balance test", config_lines)
#   ...
#   log_run_footer(elapsed_s, verdicts=..., files=..., violations=...)
#
# All prints are flushed so the log is safe under nohup.
# =====================================================================

import os
import sys
import socket
import subprocess
import datetime


SEP = "=" * 78
SUB = "-" * 78


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
    except Exception:
        return "unavailable"


def _version(pkg):
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def log_run_header(title, config_lines, extra_lines=None):
    """Print a provenance header at the start of a run."""
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(SEP)
    print(title)
    print(SEP)
    print(f"Date (UTC)     : {now}")
    print(f"Host           : {socket.gethostname()}")
    print(f"User           : {os.environ.get('USER', 'unknown')}")
    print(f"Working dir    : {os.getcwd()}")
    print(f"Git commit     : {_git_hash()}")
    print(f"Python         : {sys.version.split()[0]}")
    print(f"FiPy           : {_version('fipy')}")
    print(f"NumPy          : {_version('numpy')}")
    print(SUB)
    print("Configuration:")
    for line in config_lines:
        print(f"  {line}")
    if extra_lines:
        print(SUB)
        for line in extra_lines:
            print(f"  {line}")
    print(SEP)
    sys.stdout.flush()


def log_run_footer(elapsed_s, verdicts=None, files=None, violations=None,
                   extra_lines=None):
    """Print a completion footer."""
    print(SEP)
    print("RUN COMPLETE")
    print(SEP)
    print(f"Elapsed        : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    if verdicts:
        print("Verdicts       :")
        for name, v in verdicts.items():
            print(f"  {name:<10s}: {v}")
    if violations is not None:
        print(f"Positivity     : {violations} violations")
    if files:
        print("Files written  :")
        for f in files:
            print(f"  - {f}")
    if extra_lines:
        print(SUB)
        for line in extra_lines:
            print(f"  {line}")
    print(SEP)
    sys.stdout.flush()