import argparse
import os


def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("-p", "--plan", action='store_true', help="Min comparison value")
    ap.add_argument("-tr", "--trace", action='store_true', help="Dump index JSON files")
    ap.add_argument("--trace-out", type=str, help='Path to dump trace')
    ap.add_argument("-pr", "--processes", required=False, default=int(os.cpu_count() / 4) + 1, type=int, help="Max processes")
    ap.add_argument("-bs", "--bin-size", required=False, default=None, type=int, help="Histogram bin size")
