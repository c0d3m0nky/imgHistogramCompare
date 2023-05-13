from typing import Dict, List, Tuple
import argparse
import global_args
import cv2
from tqdm import tqdm
from pathlib import Path

import histogram
import utils

ap = argparse.ArgumentParser()
global_args.add_args(ap)
ap.add_argument("root", help="Path to the directory to dedupe")
ap.add_argument("-r", "--recurse", action='store_true', default=False, help="Recursive")
ap.add_argument("-trh", "--threshold", required=False, default=0, type=float, help="Comparison threshold")

_args = ap.parse_args()
histogram.init(_args)
_root = Path(_args.root)
_trace_path = Path(_args.trace_out if _args.trace_out else _root)


def _add_dupe(dupe_index: Dict[str, List[str]], a, b):
    if a in dupe_index:
        if b not in dupe_index[a]:
            dupe_index[a].append(b)

            if b in dupe_index:
                for dk in dupe_index[b]:
                    if dk not in dupe_index[a]:
                        dupe_index[a].append(dk)
                del dupe_index[b]
    elif b in dupe_index:
        if a not in dupe_index[b]:
            dupe_index[b].append(a)

            if a in dupe_index:
                for dk in dupe_index[a]:
                    if dk not in dupe_index[b]:
                        dupe_index[b].append(dk)
                del dupe_index[a]
    else:
        dupe_index[a] = [b]


def dedupe():
    # utils.set_normal_priority()
    files = utils.glob_images(_args.root)
    hists = histogram.get_histograms(files, _args.bin_size)
    dupe_index: Dict[str, List[str]] = {}
    log = []

    def post_compare(k1, k2, dd):
        if dd <= _args.threshold:
            log.append((k1, k2))
            _add_dupe(dupe_index, k1, k2)
            _add_dupe(dupe_index, k2, k1)

    results = histogram.find_item_distances(hists, cv2.HISTCMP_CHISQR, post_compare, True)

    if _args.trace:
        utils.dump_json(dupe_index, _trace_path / '__.dupe_index.json')
        utils.dump_json(results, _trace_path / '__.results.json')

    di = 0

    def map_file(file) -> Tuple[Path, int]:
        file = _root / file

        return (file, file.stat().st_size)

    for dk in tqdm(dupe_index.keys(), 'Marking dupes'):
        dupes: List[Tuple[Path, int]] = sorted([map_file(dk)] + list(map(map_file, dupe_index[dk])), key=lambda x: x[1], reverse=True)
        op = _root / "__.dupes"

        if not op.exists():
            op.mkdir()

        i = 0

        for d in dupes:
            (_root / d[0]).rename(op / f'__.dupe.{di}_{i}_._{d[0].name}')
            i += 1
        di += 1


dedupe()
