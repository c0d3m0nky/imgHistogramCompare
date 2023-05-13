from typing import Dict, Any
from tqdm import tqdm
import argparse
import glob
from pathlib import Path

import global_args
import histogram
import utils

ap = argparse.ArgumentParser()
global_args.add_args(ap)
ap.add_argument("dataset", help="Path to the directory of images")
ap.add_argument("-ext", "--extended-colors", action='store_true', help="Include extended colors")
ap.add_argument("-skn", "--skin-colors", action='store_true', help="Include skin colors")

_args = ap.parse_args()
histogram.init(_args)

if _args.plan:
    _args.trace = True


def sort():
    output_path = Path(_args.dataset)
    dataset = Path(_args.dataset)

    files = utils.glob_images(_args.dataset)
    hists = histogram.get_histograms(files, _args.bin_size)
    colors: Dict[str, Any] = utils.unpickle_obj('./colors/cache.pickle')

    if _args.extended_colors:
        cd: Dict[str, Any] = utils.unpickle_obj('./colors/ext/cache.pickle')
        cd.update(colors)
        colors = cd

    if _args.skin_colors:
        cd: Dict[str, Any] = utils.unpickle_obj('./colors/skin/cache.pickle')
        cd.update(colors)
        colors = cd

    results = histogram.find_distances(hists, colors, histogram.OPENCV_METHODS['Chi-Squared'])

    if _args.trace:
        utils.dump_json(results, dataset / '__.results.json')

    buckets_index: Dict[str, Dict[str, float]] = {}

    for (k, dists) in tqdm(results.items(), desc='Pachinko!!!'):
        (closest, dist) = min(dists.items(), key=lambda d: d[1])
        if closest not in buckets_index:
            buckets_index[closest] = {}
        buckets_index[closest][k] = dist

    if _args.trace:
        utils.dump_json(buckets_index, dataset / '__.buckets_index.json')

    if not _args.plan:
        with tqdm(total=len(results.keys()), desc='Moving files') as pbar:
            for (bk, b) in buckets_index.items():
                fld = output_path / bk

                if not fld.exists():
                    fld.mkdir()

                for k in b.keys():
                    (dataset / k).rename(fld / k)
                    pbar.update()


def _generate_colors_hists():
    r = histogram.get_histograms(glob.glob('colors/*.png'), map_key=lambda f: Path(f).stem)
    utils.pickle_obj(r, Path('./colors') / 'cache.pickle')
    r = histogram.get_histograms(glob.glob('colors/ext/*.png'), map_key=lambda f: Path(f).stem)
    utils.pickle_obj(r, Path('./colors/ext') / 'cache.pickle')
    r = histogram.get_histograms(glob.glob('colors/skin/*.png'), map_key=lambda f: Path(f).stem)
    utils.pickle_obj(r, Path('./colors/skin') / 'cache.pickle')


sort()
# _generate_colors_hists()
