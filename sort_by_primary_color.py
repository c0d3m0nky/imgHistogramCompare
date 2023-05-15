from typing import Dict, List
from tqdm import tqdm
import argparse
import glob
import time
from pathlib import Path

import global_args
import histogram
import utils

ap = argparse.ArgumentParser()
global_args.add_args(ap)
ap.add_argument("dataset", help="Path to the directory of images")
ap.add_argument("-ext", "--extended-colors", action='store_true', help="Include extended colors")
ap.add_argument("-skn", "--skin-colors", action='store_true', help="Include skin colors")
ap.add_argument("-ex", "--exclude-colors", type=str, help="Exclude colors (comma delimited)")
ap.add_argument("-minb", "--min-bucket-size", required=False, default=0, type=float, help="Minimum bucket size")

_args = ap.parse_args()
histogram.init(_args)

# Find primary color, then compare that to colors and sort by closest
_common_ignored_colors = ['black', 'white', 'gray', 'silver']
_skintone_hues = []

# print('Rebuild colors')
# exit()

if _args.plan:
    _args.trace = True


def sort():
    output_path = Path(_args.dataset)
    dataset = Path(_args.dataset)

    files = utils.glob_images(_args.dataset)
    # hists = histogram.get_histograms(files, _args.bin_size)
    file_hues: Dict[str, List[histogram.HueData]] = histogram.get_primary_hues(files, map_key=lambda f: Path(f).name)
    # colors: Dict[str, Any] = utils.unpickle_obj('./colors/cache.pickle')
    #
    # if _args.extended_colors:
    #     cd: Dict[str, Any] = utils.unpickle_obj('./colors/ext/cache.pickle')
    #     cd.update(colors)
    #     colors = cd
    #
    # if _args.skin_colors:
    #     cd: Dict[str, Any] = utils.unpickle_obj('./colors/skin/cache.pickle')
    #     cd.update(colors)
    #     colors = cd
    #
    # if _args.exclude_colors:
    #     exc = list(map(lambda s: s.strip(), _args.exclude_colors.split(',')))
    #     print(f'Excluding colors {exc}')
    #
    #     for c in exc:
    #         if c in colors:
    #             del colors[c]
    #
    #     print(f'Using colors {colors.keys()}')
    #     time.sleep(3)

    if not _args.plan:
        with tqdm(total=len(file_hues.keys()), desc='Moving files') as pbar:
            for (fk, hues) in file_hues.items():
                hues: List[histogram.HueData] = hues
                mx = max(hues, key=lambda h: h.hue)
                for hue_data in hues:
                    # ToDo: filter out skin tones
                    if hue_data.hue == mx.hue:
                        fld = output_path / str(hue_data.hue)

                        if not fld.exists():
                            fld.mkdir()

                        (dataset / fk).rename(fld / fk)
                        pbar.update()
                        break


    # results = histogram.find_distances(hists, colors, histogram.OPENCV_METHODS['Chi-Squared'])
    #
    # if _args.trace:
    #     utils.dump_json(results, dataset / '__.results.json')
    #
    # buckets_index: Dict[str, Dict[str, float]] = {}
    #
    # for (k, dists) in tqdm(results.items(), desc='Pachinko!!!'):
    #     (closest, dist) = min(dists.items(), key=lambda d: d[1])
    #     if closest not in buckets_index:
    #         buckets_index[closest] = {}
    #     buckets_index[closest][k] = dist
    #
    # if _args.trace:
    #     utils.dump_json(buckets_index, dataset / '__.buckets_index.json')
    #
    # if not _args.plan:
    #     with tqdm(total=len(results.keys()), desc='Moving files') as pbar:
    #         for (bk, b) in buckets_index.items():
    #             if len(b) < _args.min_bucket_size:
    #                 continue
    #             fld = output_path / bk
    #
    #             if not fld.exists():
    #                 fld.mkdir()
    #
    #             for k in b.keys():
    #                 (dataset / k).rename(fld / k)
    #                 pbar.update()


def _generate_colors_hists():
    r = histogram.get_histograms(glob.glob('colors/*.png'), map_key=lambda f: Path(f).stem)
    utils.pickle_obj(r, Path('./colors') / 'cache.pickle')
    r = histogram.get_histograms(glob.glob('colors/ext/*.png'), map_key=lambda f: Path(f).stem)
    utils.pickle_obj(r, Path('./colors/ext') / 'cache.pickle')
    r = histogram.get_histograms(glob.glob('colors/skin/*.png'), map_key=lambda f: Path(f).stem)
    utils.pickle_obj(r, Path('./colors/skin') / 'cache.pickle')


if __name__ == "__main__":
    sort()
    # _generate_colors_hists()
