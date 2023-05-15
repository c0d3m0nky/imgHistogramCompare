from typing import List, Dict, Callable, Tuple, Union
from multiprocessing import Pool, RLock, freeze_support
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt

import utils

cv2.ocl.setUseOpenCL(True)

_args = None

OPENCV_METHODS = {
    "Correlation": cv2.HISTCMP_CORREL,
    "Chi-Squared": cv2.HISTCMP_CHISQR,
    "Intersection": cv2.HISTCMP_INTERSECT,
    "Hellinger": cv2.HISTCMP_BHATTACHARYYA
}


def init(args):
    global _args
    _args = args


class HueData:
    hue: int
    hsv_raw_image: np.ndarray
    hsv_image: np.ndarray
    bgr_image: np.ndarray
    histogram: np.ndarray

    def __init__(self, hue: int, hsv_raw_image: np.ndarray, hsv_image: np.ndarray, bgr_image: np.ndarray, histogram: np.ndarray):
        self.hue = hue
        self.bgr_image = bgr_image
        self.hsv_image = hsv_image
        self.histogram = histogram


def get_histograms(files: List[str], bin_size: int = 256, map_key: Callable[[str], str] = None) -> Dict[str, np.ndarray]:
    bin_size = bin_size if bin_size else 256
    hists = {}

    with Pool(processes=_args.processes) as p:
        max_ = len(files)
        with tqdm(total=max_, desc='Generating histograms') as pbar:
            iterr = map(lambda f: (f, bin_size), files)
            for res in p.imap_unordered(_get_histogram, iterr):
                k = res[0] if not map_key else map_key(res[0])
                hists[k] = res[1]
                pbar.update()

    return hists


def get_primary_hues(files: List[str], top_n: int = 3, map_key: Callable[[str], str] = None) -> Dict[str, List[HueData]]:
    top_n = top_n if top_n else 3
    hues = {}

    with Pool(processes=_args.processes) as p:
        max_ = len(files)
        with tqdm(total=max_, desc='Getting hue data') as pbar:
            iterr = map(lambda f: (f, top_n), files)
            for res in p.imap_unordered(_get_primary_hues, iterr):
                k = res[0] if not map_key else map_key(res[0])
                hues[k] = res[1]
                pbar.update()

    return hues


def find_item_distances(hists: Dict[str, np.ndarray], method: int, post_compare: Callable[[str, str, float], None] = None, slim=False) -> Dict[str, Dict[str, float]]:
    results = {}
    processed = [] if slim else None
    total = len(hists.items())
    total = total * (total - 1)

    if slim:
        total = total / 2

    with tqdm(total=total, desc='Calculating distances') as pbar:
        for (k1, hist1) in hists.items():
            results[k1] = {}
            with Pool(processes=_args.processes) as p:
                iterr = map(lambda a: (k1, hist1, a[0], a[1], method, processed), hists.items())

                for res in p.imap_unordered(_find_distance, iterr):
                    if res:
                        (k2, d) = res
                        results[k1][k2] = d
                        if slim:
                            processed.append((k1, k2))
                        if post_compare:
                            post_compare(k1, k2, d)
                        pbar.update()
    return results


def find_distances(hist1: Dict[str, np.ndarray], hist2: Dict[str, np.ndarray], method: int, post_compare: Callable[[str, str, float], None] = None) -> Dict[str, Dict[str, float]]:
    results = {}
    total = (len(hist1.items()) * len(hist2.items()))

    with tqdm(total=total, desc='Calculating distances') as pbar:
        for (k1, hist1) in hist1.items():
            results[k1] = {}
            with Pool(processes=_args.processes) as p:
                iterr = map(lambda a: (k1, hist1, a[0], a[1], method, None), hist2.items())

                for res in p.imap_unordered(_find_distance, iterr):
                    if res:
                        (k2, d) = res
                        results[k1][k2] = d

                        if post_compare:
                            post_compare(k1, k2, d)
                        pbar.update()
    return results


def compare(h1, h2, method):
    # compute the distance between the two histograms
    # using the method and update the results dictionary
    return cv2.compareHist(h1, h2, method)


def _find_distance(args) -> Union[Tuple[str, float], None]:
    (k1, hist1, k2, hist2, method, processed) = args
    if k1 == k2:
        return None
    if processed is not None and (k2, k1) in processed:
        return None
    d = compare(hist1, hist2, method)

    return (k2, d)


# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
def _get_histogram(args):
    (file, bin_size) = args
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    image = cv2.imread(file)

    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the hists
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, bin_size, 0, bin_size, 0, bin_size])
    hist = cv2.normalize(hist, hist).flatten()

    return (file, hist)


def _get_primary_hues(args):
    (file, top_n) = args
    img = cv2.imread(file)
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsvimg], [0], None, [180], [0, 180])
    # Todo: find peaks from full hsv too
    hist2 = cv2.calcHist([hsvimg], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 255, 0, 255])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    peaks: List[Tuple[int, int]] = []

    for i in range(0, top_n):
        peaks.append((-1, 0))

    for hv in enumerate(hist):
        pc = hv
        for (i, peak) in enumerate(peaks):
            if peak[0] == -1:
                peaks[i] = pc
                break
            elif pc[1] > peak[1]:
                peaks[i] = pc
                pc = peak

    hues: List[HueData] = []

    for p in peaks:
        # create image of pure hue
        hsvpure = np.zeros((100, 100, 3), np.uint8)
        hsvpure[:] = p[0]
        bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        hist = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hues.append(HueData(p[0], hsvpure, bgr, hist))

    return (file, hues)
