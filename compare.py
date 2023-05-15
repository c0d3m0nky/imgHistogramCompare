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
ap.add_argument("-t", "--bucket-threshold", required=True, type=float, help="Bucket threshold")
ap.add_argument("-minb", "--min-bucket-size", required=False, default=0, type=float, help="Minimum bucket size")
ap.add_argument("-m", "--method", required=False, default='Chi-Squared', help="Comparison Method (Correlation|Chi-Squared|Intersection|Hellinger)")

_args = ap.parse_args()
histogram.init(_args)

if _args.plan:
    _args.trace = True


def histogram_grouping():
    output_path = Path(_args.dataset)
    dataset = Path(_args.dataset)

    files = utils.glob_images(_args.dataset)
    hists = histogram.get_histograms(files, _args.bin_size)

    values_index = []
    buckets_index = {}

    def post_compare(k1, k2, d):
        if k1 not in buckets_index:
            buckets_index[k1] = {}

        if k2 not in buckets_index:
            buckets_index[k2] = {}

        if (not reverse and d <= _args.bucket_threshold) or (reverse and d >= _args.bucket_threshold):
            values_index.append((d, k1, k2))
            buckets_index[k1][k2] = d
            buckets_index[k2][k1] = d

    method_name = _args.method
    reverse = True if method_name in ("Correlation", "Intersection") else False

    results = histogram.find_item_distances(hists, histogram.OPENCV_METHODS[method_name], post_compare)

    values_index = sorted([(v, k1, k2) for (v, k1, k2) in values_index], key=lambda x: x[0], reverse=reverse)

    if _args.trace:
        utils.dump_json(results, dataset / '__.results.json')
        utils.dump_json(values_index, dataset / '__.values_index.json')

    buckets = []
    already_in_bucket = {}

    for (v, a, b) in tqdm(values_index, desc='Pachinko!!!'):
        if a in already_in_bucket and b in already_in_bucket:
            continue
        elif a in already_in_bucket:
            bucket = already_in_bucket[a]
            bucket.append(b)
            already_in_bucket[b] = bucket
            continue
        elif b in already_in_bucket:
            bucket = already_in_bucket[b]
            bucket.append(a)
            already_in_bucket[a] = bucket
            continue

        # maybe make it where prime gets averaged and others get summed
        prime_candidate_buckets = []
        subprime_candidate_buckets = []

        for bucket in buckets:
            for k in bucket:
                adist = buckets_index[a][k] if k in buckets_index[a] else None
                bdist = buckets_index[b][k] if k in buckets_index[b] else None

                if adist and bdist:
                    prime_candidate_buckets.append((min(adist, bdist), bucket))
                elif adist:
                    subprime_candidate_buckets.append((adist, bucket))
                elif bdist:
                    subprime_candidate_buckets.append((bdist, bucket))

        candidate_buckets = prime_candidate_buckets if prime_candidate_buckets else subprime_candidate_buckets if subprime_candidate_buckets else None

        if candidate_buckets:
            candidate_buckets = sorted([(v, bucket) for (v, bucket) in candidate_buckets], key=lambda x: x[0], reverse=reverse)
            bucket = candidate_buckets[0][1]
            bucket.append(a)
            bucket.append(b)
            already_in_bucket[a] = bucket
            already_in_bucket[b] = bucket
            continue

        bucket = [a, b]
        buckets.append(bucket)
        already_in_bucket[a] = bucket
        already_in_bucket[b] = bucket

    if _args.trace:
        utils.dump_json(buckets, dataset / '__.buckets.json')

    if not _args.plan:
        with tqdm(len(already_in_bucket.keys()), desc='Moving files') as pbar:
            folderNameIndex = 0

            for bucket in buckets:
                if len(bucket) < _args.min_bucket_size:
                    continue
                fld = None

                while True:
                    folderNameIndex += 1
                    fld = output_path / str(folderNameIndex)
                    if not fld.exists():
                        fld.mkdir(parents=True, exist_ok=False)
                        break

                for k in bucket:
                    kk = Path(dataset / k)
                    kk.rename(Path(fld / k))
                    pbar.update()


if __name__ == "__main__":
    histogram_grouping()
