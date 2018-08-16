import sys
import os
import re
import time
import collections
import itertools
import pathlib
import attr
import numpy as np
import scipy.optimize
import skimage
import skimage.io
from skimage.morphology.misc import remove_small_objects
from skimage.measure import label, regionprops
import concurrent.futures


def segment(I, threshold):
    I[:,0] = 0
    I[:,-1] = 0
    I[0, :] = 0
    I[-1, :] = 0
    S = I > threshold
    S = remove_small_objects(S, 2000, connectivity=2)
    S = label(S, connectivity=2)
    stats = regionprops(S, I)
    return stats


def map_progress(pool, fn, *iterables):
    t_start = time.perf_counter()
    # Assumes all iterables are the same length!
    for it in iterables:
        if isinstance(it, collections.Sized):
            total = "/" + str(len(it))
            break
    else:
        total = ""
    for i, result in enumerate(pool.map(fn, *iterables), 1):
        print(f"\r    {i}{total}", end="")
        yield result
    t_end = time.perf_counter()
    print(f"    Elapsed: {t_end - t_start}")


@attr.s
class ReferenceColony(object):
    centroid = attr.ib()
    area = attr.ib()
    img = attr.ib()


@attr.s
class ComparisonColony(object):
    region = attr.ib()
    img = attr.ib()


def load_ref_img(path):
    f = path.open()
    x = float(f.readline()[8:])
    y = float(f.readline()[8:])
    centroid = (y, x)
    idx = re.findall(r'img_coords_(\d+)\.txt', path.name)[0]
    img_name = f"img_Cy5_{idx}.tif"
    img_path = recentered_path / img_name
    assert img_path.exists(), f"Missing image file: {img_name}"
    img = skimage.io.imread(img_path, plugin="tifffile")
    area = calc_ref_colony_area(img)
    colony = ReferenceColony(centroid, area, img)
    return colony


def calc_ref_colony_area(img):
    regions = segment(img, 500)
    center = np.array(img.shape) / 2
    centroids = np.array([r.centroid for r in regions])
    distances = np.linalg.norm(centroids - center, axis=1)
    idx = np.argmin(distances)
    return regions[idx].area


def extract_comparison_colony(stitched_img, region):
    y1, x1, y2, x2 = region.bbox
    img = stitched_img[y1:y2, x1:x2]
    colony = ComparisonColony(region, img)
    return colony


def match_reference_colonies(comp_colony, ref_colonies):
    img_c = comp_colony.img
    costs = np.empty(len(ref_colonies))
    costs[:] = -np.inf
    for ri, rc in enumerate(ref_colonies):
        crop_exact = np.subtract(rc.img.shape, img_c.shape) / 2
        if np.any(crop_exact <= 0):
            continue
        crop_before = np.floor(crop_exact)
        crop_after = np.ceil(crop_exact)
        crop = np.vstack([crop_before, crop_after]).astype(int).T
        img_r = skimage.util.crop(rc.img, crop)
        costs[ri] = ncc(img_c, img_r)
    return 1 - costs


def ncc(i1, i2):
    i1 = i1 - i1.mean()
    i2 = i2 - i2.mean()
    i1 /= np.linalg.norm(i1)
    i2 /= np.linalg.norm(i2)
    corr = np.dot(i1.reshape(-1), i2.reshape(-1))
    return corr


reference_path, stitched_img_path, bg_threshold = sys.argv[1:]

reference_path = pathlib.Path(reference_path)
bg_threshold = int(bg_threshold)

recentered_path = reference_path / 'recentered_images'

# Seems to be diminishing returns above 4 threads.
num_cpus = min(len(os.sched_getaffinity(0)), 4)
pool = concurrent.futures.ThreadPoolExecutor(num_cpus)

print("Loading and processing reference data")
centroid_paths = sorted(recentered_path.glob('img_coords_*.txt'))
ref_colonies = list(map_progress(pool, load_ref_img, centroid_paths))
assert len(set(c.img.shape for c in ref_colonies)) == 1, \
    "Reference image size mismatch"

print("Loading stitched image")
stitched_img = skimage.io.imread(stitched_img_path)

print("Segmenting stitched image")
regions = segment(stitched_img, bg_threshold)
comp_colonies = list(map(
    extract_comparison_colony, itertools.repeat(stitched_img), regions
))

costs = np.empty((len(comp_colonies), len(ref_colonies)))
print("Matching colony images")
costs = np.array(list(map_progress(
    pool, match_reference_colonies, comp_colonies,
    itertools.repeat(ref_colonies)
)))

pool.shutdown()

comp_idxs, ref_idxs = scipy.optimize.linear_sum_assignment(costs)
