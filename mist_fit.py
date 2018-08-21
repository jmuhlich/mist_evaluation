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

    @property
    def img(self):
        return self.region.intensity_image

    @property
    def centroid(self):
        return self.region.centroid

    @property
    def local_centroid(self):
        return self.region.local_centroid


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


def match_reference_colonies(comp_colony, ref_colonies):
    corr = np.empty(len(ref_colonies))
    corr[:] = -10000
    for ri, rc in enumerate(ref_colonies):
        r1 = np.array(rc.img.shape) / 2 - comp_colony.local_centroid
        r1 = np.floor(r1).astype(int)
        c1 = -np.minimum(r1, 0)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r1 + comp_colony.img.shape - c1, rc.img.shape)
        c2 = c1 + (r2 - r1)
        img_c = comp_colony.img[c1[0]:c2[0], c1[1]:c2[1]]
        img_r = rc.img[r1[0]:r2[0], r1[1]:r2[1]]
        img_r = np.where(img_r > 500, img_r, 0)
        corr[ri] = ncc(img_c, img_r)
    return 1 - corr


def ncc(i1, i2):
    i1 = i1 - i1.mean()
    i2 = i2 - i2.mean()
    i1 /= np.linalg.norm(i1)
    i2 /= np.linalg.norm(i2)
    corr = np.dot(i1.reshape(-1), i2.reshape(-1))
    return corr


def recover_transformation(P, Q):
    # Kabsch algorithm.
    Pc = np.mean(P, axis=0)
    Qc = np.mean(Q, axis=0)
    P = P - Pc
    Q = Q - Qc
    H = P.T @ Q
    B, L, Wt = np.linalg.svd(H)
    R = Wt.T @ B.T
    T = -R @ Qc + Pc
    # Extract rotation angle from rotation matrix.
    # Based on https://math.stackexchange.com/a/78165 .
    det = np.linalg.det(R)
    assert det != 0, "Degenerate matrix"
    (a, b), (c, d) = R
    scale_y = np.linalg.norm([a, b])
    scale_x = det / scale_y
    shear = (a * c + b * d) / det
    theta = np.arctan2(b, a)
    assert np.allclose([scale_x, scale_y], 1), "Unexpected scale component"
    assert np.allclose(shear, 0), "Unexpected shear component"
    return T, R, theta


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
comp_colonies = [ComparisonColony(r) for r in regions]

# Filter reference colonies that are out of bounds in the comparison data.
ref_colonies = [
    c for c in ref_colonies
    if c.centroid[0] < 4440 and c.centroid[1] < -477
]

# Filter comparison colonies that are out of bounds in the reference data.
comp_colonies = [
    c for c in comp_colonies
    if c.centroid[0] > 100 and c.centroid[1] > 350
]

print("Matching colony images")
costs = np.array(list(map_progress(
    pool, match_reference_colonies, comp_colonies,
    itertools.repeat(ref_colonies)
)))

pool.shutdown()

comp_idxs, ref_idxs = scipy.optimize.linear_sum_assignment(costs)

ref_c = np.array([ref_colonies[i].centroid for i in ref_idxs])
comp_c = np.array([comp_colonies[i].centroid for i in comp_idxs]) * .658

T, R, theta = recover_transformation(ref_c, comp_c)
