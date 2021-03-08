import sys
import os
import re
import time
import collections.abc
import itertools
import pathlib
import attr
import numpy as np
import scipy.optimize
import skimage
import skimage.io
from skimage.morphology import remove_small_objects, dilation
from skimage.measure import label, regionprops
import concurrent.futures


def segment(I, threshold):
    S = I > threshold
    S = remove_small_objects(S, 2000, connectivity=2)
    S = label(S, connectivity=2).astype(np.uint16)
    regions = regionprops(dilation(S), I)
    for r in regions:
        if r.min_intensity == 0:
            y1, x1, y2, x2 = r.bbox
            S[y1:y2, x1:x2] = 0
    S = label(S, connectivity=2).astype(np.uint16)
    regions = regionprops(S, I)
    return regions


def map_progress(pool, fn, *iterables):
    t_start = time.perf_counter()
    # Assumes all iterables are the same length!
    for it in iterables:
        if isinstance(it, collections.abc.Sized):
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
    region = attr.ib()
    img = attr.ib()

    @property
    def intensity_image(self):
        return self.region.intensity_image

    @property
    def area(self):
        return self.region.area


@attr.s
class ComparisonColony(object):

    region = attr.ib()

    @property
    def intensity_image(self):
        return self.region.intensity_image

    @property
    def area(self):
        return self.region.area

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
    region = extract_ref_colony_region(img)
    colony = ReferenceColony(centroid, region, img)
    return colony


def extract_ref_colony_region(img):
    regions = segment(img, 500)
    center = np.array(img.shape) / 2
    centroids = np.array([r.centroid for r in regions])
    distances = np.linalg.norm(centroids - center, axis=1)
    idx = np.argmin(distances)
    return regions[idx]


def match_reference_colonies(comp_colony, ref_colonies):
    corr = np.array([pair_correlation(comp_colony, c) for c in ref_colonies])
    return 1 - corr


def pair_correlation(comp_colony, ref_colony):
    img_c = comp_colony.intensity_image
    img_r = ref_colony.img
    r1 = np.array(img_r.shape) / 2 - comp_colony.local_centroid
    r1 = np.floor(r1).astype(int)
    c1 = -np.minimum(r1, 0)
    r1 = np.maximum(r1, 0)
    r2 = np.minimum(r1 + img_c.shape - c1, img_r.shape)
    c2 = c1 + (r2 - r1)
    img_c = img_c[c1[0]:c2[0], c1[1]:c2[1]]
    img_r = img_r[r1[0]:r2[0], r1[1]:r2[1]]
    img_r = np.where(img_r > 500, img_r, 0)
    return normalized_cross_correlation(img_c, img_r)


def normalized_cross_correlation(i1, i2):
    i1 = i1 - i1.mean()
    i2 = i2 - i2.mean()
    n1 = np.linalg.norm(i1)
    n2 = np.linalg.norm(i2)
    if n1 == 0 or n2 == 0:
        return -10000
    i1 /= n1
    i2 /= n2
    corr = np.dot(i1.reshape(-1), i2.reshape(-1))
    return corr


def rotation_matrix(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c, s], [-s, c]])


def recover_transformation(P, Q, theta):
    # Kabsch algorithm.
    Pc = np.mean(P, axis=0)
    Qc = np.mean(Q, axis=0)
    P = P - Pc
    Q = Q - Qc
    Q = Q @ rotation_matrix(theta)
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
    theta_recovered = np.arctan2(b, a)
    assert np.allclose([scale_x, scale_y], 1), "Unexpected scale component"
    assert np.allclose(shear, 0), "Unexpected shear component"
    return T, theta_recovered


# Reads lots of globals -- only for debugging.
def plot_quality():
    import matplotlib.pyplot as plt
    import skimage.exposure
    qP = np.array([c.centroid for c in ref_colonies])
    qQ = np.array([c.centroid for c in comp_colonies])
    qPc = qP.mean(axis=0)
    qPt = ((qP - qPc) @ R.T + qPc - T) / scale
    fig = plt.figure()
    ax = fig.gca()
    img = stitched_img[::10,::10]
    img = skimage.exposure.rescale_intensity(img, (250, 65535))
    img = skimage.exposure.adjust_gamma(img, 1/2.2)
    extent = (0, stitched_img.shape[1], stitched_img.shape[0], 0)
    ax.imshow(img, extent=extent, cmap='gray')
    ax.plot(*np.flipud(np.transpose(np.dstack([Pt, Q]), (1,2,0))), c='olive')
    for i, (y,x) in enumerate(qPt):
        ax.annotate(i, (x,y), color='red', ha='right')
    for i, (y,x) in enumerate(qQ):
        ax.annotate(i, (x,y), color='deepskyblue', ha='left')


scale = 0.658
theta = -0.0036

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
stitched_img[:,0] = 0
stitched_img[:,-1] = 0
stitched_img[0, :] = 0
stitched_img[-1, :] = 0

print("Segmenting stitched image")
regions = segment(stitched_img, bg_threshold)
comp_colonies = [
    ComparisonColony(r) for r in regions
    if np.diff(r.bbox[0::2]) < 1040 and np.diff(r.bbox[1::2]) < 1392
]

# Filter reference colonies that are out of bounds in the comparison data.
# ref_colonies = [
#     c for c in ref_colonies
#     if c.centroid[0] < 4440 and c.centroid[1] < -477
# ]

# # Filter comparison colonies that are out of bounds in the reference data.
# comp_colonies = [
#     c for c in comp_colonies
#     if c.centroid[0] > 100 and c.centroid[1] > 350
# ]

print("Matching colony images")
costs = np.array(list(map_progress(
    pool, match_reference_colonies, comp_colonies,
    itertools.repeat(ref_colonies)
)))

pool.shutdown()

comp_idxs, ref_idxs = scipy.optimize.linear_sum_assignment(costs)
ref_colonies_matched = [ref_colonies[i] for i in ref_idxs]
comp_colonies_matched = [comp_colonies[i] for i in comp_idxs]

ref_c = np.array([c.centroid for c in ref_colonies_matched])
comp_c = np.array([c.centroid for c in comp_colonies_matched])

ref_center = np.mean(ref_c, axis=0)
comp_center = np.mean(comp_c, axis=0)
centroid_diffs = (ref_c - ref_center) / scale + comp_center - comp_c
keep = np.linalg.norm(centroid_diffs, axis=1) < 1040 / 2
keep_idxs = np.nonzero(keep)[0]

ref_colonies_matched = [ref_colonies_matched[i] for i in keep_idxs]
comp_colonies_matched = [comp_colonies_matched[i] for i in keep_idxs]
ref_c = np.array([c.centroid for c in ref_colonies_matched])
comp_c = np.array([c.centroid for c in comp_colonies_matched])
ref_a = np.array([c.area for c in ref_colonies_matched])
comp_a = np.array([c.area for c in comp_colonies_matched])

T, theta_recovered = recover_transformation(ref_c, comp_c * scale, theta)
if not np.allclose(theta_recovered, 0, atol=1e-5):
    print(f"Warning: theta_recovered is significant ({theta_recovered:g})")

R = rotation_matrix(theta)

P = ref_c
Q = comp_c
Pc = P.mean(axis=0)
Qc = Q.mean(axis=0)
Pt = ((P - Pc) @ R.T + Pc - T) / scale
Qt = ((Q - Qc) @ R + Qc) * scale + T

Derr = np.mean(np.linalg.norm(Pt - Q, axis=1))
Serr = np.mean(np.abs(comp_a - ref_a) / ref_a) * 100

print(f"Derr = {Derr:.1f} pixels")
print(f"Serr = {Serr:.2f} percent")
