import sys
import pathlib
import re
import numpy as np
import pandas as pd
import skimage.io
import ashlar.utils

if len(sys.argv) != 4:
    print("Usage: global_positions_file tile_image_dir output_file")
    sys.exit()

global_positions_file, tile_image_dir, output_file = (
    pathlib.Path(x) for x in sys.argv[1:]
)
df = pd.read_csv(
    global_positions_file, sep='; ?', engine='python', index_col=False,
    names=['file', 'corr', 'position', 'grid']
)
for name, col in df.iteritems():
    col[:] = col.str.replace(r'^\w+: ', '')
xy = df['position'].str.extract(r'\((?P<x>\d+), (?P<y>\d+)').astype(int)
df = pd.concat([df[['file']], xy], axis=1)

assert df['x'].min() == 0 and df['y'].min() == 0

tile = skimage.io.imread(str(tile_image_dir / df.iloc[0]['file']))
th, tw = tile.shape
mh = df['y'].max() + th
mw = df['x'].max() + tw
img = np.zeros((mh, mw), tile.dtype)

for rec in df.itertuples():
    print(f"{rec.file:20}  x:{rec.x:6}  y:{rec.y:6}")
    tile = skimage.io.imread(str(tile_image_dir / rec.file))
    mslice = img[rec.y:rec.y+th, rec.x:rec.x+tw]
    mslice[:] = ashlar.utils.pastefunc_blend(mslice, tile)

skimage.io.imsave(str(output_file), img, check_contrast=False)
