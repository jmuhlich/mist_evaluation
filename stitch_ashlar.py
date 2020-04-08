import sys
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import numpy as np
from ashlar.reg import EdgeAligner
from ashlar.nist import NistReader


input_path, percent_overlap, output_path = sys.argv[1:]

overlap = int(percent_overlap) / 100.0
assert Path(output_path).parent.exists(), "Output location does not exist"
pattern = 'img_w1_r{row:03}_c{col:03}_t000000000_Cy5_000.tif'

reader = NistReader(input_path, pattern=pattern, overlap=overlap)

aligner = EdgeAligner(reader, verbose=True)
aligner.run()

filenames = [reader.filename(i) for i in range(reader.metadata.num_images)]
positions = np.fliplr(aligner.positions)
with open(output_path, 'w') as f:
    for filename, (x, y) in zip(filenames, positions):
        f.write("%s,%.3f,%.3f\n" % (filename, x, y))
