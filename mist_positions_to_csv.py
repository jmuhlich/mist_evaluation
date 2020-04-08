# Convert MIST output global positions format to CSV format as expected by the
# BII Stitching Challenege evaluation code.

import fileinput
import re

for line in fileinput.input():
    values = re.match(r'file: (.*?);.*position: \((.*?), (.*?)\)', line).groups()
    print(','.join(values))
