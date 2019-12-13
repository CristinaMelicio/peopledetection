import numpy as np
import sys

if len(sys.argv) > 1:
    rec_id = int(sys.argv[1])

np.save('rec_id.npy', rec_id)
