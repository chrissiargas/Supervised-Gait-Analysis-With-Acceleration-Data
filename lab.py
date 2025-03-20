import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.building import builder

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
from scipy.spatial.transform import Rotation

angle = np.random.normal(loc=0, scale=15, size=10000)

plt.hist(angle, bins=100)
plt.show()
