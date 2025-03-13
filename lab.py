import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.building import builder

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

x = pd.DataFrame({'A': [0.2,0.8], 'B': [3,4]})
print(x)
x['A_int'] = x['A'].round(decimals=0)
print(x)
