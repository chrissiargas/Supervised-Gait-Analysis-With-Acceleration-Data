import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.building import builder

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

x = pd.DataFrame({'A': [1,2], 'B': [3,4]})
print(x)
x = x.rename(columns={'A':'B', 'B':'A'})
print(x)
