import numpy as np
import pandas as pd

from post_processing.postprocess import *
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

B = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
C = pd.DataFrame({'a': [1,2,3], 'c': [7,8,9]})
print(pd.merge(B, C, on='a', how='left'))