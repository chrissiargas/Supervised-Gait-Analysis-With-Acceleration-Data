import os

import numpy as np
import pandas as pd

from post_processing.postprocess import *
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from datetime import datetime
from plot_utils.plots import plot_cummulative

import pandas as pd
from experiments import get_dataset_subjects

# Load the CSV file
file_path = os.path.join(
    os.path.expanduser('~'),
    'datasets',
    'wrist_gait'
)

s1 = get_dataset_subjects('MMgait')
s2 = get_dataset_subjects('nonan_young')
s3 = get_dataset_subjects('nonan_old')

s = [*s1, *s2, *s3]

print(len(s))

s_ = [f for f in os.listdir(file_path)]

print(len(s_))

print(sorted(s))
print(sorted(s_))
print(sorted(s_) == sorted(s))


