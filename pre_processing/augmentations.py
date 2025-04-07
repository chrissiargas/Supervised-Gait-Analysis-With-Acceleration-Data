import scipy.optimize as opt
from numpy.f2py.rules import defmod_rules
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, norm
import numpy as np
import pandas as pd
from pre_processing.filters import butter_lowpass_filter
from pre_processing.irregularities import get_gravity
from rotation_utils import rotation_by_axis
from scipy.spatial.transform import Rotation

def random_rotate(xyz, around: str = 'x'):
    angle = np.random.normal(loc=0, scale=15)
    angle = np.clip(angle, a_min=-45, a_max=45)
    R = rotation_by_axis(angle, around, degrees=True)
    xyz_rotated = Rotation.from_matrix(R).apply(xyz)

    return xyz_rotated


