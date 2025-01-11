import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.building import builder

b = builder()
data, _, _ = b()

for batch in data.take(1):
    X, Y = batch

    print(X.shape)
    x = X[0, :, 0]

    plt.plot(x)
    plt.show()

    sp = np.fft.fft(x)
    power_fft = np.abs(sp)
    print(power_fft.shape)

    plt.plot(power_fft)
    plt.show()