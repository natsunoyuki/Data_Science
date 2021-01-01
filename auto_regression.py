import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def auto_regression(y, delay = 10):
    """
    From a time series y, create an auto regression data set using a time delay
    
    Inputs
    ------
    y: np.array
        np.array of values to create the auto regression data
    delay: int
        delay to use

    Returns
    -------
    R: np.array
        auto regression data features created from y and delay
    Y: np.array
       auto regression data target created from y and delay
    """
    M = len(y)
    delay = int(delay)

    R = np.zeros([M - delay, delay])
    Y = y[delay:]
    i = delay
    while i < M:
        R[i - delay] = y[i - delay:i]
        i = i + 1
    return R, Y
