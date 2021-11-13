import numpy as np
import pandas as pd

# Data preprocessing functions.

# Town
median_price_town_rank = {'ANG MO KIO': 0.038461538461538464,
                          'YISHUN': 0.07692307692307693,
                          'BUKIT BATOK': 0.11538461538461539,
                          'BEDOK': 0.15384615384615385,
                          'WOODLANDS': 0.19230769230769232,
                          'GEYLANG': 0.23076923076923078,
                          'TOA PAYOH': 0.2692307692307692,
                          'SEMBAWANG': 0.3076923076923077,
                          'CHOA CHU KANG': 0.34615384615384615,
                          'JURONG WEST': 0.38461538461538464,
                          'JURONG EAST': 0.4230769230769231,
                          'HOUGANG': 0.46153846153846156,
                          'CLEMENTI': 0.5,
                          'BUKIT PANJANG': 0.5384615384615384,
                          'SENGKANG': 0.5769230769230769,
                          'TAMPINES': 0.6153846153846154,
                          'PUNGGOL': 0.6538461538461539,
                          'SERANGOON': 0.6923076923076923,
                          'MARINE PARADE': 0.7307692307692307,
                          'KALLANG/WHAMPOA': 0.7692307692307693,
                          'PASIR RIS': 0.8076923076923077,
                          'CENTRAL AREA': 0.8461538461538461,
                          'BUKIT MERAH': 0.8846153846153846,
                          'QUEENSTOWN': 0.9230769230769231,
                          'BISHAN': 0.9615384615384616,
                          'BUKIT TIMAH': 1.0}

def town_rank(x):
    return median_price_town_rank[x]

# Flat type
def flat_type_formatter(x):
    if x == '1 ROOM':
        res = 1/7
    elif x == '2 ROOM':
        res = 2/7
    elif x == '3 ROOM':
        res = 3/7
    elif x == '4 ROOM':
        res = 4/7
    elif x == '5 ROOM':
        res = 5/7
    elif x == 'EXECUTIVE':
        res = 6/7
    elif x == 'MULTI-GENERATION' or x == "MULTI GENERATION":
        res = 7/7
    else:
        res = 0
    return res

# Storey range
min_storey_range = 1
max_storey_range = 50

def storey_formatter(x):
    if 1 <= x <= 3:
        s = 0.04
    elif 4 <= x <= 6:
        s = 0.10
    elif 7 <= x <= 9:
        s = 0.16
    elif 10 <= x <= 12:
        s = 0.22
    elif 13 <= x <= 15:
        s = 0.28
    elif 16 <= x <= 18:
        s = 0.34
    elif 19 <= x <= 21:
        s = 0.40
    elif 22 <= x <= 24:
        s = 0.46
    elif 25 <= x <= 27:
        s = 0.52
    elif 28 <= x <= 30:
        s = 0.58
    elif 31 <= x <= 33:
        s = 0.64
    elif 34 <= x <= 36:
        s = 0.70
    elif 37 <= x <= 39:
        s = 0.76
    elif 40 <= x <= 42:
        s = 0.82
    elif 43 <= x <= 45:
        s = 0.88
    elif 46 <= x <= 48:
        s = 0.94
    elif 49 <= x <= 50:
        s = 1.00
    return s
        
# Floor area
min_floor_area = 31
max_floor_area = 280

def floor_area_scaler(x, xmin = min_floor_area, xmax = max_floor_area):
    return (np.log10(x) - np.log10(xmin)) / (np.log10(xmax) - np.log10(xmin))

# Age
min_age = 2 
max_age = 55

def age_scaler(x, xmin = min_age, xmax = max_age):
    return (x - xmin) / (xmax - xmin)

# Target post processing functions.

# Resale price
def y_descaler(y):
    return 10**y