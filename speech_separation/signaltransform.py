import numpy as np


def standarize_radians(x):
    """Docstring"""
    return (x * 2) / (2 * np.pi)


def reduce_standarize(x):
    return (x - 7.6765018) / 17.440527


def reduce_standarize_mag(x):
    return (x - 0.0014054106) / 0.0049425485


def db(x,MIN_AMP,AMP_FAC):
    """Docstring"""
    return 20. * np.log10(np.maximum(x, np.max(x) / MIN_AMP) * AMP_FAC)


def vad(x,threshold):
    """Docstring"""
    return (x > (np.max(x) - threshold)).astype(np.float32)
