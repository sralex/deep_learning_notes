import numpy as np
import tensorflow as tf
from functools import wraps

def _generate_tp(y_true,y_pred):
    true = y_true[...,0:2]
    vad = y_true[...,2:3]
    mag = y_true[...,2:3]

    true = true * vad
    y_pred = y_pred * vad

    return true, y_pred, mag


def loss_acc_decorator(func, *args0, **kwargs0):
    @wraps(func)
    def wrapper(y_true,y_pred):
        y_true, y_pred , mag = _generate_tp(y_true,y_pred)
        return func(y_true, y_pred, mag)
    return wrapper