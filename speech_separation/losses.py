import tensorflow as tf
import numpy as np
from functools import wraps
from tpconverter import loss_acc_decorator


@loss_acc_decorator
def MSE(y_true, y_pred, *args, **kwargs):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)