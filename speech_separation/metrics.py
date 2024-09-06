import tensorflow as tf
import numpy as np
from tpconverter import loss_acc_decorator

@loss_acc_decorator
def binary_accuracy_c(y_true, y_pred,*args,**kwargs):
	return tf.keras.metrics.binary_accuracy(y_true,y_pred)
